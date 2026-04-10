import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import CLIPModel


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x_prop = torch.matmul(A, x)
        return self.lin(x_prop)


def make_grid_adjacency_norm(h: int = 7, w: int = 7, device: str = "cpu") -> torch.Tensor:
    n = h * w
    A = torch.zeros((n, n), dtype=torch.float32)

    def idx(r: int, c: int) -> int:
        return r * w + c

    for r in range(h):
        for c in range(w):
            u = idx(r, c)
            A[u, u] = 1.0
            if r > 0:
                A[u, idx(r - 1, c)] = 1.0
            if r < h - 1:
                A[u, idx(r + 1, c)] = 1.0
            if c > 0:
                A[u, idx(r, c - 1)] = 1.0
            if c < w - 1:
                A[u, idx(r, c + 1)] = 1.0

    deg = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg + 1e-8, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.to(device)


class ConvNeXtGCNImageEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        gcn_hidden: int = 256,
        proj_dim: int = 512,
        freeze_backbone: bool = False,
        device: str = "cpu",
        symmetrize: bool = True,
        init_scale: float = 4.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = backbone.features
        feat_dim = 768

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.h, self.w = 7, 7
        self.N = self.h * self.w

        self.gcn1 = DenseGCNLayer(feat_dim, gcn_hidden)
        self.gcn2 = DenseGCNLayer(gcn_hidden, gcn_hidden)
        self.norm1 = nn.LayerNorm(gcn_hidden)
        self.norm2 = nn.LayerNorm(gcn_hidden)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            A_init = make_grid_adjacency_norm(self.h, self.w, device=device)
            A_init = A_init / (A_init.sum(dim=-1, keepdim=True) + 1e-12)
            A_logits_init = torch.log(A_init + 1e-8) * init_scale

        self.A_logits = nn.Parameter(A_logits_init)
        self.symmetrize = symmetrize

        self.node_proj = nn.Linear(gcn_hidden, proj_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, proj_dim),
        )

    def _get_A_norm(self) -> torch.Tensor:
        A = F.softmax(self.A_logits, dim=-1)
        if self.symmetrize:
            A = 0.5 * (A + A.transpose(0, 1))
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
        return A

    def forward(
        self, x: torch.Tensor, return_nodes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.features(x)
        B, C, H, W = f.shape
        if (H, W) != (self.h, self.w):
            raise ValueError(f"Expected {(self.h, self.w)}, got {(H, W)}")

        nodes = f.view(B, C, H * W).transpose(1, 2)
        A_norm = self._get_A_norm()

        z = self.gcn1(nodes, A_norm)
        z = self.norm1(z)
        z = F.gelu(z)

        z = self.gcn2(z, A_norm)
        z = self.norm2(z)
        z = F.gelu(z)

        pooled = z.mean(dim=1)
        pooled = self.dropout(pooled)

        image_embed = self.global_proj(pooled)
        image_embed = F.normalize(image_embed, dim=-1)

        if return_nodes:
            node_embed = self.node_proj(z)
            node_embed = F.normalize(node_embed, dim=-1)
            return image_embed, node_embed, pooled, z, A_norm

        return image_embed, pooled, z, A_norm

    def adj_regularizer(self, lam_entropy: float = 0.0, lam_deviation: float = 0.0) -> torch.Tensor:
        reg = torch.tensor(0.0, device=self.A_logits.device)
        if lam_entropy > 0:
            A = F.softmax(self.A_logits, dim=-1)
            ent = (-A * (A.add(1e-12).log())).sum(dim=-1).mean()
            reg = reg - lam_entropy * ent
        if lam_deviation > 0:
            reg = reg + lam_deviation * (self.A_logits ** 2).mean()
        return reg


class ConvNeXtGCN_CLIP(nn.Module):
    def __init__(
        self,
        class_names: List[str],
        text_prompts: Dict[str, List[str]],
        processor,
        device: torch.device,
        pretrained: bool = True,
        gcn_hidden: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        freeze_text_encoder: bool = True,
        cls_loss_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        text_proto_cls_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.class_names = list(class_names)
        self.text_prompts = text_prompts
        self.device = device
        self.processor = processor
        self.num_classes = len(class_names)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_embed_dim = self.clip_model.config.projection_dim

        if freeze_text_encoder:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        self.image_encoder = ConvNeXtGCNImageEncoder(
            pretrained=pretrained,
            gcn_hidden=gcn_hidden,
            proj_dim=self.text_embed_dim,
            freeze_backbone=freeze_backbone,
            device=str(device),
            symmetrize=True,
            init_scale=4.0,
            dropout=dropout,
        )

        self.classifier = nn.Linear(gcn_hidden, self.num_classes)
        self.prompt_attn_q = nn.Linear(self.text_embed_dim, self.text_embed_dim)
        self.prompt_attn_k = nn.Linear(self.text_embed_dim, self.text_embed_dim)
        self.prompt_attn_v = nn.Linear(self.text_embed_dim, self.text_embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cls_loss_weight = cls_loss_weight
        self.contrastive_weight = contrastive_weight
        self.text_proto_cls_weight = text_proto_cls_weight

        all_prompts: List[str] = []
        self.num_prompts_per_class: List[int] = []
        for cls in self.class_names:
            prompts = self.text_prompts[cls]
            self.num_prompts_per_class.append(len(prompts))
            all_prompts.extend(prompts)

        text_batch = self.processor(
            text=all_prompts, return_tensors="pt", padding=True, truncation=True
        )
        self.register_buffer("input_ids", text_batch["input_ids"])
        self.register_buffer("attention_mask", text_batch["attention_mask"])

    def encode_prompt_embeddings(self) -> torch.Tensor:
        text_outputs = self.clip_model.text_model(
            input_ids=self.input_ids, attention_mask=self.attention_mask
        )
        text_features = self.clip_model.text_projection(text_outputs.pooler_output)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_text_prototypes(self) -> torch.Tensor:
        prompt_embeds = self.encode_prompt_embeddings()
        prototypes = []
        start = 0
        for n in self.num_prompts_per_class:
            cls_prompts = prompt_embeds[start : start + n]
            query = self.prompt_attn_q(cls_prompts.mean(dim=0, keepdim=True))
            keys = self.prompt_attn_k(cls_prompts)
            vals = self.prompt_attn_v(cls_prompts)
            scores = (query @ keys.t()) / math.sqrt(keys.shape[-1])
            weights = torch.softmax(scores, dim=-1)
            proto = weights @ vals
            proto = F.normalize(proto.squeeze(0), dim=-1)
            prototypes.append(proto)
            start += n
        return torch.stack(prototypes, dim=0)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        image_embed, node_embed, pooled, _, A_norm = self.image_encoder(x, return_nodes=True)
        class_logits = self.classifier(pooled)

        text_prototypes = self.encode_text_prototypes()
        sim_logits = image_embed @ text_prototypes.t()
        sim_logits = sim_logits * self.logit_scale.exp()

        if return_attention:
            pred_idx = sim_logits.argmax(dim=1)
            chosen_text = text_prototypes[pred_idx]
            node_scores = torch.einsum("bnd,bd->bn", node_embed, chosen_text)
            node_attn = torch.softmax(node_scores, dim=-1)
            return {
                "class_logits": class_logits,
                "contrastive_logits": sim_logits,
                "image_embed": image_embed,
                "node_embed": node_embed,
                "graph_features": pooled,
                "adjacency": A_norm,
                "text_prototypes": text_prototypes,
                "node_attention": node_attn,
                "pred_idx": pred_idx,
            }

        return {
            "class_logits": class_logits,
            "contrastive_logits": sim_logits,
            "image_embed": image_embed,
            "graph_features": pooled,
            "adjacency": A_norm,
            "text_prototypes": text_prototypes,
        }
