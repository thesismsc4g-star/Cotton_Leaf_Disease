import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

import config
from prompts import CLASS_NAMES, ORDERED_TEXT_PROMPTS
from src.data import CottonLeafDataset, build_transforms, list_image_paths
from src.modeling import ConvNeXtGCN_CLIP


class HybridCLIPLoss(nn.Module):
    def __init__(self, cls_w: float = 1.0, contrast_w: float = 1.0, proto_ce_w: float = 0.3, label_smoothing: float = 0.03):
        super().__init__()
        self.cls_w = cls_w
        self.contrast_w = contrast_w
        self.proto_ce_w = proto_ce_w
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, labels):
        class_logits = outputs["class_logits"]
        contrastive_logits = outputs["contrastive_logits"]

        cls_loss = self.ce(class_logits, labels)
        img_to_text = self.ce(contrastive_logits, labels)

        unique_labels = labels.unique(sorted=True)
        if len(unique_labels) > 1:
            text_protos = outputs["text_prototypes"][unique_labels]
            img_embeds = outputs["image_embed"]
            sim = text_protos @ img_embeds.t()

            targets = []
            label_list = labels.tolist()
            for ul in unique_labels.tolist():
                targets.append(label_list.index(ul))
            targets = torch.tensor(targets, device=labels.device, dtype=torch.long)
            text_to_img = self.ce(sim, targets)
        else:
            text_to_img = torch.tensor(0.0, device=labels.device)

        proto_ce = self.ce(contrastive_logits, labels)

        total = self.cls_w * cls_loss + self.contrast_w * 0.5 * (img_to_text + text_to_img) + self.proto_ce_w * proto_ce
        logs = {
            "loss_cls": cls_loss.item(),
            "loss_img2txt": img_to_text.item(),
            "loss_txt2img": text_to_img.item() if isinstance(text_to_img, torch.Tensor) else float(text_to_img),
            "loss_proto_ce": proto_ce.item(),
        }
        return total, logs


def set_backbone_trainable(model, trainable: bool = True) -> None:
    for p in model.image_encoder.features.parameters():
        p.requires_grad = trainable


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    freeze_epochs: int,
    model_path: Path,
    lam_adj_entropy: float,
    lam_adj_deviation: float,
    device,
):
    best_val_acc = 0.0
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    set_backbone_trainable(model, False)
    start_time = time.time()

    for epoch in range(num_epochs):
        if epoch == freeze_epochs:
            set_backbone_trainable(model, True)
            print(f"Unfroze backbone at epoch {epoch + 1}")

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss, _ = criterion(outputs, labels)
                adj_reg = model.image_encoder.adj_regularizer(
                    lam_entropy=lam_adj_entropy, lam_deviation=lam_adj_deviation
                )
                total_loss = loss + adj_reg

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            fused_logits = outputs["contrastive_logits"] + outputs["class_logits"]
            _, predicted = torch.max(fused_logits, 1)

            running_loss += total_loss.item()
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_predictions / total_samples

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss, _ = criterion(outputs, labels)
                    adj_reg = model.image_encoder.adj_regularizer(
                        lam_entropy=lam_adj_entropy, lam_deviation=lam_adj_deviation
                    )
                    total_loss = loss + adj_reg

                fused_logits = outputs["contrastive_logits"] + outputs["class_logits"]
                _, predicted = torch.max(fused_logits, 1)

                val_loss_total += total_loss.item()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss_total / len(val_loader)
        val_acc = 100 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%\n")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(config.DATASET_DIR))
    parser.add_argument("--model-path", default=str(config.MODEL_WEIGHTS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    all_image_paths, all_labels, class_names, _ = list_image_paths(dataset_dir)
    if class_names != CLASS_NAMES:
        raise ValueError("Dataset classes do not match prompts.py class names.")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.30, random_state=42, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    train_transform, eval_transform, _, _ = build_transforms()
    train_dataset = CottonLeafDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CottonLeafDataset(val_paths, val_labels, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(config.CLIP_MODEL_NAME)

    model = ConvNeXtGCN_CLIP(
        class_names=CLASS_NAMES,
        text_prompts=ORDERED_TEXT_PROMPTS,
        processor=processor,
        device=device,
        pretrained=True,
        gcn_hidden=256,
        dropout=0.3,
        freeze_backbone=False,
        freeze_text_encoder=True,
        cls_loss_weight=1.0,
        contrastive_weight=1.0,
        text_proto_cls_weight=0.3,
    ).to(device)

    criterion = HybridCLIPLoss(cls_w=1.0, contrast_w=0.8, proto_ce_w=0.4, label_smoothing=0.03)

    backbone_params = list(model.image_encoder.features.parameters())
    graph_and_proj_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("image_encoder.features.") and not n.startswith("clip_model.")
    ]
    text_params = [p for p in model.clip_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": graph_and_proj_params, "lr": 1e-4},
            {"params": text_params, "lr": 5e-6},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        model_path=Path(args.model_path),
        lam_adj_entropy=0.0,
        lam_adj_deviation=1e-4,
        device=device,
    )

    print(f"Best model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
