from typing import Dict
import os
import gdown

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

# 🔥 Safe OpenCV import
try:
    import cv2
except ImportError:
    cv2 = None

import config
from prompts import CLASS_NAMES, ORDERED_TEXT_PROMPTS, pretty_class_name
from src.data import build_transforms, load_image_for_model
from src.modeling import ConvNeXtGCN_CLIP


# ==============================
# 🔥 GOOGLE DRIVE DOWNLOAD
# ==============================
def download_model_if_needed(model_path: str):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 👉 🔴 IMPORTANT: Replace with your file ID
        FILE_ID = "YOUR_FILE_ID"
        url = f"https://drive.google.com/uc?id={FILE_ID}"

        print("⬇️ Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        print("✅ Model download complete!")


# ==============================
# 🧠 PREDICTOR CLASS
# ==============================
class CottonLeafPredictor:
    def __init__(self, weights_path: str) -> None:

        # ✅ Download model if missing
        download_model_if_needed(weights_path)

        print("MODEL PATH:", weights_path)
        print("FILE EXISTS:", os.path.exists(weights_path))

        # 🔥 FORCE CPU (Streamlit Cloud safe)
        self.device = torch.device("cpu")

        # 🔥 Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.CLIP_MODEL_NAME
        )

        _, self.eval_transform, _, _ = build_transforms()

        # 🔥 MEMORY OPTIMIZED MODEL
        self.model = ConvNeXtGCN_CLIP(
            class_names=CLASS_NAMES,
            text_prompts=ORDERED_TEXT_PROMPTS,
            processor=self.processor,
            device=self.device,
            pretrained=False,          # ❗ IMPORTANT
            gcn_hidden=128,            # reduced
            dropout=0.1,               # reduced
            freeze_backbone=True,      # ❗ IMPORTANT
            freeze_text_encoder=True,
            cls_loss_weight=1.0,
            contrastive_weight=1.0,
            text_proto_cls_weight=0.3,
        )

        # 🔥 LOAD WEIGHTS SAFELY (LOW RAM)
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        # 🔥 DISABLE GRADIENTS (CRITICAL)
        for param in self.model.parameters():
            param.requires_grad = False

        torch.set_grad_enabled(False)

        print("✅ Model loaded successfully")

    # ==============================
    # 🔍 PREDICTION
    # ==============================
    @torch.no_grad()
    def predict(self, image_pil: Image.Image) -> Dict:

        x = load_image_for_model(image_pil, self.eval_transform).to(self.device)

        outputs = self.model(x, return_attention=True)

        logits = outputs["contrastive_logits"] + outputs["class_logits"]

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        class_name = CLASS_NAMES[pred_idx]

        # 🔥 Attention normalize
        attention = outputs["node_attention"][0].reshape(7, 7).cpu().numpy()
        attention = (attention - attention.min()) / (
            attention.max() - attention.min() + 1e-8
        )

        overlay = self._make_attention_overlay(image_pil, attention)

        return {
            "class_name": class_name,
            "class_label": pretty_class_name(class_name),
            "probabilities": probs,
            "pred_idx": pred_idx,
            "overlay": overlay,
        }

    # ==============================
    # 🎨 ATTENTION OVERLAY
    # ==============================
    def _make_attention_overlay(
        self, image_pil: Image.Image, attention: np.ndarray
    ) -> np.ndarray:

        rgb = (
            np.array(image_pil.convert("RGB").resize(config.IMAGE_SIZE))
            .astype(np.float32)
            / 255.0
        )

        # ❗ fallback if cv2 না থাকে
        if cv2 is None:
            return (rgb * 255).astype(np.uint8)

        try:
            attn_resized = cv2.resize(
                attention,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            heatmap = cv2.applyColorMap(
                np.uint8(255 * attn_resized), cv2.COLORMAP_JET
            )

            heatmap = (
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )

            overlay = np.clip(0.6 * rgb + 0.4 * heatmap, 0, 1)

            return (overlay * 255).astype(np.uint8)

        except Exception:
            return (rgb * 255).astype(np.uint8)


# ==============================
# 🚀 LOADER FUNCTION
# ==============================
def load_predictor(weights_path: str) -> CottonLeafPredictor:
    return CottonLeafPredictor(weights_path)
