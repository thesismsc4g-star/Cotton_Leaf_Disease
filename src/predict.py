# ===== predict.py =====

from typing import Dict
import os
import gdown
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from transformers import AutoProcessor

try:
    import cv2
except Exception:
    cv2 = None

from lime import lime_image
from skimage.segmentation import mark_boundaries

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from prompts import CLASS_NAMES, ORDERED_TEXT_PROMPTS, pretty_class_name
from src.data import build_transforms, load_image_for_model
from src.modeling import ConvNeXtGCN_CLIP


# ==============================
# DOWNLOAD MODEL
# ==============================
def download_model_if_needed(model_path: str):

    if not os.path.exists(model_path):

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        FILE_ID = "19hs4HeRAOZ3UqQ2M9y8va2BMPvjFK19J"

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        print("⬇️ Downloading model...")
        gdown.download(url, model_path, quiet=False)
        print("✅ Download complete!")


# ==============================
# GRADCAM WRAPPER
# ==============================
class GradCAMWrapper(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model

    def forward(self, x):

        outputs = self.model(x)

        logits = (
            outputs["contrastive_logits"]
            + outputs["class_logits"]
        )

        return logits


# ==============================
# MAIN PREDICTOR
# ==============================
class CottonLeafPredictor:

    def __init__(self, weights_path: str):

        download_model_if_needed(weights_path)

        self.device = torch.device("cpu")

        self.processor = AutoProcessor.from_pretrained(
            config.CLIP_MODEL_NAME,
            trust_remote_code=True
        )

        _, self.eval_transform, _, _ = build_transforms()

        self.model = ConvNeXtGCN_CLIP(
            class_names=CLASS_NAMES,
            text_prompts=ORDERED_TEXT_PROMPTS,
            processor=self.processor,
            device=self.device,
            pretrained=False,
            gcn_hidden=256,
            dropout=0.1,
            freeze_backbone=True,
            freeze_text_encoder=True,
            cls_loss_weight=1.0,
            contrastive_weight=1.0,
            text_proto_cls_weight=0.3,
        )

        state = torch.load(
            weights_path,
            map_location="cpu"
        )

        self.model.load_state_dict(
            state,
            strict=False
        )

        self.model.to(self.device)

        self.model.eval()

        print("✅ Model loaded successfully")

        # =========================
        # 🔥 GRADCAM SETUP
        # =========================
        self.cam_model = GradCAMWrapper(self.model)

        self.target_layers = [
            self.model.image_encoder.features[-1]
        ]

        self.cam = GradCAM(
            model=self.cam_model,
            target_layers=self.target_layers
        )

    # =========================
    # PREDICT
    # =========================
    def predict(self, image_pil: Image.Image) -> Dict:

        x = load_image_for_model(
            image_pil,
            self.eval_transform
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model(x)

            logits = (
                outputs["contrastive_logits"]
                + outputs["class_logits"]
            )

            probs = torch.softmax(
                logits,
                dim=1
            )[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))

        class_name = CLASS_NAMES[pred_idx]

        # =========================
        # 🔥 GRADCAM
        # =========================
        gradcam_overlay = self.generate_gradcam(
            image_pil,
            x
        )

        # =========================
        # 🔥 LIME
        # =========================
        lime_overlay = self.generate_lime(
            image_pil
        )

        return {

            "class_name": class_name,

            "class_label": pretty_class_name(class_name),

            "probabilities": probs,

            "pred_idx": pred_idx,

            "gradcam": gradcam_overlay,

            "lime": lime_overlay,
        }

    # =========================
    # 🔥 GRADCAM
    # =========================
    def generate_gradcam(
        self,
        image_pil,
        x
    ):

        rgb = np.array(
            image_pil.resize(config.IMAGE_SIZE)
        ).astype(np.float32) / 255.0

        grayscale_cam = self.cam(
            input_tensor=x
        )[0]

        visualization = show_cam_on_image(
            rgb,
            grayscale_cam,
            use_rgb=True
        )

        return visualization

    # =========================
    # 🔥 LIME
    # =========================
    def generate_lime(
        self,
        image_pil
    ):

        image = np.array(
            image_pil.resize(config.IMAGE_SIZE)
        )

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            image,
            self.lime_predict,
            top_labels=1,
            hide_color=0,
            num_samples=100
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        lime_img = mark_boundaries(
            temp / 255.0,
            mask
        )

        lime_img = (
            lime_img * 255
        ).astype(np.uint8)

        return lime_img

    # =========================
    # LIME PREDICT FUNCTION
    # =========================
    def lime_predict(
        self,
        images
    ):

        batch = []

        for img in images:

            pil = Image.fromarray(
                img.astype(np.uint8)
            )

            x = load_image_for_model(
                pil,
                self.eval_transform
            )

            batch.append(x)

        batch = torch.cat(batch).to(self.device)

        with torch.no_grad():

            outputs = self.model(batch)

            logits = (
                outputs["contrastive_logits"]
                + outputs["class_logits"]
            )

            probs = F.softmax(
                logits,
                dim=1
            )

        return probs.cpu().numpy()


def load_predictor(weights_path: str):

    return CottonLeafPredictor(weights_path)
