# ===== predict.py =====

from typing import Dict
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from transformers import AutoProcessor

# ==========================================
# SAFE OPENCV IMPORT
# ==========================================
try:
    import cv2
except Exception:
    cv2 = None

# ==========================================
# LIME
# ==========================================
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==========================================
# GRADCAM
# ==========================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# DOWNLOAD
# ==========================================
import gdown

# ==========================================
# LOCAL IMPORTS
# ==========================================
import config

from prompts import (
    CLASS_NAMES,
    ORDERED_TEXT_PROMPTS,
    pretty_class_name
)

from src.data import (
    build_transforms,
    load_image_for_model
)

from src.modeling import (
    ConvNeXtGCN_CLIP
)

# ==========================================
# DOWNLOAD MODEL
# ==========================================
def download_model_if_needed(model_path: str):

    if os.path.exists(model_path):
        return

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    FILE_ID = "19hs4HeRAOZ3UqQ2M9y8va2BMPvjFK19J"

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("⬇️ Downloading model...")

    gdown.download(
        url,
        model_path,
        quiet=False
    )

    print("✅ Download complete!")


# ==========================================
# PREDICTOR
# ==========================================
class CottonLeafPredictor:

    def __init__(self, weights_path: str):

        # ==================================
        # DOWNLOAD MODEL
        # ==================================
        download_model_if_needed(
            weights_path
        )

        print("MODEL PATH:", weights_path)

        self.device = torch.device(
            config.DEVICE
        )

        # ==================================
        # PROCESSOR
        # ==================================
        self.processor = AutoProcessor.from_pretrained(
            config.CLIP_MODEL_NAME,
            trust_remote_code=True
        )

        # ==================================
        # TRANSFORMS
        # ==================================
        _, self.eval_transform, _, _ = build_transforms()

        # ==================================
        # MODEL
        # ==================================
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

        # ==================================
        # LOAD WEIGHTS
        # ==================================
        state = torch.load(
            weights_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            state,
            strict=False
        )

        self.model.to(self.device)

        self.model.eval()

        print("✅ Model loaded successfully")

        # ==================================
        # TARGET LAYER FOR GRADCAM
        # ==================================
        self.target_layers = [
            self.model.image_encoder.backbone.features[-1]
        ]

    # ======================================
    # PREDICT
    # ======================================
    @torch.no_grad()
    def predict(self, image_pil: Image.Image) -> Dict:

        # ==================================
        # IMAGE TO TENSOR
        # ==================================
        x = load_image_for_model(
            image_pil,
            self.eval_transform
        ).to(self.device)

        # ==================================
        # FORWARD
        # ==================================
        outputs = self.model(x)

        logits = (
            outputs["contrastive_logits"]
            + outputs["class_logits"]
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

        pred_idx = int(
            np.argmax(probs)
        )

        class_name = CLASS_NAMES[pred_idx]

        # ==================================
        # GRADCAM
        # ==================================
        gradcam_image = self.generate_gradcam(
            image_pil,
            x,
            pred_idx
        )

        # ==================================
        # LIME
        # ==================================
        lime_image_result = self.generate_lime(
            image_pil
        )

        return {

            "class_name": class_name,

            "class_label": pretty_class_name(
                class_name
            ),

            "probabilities": probs,

            "pred_idx": pred_idx,

            "gradcam": gradcam_image,

            "lime": lime_image_result,
        }

    # ======================================
    # FORWARD FOR CAM
    # ======================================
    def cam_forward(self, x):

        outputs = self.model(x)

        logits = (
            outputs["contrastive_logits"]
            + outputs["class_logits"]
        )

        return logits

    # ======================================
    # GENERATE GRADCAM
    # ======================================
    def generate_gradcam(
        self,
        image_pil,
        x,
        pred_idx
    ):

        rgb_img = (
            np.array(
                image_pil.resize(
                    config.IMAGE_SIZE
                )
            ).astype(np.float32) / 255.0
        )

        try:

            cam = GradCAM(
                model=self,
                target_layers=self.target_layers
            )

            targets = [
                ClassifierOutputTarget(
                    pred_idx
                )
            ]

            grayscale_cam = cam(
                input_tensor=x,
                targets=targets
            )[0]

            visualization = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True
            )

            return visualization

        except Exception as e:

            print("GradCAM Error:", e)

            return (
                rgb_img * 255
            ).astype(np.uint8)

    # ======================================
    # GENERATE LIME
    # ======================================
    def generate_lime(self, image_pil):

        image = np.array(
            image_pil.resize(
                config.IMAGE_SIZE
            )
        )

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            image,
            self.batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=30
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        lime_result = mark_boundaries(
            temp / 255.0,
            mask
        )

        lime_result = (
            lime_result * 255
        ).astype(np.uint8)

        return lime_result

    # ======================================
    # BATCH PREDICT FOR LIME
    # ======================================
    def batch_predict(self, images):

        self.model.eval()

        batch = []

        for img in images:

            pil = Image.fromarray(
                img.astype(np.uint8)
            )

            tensor = load_image_for_model(
                pil,
                self.eval_transform
            )

            batch.append(tensor)

        batch = torch.cat(
            batch,
            dim=0
        ).to(self.device)

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

    # ======================================
    # REQUIRED FOR GRADCAM
    # ======================================
    def __call__(self, x):

        return self.cam_forward(x)


# ==========================================
# LOAD PREDICTOR
# ==========================================
def load_predictor(weights_path: str):

    return CottonLeafPredictor(
        weights_path
    )
