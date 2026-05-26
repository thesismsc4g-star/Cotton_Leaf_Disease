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

import matplotlib.cm as cm

import gdown

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

    print("Downloading model...")

    gdown.download(
        url,
        model_path,
        quiet=False
    )

    print("Download complete!")

# ==========================================
# PREDICTOR
# ==========================================
class CottonLeafPredictor:

    def __init__(self, weights_path: str):

        download_model_if_needed(weights_path)

        self.device = torch.device(
            config.DEVICE
        )

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

        print("Model loaded successfully")

    # ======================================
    # PREDICT
    # ======================================
    def predict(self, image_pil: Image.Image) -> Dict:

        x = load_image_for_model(
            image_pil,
            self.eval_transform
        ).to(self.device)

        with torch.enable_grad():

            x.requires_grad_(True)

            outputs = self.model(x)

            logits = (
                outputs["contrastive_logits"]
                + outputs["class_logits"]
            )

            probs = torch.softmax(
                logits,
                dim=1
            )[0].detach().cpu().numpy()

            pred_idx = int(
                np.argmax(probs)
            )

            class_name = CLASS_NAMES[pred_idx]

            gradcam = self.generate_gradcam(
                image_pil,
                x,
                pred_idx
            )

        return {

            "class_name": class_name,

            "class_label": pretty_class_name(
                class_name
            ),

            "probabilities": probs,

            "pred_idx": pred_idx,

            "gradcam": gradcam,
        }

    # ======================================
    # GRADCAM
    # ======================================
    def generate_gradcam(
        self,
        image_pil,
        x,
        pred_idx
    ):

        features = []
        gradients = []

        # ==================================
        # HOOKS
        # ==================================
        def forward_hook(module, inp, out):

            features.append(out)

        def backward_hook(module, grad_in, grad_out):

            gradients.append(
                grad_out[0]
            )

        target_layer = (
            self.model.image_encoder.features[-1]
        )

        fh = target_layer.register_forward_hook(
            forward_hook
        )

        bh = target_layer.register_full_backward_hook(
            backward_hook
        )

        # ==================================
        # FORWARD
        # ==================================
        outputs = self.model(x)

        logits = (
            outputs["contrastive_logits"]
            + outputs["class_logits"]
        )

        score = logits[:, pred_idx]

        self.model.zero_grad()

        # ==================================
        # BACKWARD
        # ==================================
        score.backward(
            retain_graph=True
        )

        fmap = features[0]

        grads = gradients[0]

        # ==================================
        # WEIGHTS
        # ==================================
        weights = grads.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ==================================
        # CAM
        # ==================================
        cam = (
            weights * fmap
        ).sum(dim=1).squeeze()

        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        fh.remove()
        bh.remove()

        # ==================================
        # RESIZE
        # ==================================
        cam_img = Image.fromarray(
            np.uint8(cam * 255)
        ).resize(config.IMAGE_SIZE)

        cam_img = np.array(
            cam_img
        ) / 255.0

        heatmap = cm.jet(
            cam_img
        )[:, :, :3]

        rgb = np.array(
            image_pil.resize(
                config.IMAGE_SIZE
            )
        ) / 255.0

        overlay = (
            0.6 * rgb
            + 0.4 * heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            1
        )

        return np.uint8(
            overlay * 255
        )

# ==========================================
# LOAD
# ==========================================
def load_predictor(weights_path: str):

    return CottonLeafPredictor(
        weights_path
    )
