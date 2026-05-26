# ===== config.py =====

from pathlib import Path
import os
import torch


# ==============================
# BASE DIR
# ==============================
BASE_DIR = Path(__file__).resolve().parent


# ==============================
# DATA PATHS
# ==============================
DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        BASE_DIR / "data"
    )
)

DATASET_DIR = Path(
    os.getenv(
        "DATASET_DIR",
        DATA_DIR / "Cotton_Augmented_Dataset"
    )
)


# ==============================
# MODEL PATH
# ==============================
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    str(MODEL_DIR / "cotton_model.pth")
)


# ==============================
# CLIP MODEL
# ==============================
CLIP_MODEL_NAME = os.getenv(
    "CLIP_MODEL_NAME",
    "openai/clip-vit-base-patch32"
)


# ==============================
# IMAGE SETTINGS
# ==============================
IMAGE_SIZE = (224, 224)


# ==============================
# DEVICE
# ==============================
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==============================
# PERFORMANCE
# ==============================
NUM_WORKERS = 0


# ==============================
# DEBUG
# ==============================
DEBUG = os.getenv(
    "DEBUG",
    "False"
) == "True"


# ==============================
# STARTUP LOG
# ==============================
print("=" * 50)
print("🔧 CONFIG LOADED")
print("BASE_DIR:", BASE_DIR)
print("MODEL PATH:", MODEL_WEIGHTS)
print("DEVICE:", DEVICE)
print("=" * 50)
