# ===== config.py =====

from pathlib import Path
import os

# ==============================
# BASE DIRECTORY
# ==============================
BASE_DIR = Path(__file__).resolve().parent

# ==============================
# MODEL PATH
# ==============================
MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    str(BASE_DIR / "models" / "cotton_model.pth")
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
# PERFORMANCE
# ==============================
DEVICE = "cpu"

NUM_WORKERS = 0

# ==============================
# DEBUG
# ==============================
DEBUG = False

print("CONFIG LOADED")
print("MODEL:", MODEL_WEIGHTS)
print("DEVICE:", DEVICE)
