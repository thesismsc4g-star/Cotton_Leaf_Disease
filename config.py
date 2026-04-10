# ===== config.py =====

from pathlib import Path
import os


# ==============================
# 📂 BASE DIRECTORY
# ==============================
BASE_DIR = Path(__file__).resolve().parent


# ==============================
# 📂 DATA PATHS
# ==============================
DATA_DIR = Path(
    os.getenv("DATA_DIR", BASE_DIR / "data")
)

DATASET_DIR = Path(
    os.getenv("DATASET_DIR", DATA_DIR / "Cotton_Augmented_Dataset")
)


# ==============================
# 🧠 MODEL PATH (IMPORTANT)
# ==============================
MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    str(BASE_DIR / "models" / "cotton_model.pth")
)


# ==============================
# 📚 KNOWLEDGE BASE (OPTIONAL)
# ==============================
KB_DIR = Path(
    os.getenv("KB_DIR", BASE_DIR / "knowledge_base")
)


# ==============================
# 🤖 GROQ CONFIG (SAFE)
# ==============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama3-70b-8192"
)


# ==============================
# 🧩 CLIP CONFIG
# ==============================
CLIP_MODEL_NAME = os.getenv(
    "CLIP_MODEL_NAME",
    "openai/clip-vit-base-patch32"
)


# ==============================
# 🖼️ IMAGE CONFIG
# ==============================
IMAGE_SIZE = (224, 224)


# ==============================
# ⚡ PERFORMANCE SETTINGS
# ==============================
DEVICE = "cpu"   # 🔥 Streamlit Cloud safe
NUM_WORKERS = 0  # 🔥 avoid multiprocessing crash


# ==============================
# 🧪 DEBUG FLAGS (OPTIONAL)
# ==============================
DEBUG = os.getenv("DEBUG", "False") == "True"


# ==============================
# ✅ STARTUP LOG
# ==============================
print("🔧 CONFIG LOADED")
print("MODEL PATH:", MODEL_WEIGHTS)
print("DEVICE:", DEVICE)
