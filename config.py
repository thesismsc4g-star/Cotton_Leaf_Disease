from pathlib import Path
import os

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# 📂 DATA PATHS
# -----------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

DATASET_DIR = Path(
    os.getenv("DATASET_DIR", DATA_DIR / "Cotton_Augmented_Dataset")
)

# -----------------------------
# 🧠 MODEL PATH
# -----------------------------
MODEL_WEIGHTS = os.getenv(
    "MODEL_WEIGHTS",
    str(BASE_DIR / "models" / "cotton_model.pth"),  # 🔥 renamed clean
)

# -----------------------------
# 📚 KNOWLEDGE BASE
# -----------------------------
KB_DIR = Path(
    os.getenv("KB_DIR", BASE_DIR / "knowledge_base")
)

# -----------------------------
# 🤖 GROQ CONFIG
# -----------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# -----------------------------
# 🧩 CLIP CONFIG
# -----------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# -----------------------------
# 🖼️ IMAGE CONFIG
# -----------------------------
IMAGE_SIZE = (224, 224)
