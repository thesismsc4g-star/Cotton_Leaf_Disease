from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", DATA_DIR / "Cotton_Augmented_Dataset"))
MODEL_WEIGHTS = Path(os.getenv("MODEL_WEIGHTS", BASE_DIR / "models" / "convnext_gcn_clip_cotton_leaf.pth"))
KB_DIR = Path(os.getenv("KB_DIR", BASE_DIR / "knowledge_base"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = (224, 224)
