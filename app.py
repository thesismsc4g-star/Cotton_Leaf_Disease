import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import gdown

import config
from prompts import CLASS_NAMES, pretty_class_name
from src.groq_client import generate_response
from src.predict import load_predictor
from src.rag import build_rag_index

load_dotenv()

st.set_page_config(page_title="Cotton Leaf Disease Classifier", layout="wide")

st.title("ConvNeXt-GCN based CLIP for Cotton Leaf Disease Classification")
st.write(
    "Upload a cotton leaf image, get the predicted disease, and retrieve management guidance."
)

# ==============================
# 🔥 MODEL DOWNLOAD FUNCTION
# ==============================
def download_model_if_needed():
    model_path = Path(config.MODEL_WEIGHTS)

    if not model_path.exists():
        os.makedirs(model_path.parent, exist_ok=True)

        file_id = "1mWD-qjyaE8Ti6JK3VsqcGDe9i1daEX9E"
        url = f"https://drive.google.com/uc?id={file_id}"

        with st.spinner("Downloading model from Google Drive... ⏳"):
            gdown.download(url, str(model_path), quiet=False)

    return model_path


# ✅ Load model path (auto-download)
@st.cache_resource
def get_model_path():
    return download_model_if_needed()


weights_path = get_model_path()
kb_dir = Path(os.getenv("KB_DIR", config.KB_DIR))

# ==============================
# 🔍 KEYWORDS
# ==============================
CAUSE_KEYWORDS = (
    "cause", "causes", "caused", "pathogen", "fungus", "fungal",
    "bacteria", "bacterial", "infection", "infected", "due to", "spread",
)

PREVENT_KEYWORDS = (
    "prevent", "prevention", "control", "management", "manage",
    "treat", "treatment", "spray", "apply", "recommended", "avoid",
    "sanitation", "remove", "rotate", "resistant", "fungicide", "bactericide",
)

if not kb_dir.exists():
    st.warning(f"Knowledge base directory not found: {kb_dir}")

# ==============================
# ⚡ CACHE MODELS
# ==============================
@st.cache_resource
def get_predictor(path: Path):
    return load_predictor(str(path))


@st.cache_resource
def get_rag_index(path: Path):
    return build_rag_index(path)


def groq_enabled() -> bool:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    enabled = os.getenv("ENABLE_GROQ", "true").strip().lower()
    return bool(api_key) and enabled not in {"0", "false", "no", "off"}


# ==============================
# 🧠 TEXT PROCESSING
# ==============================
def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def rule_based_summary(context: str) -> dict:
    cause, prevent, other = [], [], []

    for sentence in _split_sentences(context):
        lower = sentence.lower()
        is_cause = any(k in lower for k in CAUSE_KEYWORDS)
        is_prevent = any(k in lower for k in PREVENT_KEYWORDS)

        if is_cause and not is_prevent:
            cause.append(sentence)
        elif is_prevent and not is_cause:
            prevent.append(sentence)
        elif is_cause and is_prevent:
            prevent.append(sentence)
        else:
            other.append(sentence)

    return {
        "cause": cause or other,
        "prevention": prevent or other,
    }


# ==============================
# 🎨 UI
# ==============================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload leaf image", type=["jpg", "jpeg", "png", "webp"]
    )

with col2:
    user_question = st.text_area(
        "Ask about disease cause and prevention",
        value="What causes this disease and how to prevent it?",
        height=120,
    )


# ==============================
# 🚀 MAIN PIPELINE
# ==============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Running model..."):
        predictor = get_predictor(weights_path)
        result = predictor.predict(image)

    # ------------------------------
    # 📊 Prediction
    # ------------------------------
    st.subheader("Prediction")
    pred_label = result["class_label"]
    pred_idx = result["pred_idx"]
    probs = result["probabilities"]

    st.write(f"**{pred_label}** ({probs[pred_idx] * 100:.2f}% confidence)")

    topk = sorted(
        [(CLASS_NAMES[i], probs[i]) for i in range(len(CLASS_NAMES))],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    st.table(
        {
            "Class": [pretty_class_name(c) for c, _ in topk],
            "Confidence": [f"{p * 100:.2f}%" for _, p in topk],
        }
    )

    # ------------------------------
    # 🔥 Attention Map
    # ------------------------------
    st.subheader("Text-conditioned Attention Overlay")
    st.image(result["overlay"], use_container_width=True)

    # ------------------------------
    # 📚 RAG + Groq
    # ------------------------------
    st.subheader("Management Guidance (RAG + Groq)")

    rag_index = get_rag_index(kb_dir)
    query = f"{pred_label}. {user_question}"
    chunks = rag_index.query(query, top_k=3)

    if not chunks:
        st.info("No knowledge base documents found.")
    else:
        context = "\n\n".join(
            [f"Source: {c.source}\n{c.text}" for c in chunks]
        )

        groq_model = os.getenv("GROQ_MODEL", config.GROQ_MODEL)

        if groq_enabled():
            try:
                response = generate_response(
                    groq_model,
                    [
                        {
                            "role": "system",
                            "content": "You are an agronomy assistant. Provide clear cause and prevention steps.",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Disease: {pred_label}\n\n"
                                f"Question: {user_question}\n\n"
                                f"Context:\n{context}"
                            ),
                        },
                    ],
                )
                st.write(response)

            except Exception as exc:
                st.error(f"Groq request failed: {exc}")

        else:
            summary = rule_based_summary(context)

            st.subheader("Cause")
            st.markdown("\n".join(f"- {s}" for s in summary["cause"]))

            st.subheader("Prevention / Management")
            st.markdown("\n".join(f"- {s}" for s in summary["prevention"]))

            with st.expander("Raw Context"):
                st.write(context)
