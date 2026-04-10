import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

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

weights_path = Path(os.getenv("MODEL_WEIGHTS", config.MODEL_WEIGHTS))
kb_dir = Path(os.getenv("KB_DIR", config.KB_DIR))

CAUSE_KEYWORDS = (
    "cause",
    "causes",
    "caused",
    "pathogen",
    "fungus",
    "fungal",
    "bacteria",
    "bacterial",
    "infection",
    "infected",
    "due to",
    "spread",
)
PREVENT_KEYWORDS = (
    "prevent",
    "prevention",
    "control",
    "management",
    "manage",
    "treat",
    "treatment",
    "spray",
    "apply",
    "recommended",
    "avoid",
    "sanitation",
    "remove",
    "rotate",
    "resistant",
    "fungicide",
    "bactericide",
)

if not weights_path.exists():
    st.warning(
        "Model weights not found. Train the model or place weights at: "
        f"{weights_path}"
    )

if not kb_dir.exists():
    st.warning(f"Knowledge base directory not found: {kb_dir}")

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


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def rule_based_summary(context: str) -> dict:
    cause: list[str] = []
    prevent: list[str] = []
    other: list[str] = []

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


col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "webp"])

with col2:
    user_question = st.text_area(
        "Ask about disease cause and prevention",
        value="What causes this disease and how to prevent it?",
        height=120,
    )

if uploaded_file and weights_path.exists():
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Running model..."):
        predictor = get_predictor(weights_path)
        result = predictor.predict(image)

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

    st.subheader("Text-conditioned Attention Overlay")
    st.image(result["overlay"], use_container_width=True)

    st.subheader("Management Guidance (RAG + Groq)")
    rag_index = get_rag_index(kb_dir)
    query = f"{pred_label}. {user_question}"
    chunks = rag_index.query(query, top_k=3)

    if not chunks:
        st.info("No knowledge base documents found. Add files in knowledge_base/")
    else:
        context = "\n\n".join([f"Source: {c.source}\n{c.text}" for c in chunks])
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
                st.write("Retrieved context:")
                for c in chunks:
                    st.write(f"- {c.source}")
                    st.write(c.text)
        else:
            summary = rule_based_summary(context)
            st.subheader("Cause (from KB)")
            if summary["cause"]:
                st.markdown("\n".join(f"- {s}" for s in summary["cause"]))
            else:
                st.write("No clear cause sentences found in retrieved context.")

            st.subheader("Prevention/Management (from KB)")
            if summary["prevention"]:
                st.markdown("\n".join(f"- {s}" for s in summary["prevention"]))
            else:
                st.write("No clear prevention sentences found in retrieved context.")

            with st.expander("Raw Context"):
                st.write(context)
