# ===== app.py =====
import streamlit as st
from PIL import Image
import sys
import os

# 🔥 FIX: ensure correct path
sys.path.append(os.path.dirname(__file__))

import config

# 🔥 IMPORTS
try:
    from src.predict import load_predictor
    from prompts import CLASS_NAMES, pretty_class_name
except:
    from predict import load_predictor
    from prompts import CLASS_NAMES, pretty_class_name


# ==============================
# 🎯 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Cotton Leaf Disease Detection")
st.write("Upload a cotton leaf image to detect disease.")


# ==============================
# 🚀 LOAD MODEL (CACHED)
# ==============================
@st.cache_resource
def get_model():
    return load_predictor(config.MODEL_WEIGHTS)


# ==============================
# 🔄 LOAD MODEL WITH ERROR HANDLE
# ==============================
try:
    model = get_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Model loading failed")
    st.code(str(e))
    st.stop()


# ==============================
# 📤 IMAGE UPLOAD
# ==============================
uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # 🔥 UPDATED (new Streamlit syntax)
    st.image(image, caption="📷 Uploaded Image", width="stretch")

    # ==============================
    # 🔍 PREDICTION
    # ==============================
    with st.spinner("🔍 Predicting..."):
        result = model.predict(image)

    st.success(f"🌱 Prediction: {result['class_label']}")

    # ==============================
    # 📊 PROBABILITIES (FIXED)
    # ==============================
    st.subheader("📊 Class Probabilities")

    probs = result["probabilities"]

    for i, prob in enumerate(probs):
        class_name = CLASS_NAMES[i]
        display_name = pretty_class_name(class_name)

        st.write(f"{display_name}: {prob*100:.2f}%")
        st.progress(float(prob))

    # ==============================
    # 🔥 ATTENTION MAP
    # ==============================
    if result.get("overlay") is not None:
        st.subheader("🔥 Attention Map")
        st.image(result["overlay"], width="stretch")
