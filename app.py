# ===== app.py =====

import os
import sys

import torch
torch.set_num_threads(1)

import streamlit as st
from PIL import Image

# =====================================
# PATH FIX
# =====================================
sys.path.append(
    os.path.dirname(__file__)
)

import config

# =====================================
# IMPORTS
# =====================================
try:
    from src.predict import load_predictor
    from prompts import (
        CLASS_NAMES,
        pretty_class_name
    )
except:
    from predict import load_predictor
    from prompts import (
        CLASS_NAMES,
        pretty_class_name
    )

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# =====================================
# TITLE
# =====================================
st.title(
    "🌿 Cotton Leaf Disease Detection"
)

st.write(
    "Upload a cotton leaf image to detect disease."
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def get_model():

    return load_predictor(
        config.MODEL_WEIGHTS
    )

# =====================================
# LOAD WITH ERROR HANDLE
# =====================================
try:

    model = get_model()

    st.success(
        "✅ Model loaded successfully!"
    )

except Exception as e:

    st.error(
        "❌ Model loading failed"
    )

    st.code(str(e))

    st.stop()

# =====================================
# IMAGE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICT
# =====================================
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Predicting..."):

        result = model.predict(image)

    # =================================
    # RESULT
    # =================================
    st.success(
        f"Prediction: {result['class_label']}"
    )

    # =================================
    # PROBABILITIES
    # =================================
    st.subheader(
        "Class Probabilities"
    )

    probs = result["probabilities"]

    for i, prob in enumerate(probs):

        class_name = CLASS_NAMES[i]

        display_name = pretty_class_name(
            class_name
        )

        st.write(
            f"{display_name}: {prob*100:.2f}%"
        )

        st.progress(
            float(prob)
        )

    # =================================
    # GRADCAM
    # =================================
    st.subheader(
        "🔥 GradCAM Visualization"
    )

    st.image(
        result["gradcam"],
        use_container_width=True
    )
