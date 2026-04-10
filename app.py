# ===== app.py =====
import streamlit as st
from predict import load_predictor
import config
from PIL import Image

st.title("🌿 Cotton Leaf Disease Detection")

@st.cache_resource
def get_model():
    return load_predictor(config.MODEL_WEIGHTS)

model = get_model()

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    result = model.predict(image)

    st.success(f"Prediction: {result['class_label']}")
