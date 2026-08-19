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

except Exception:
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
    "Take a photo or upload a cotton leaf image "
    "to detect the disease."
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
# LOAD MODEL
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
# IMAGE INPUT
# =====================================

st.subheader(
    "📷 Choose Image"
)


# =====================================
# SESSION STATE
# =====================================

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False


if "camera_image" not in st.session_state:
    st.session_state.camera_image = None


# =====================================
# TAKE PHOTO BUTTON
# =====================================

if not st.session_state.camera_active:

    if st.button(
        "📷 Take Photo",
        use_container_width=True
    ):

        st.session_state.camera_active = True

        st.rerun()


# =====================================
# CAMERA
# =====================================

if st.session_state.camera_active:

    st.info(
        "📸 Camera is ready. "
        "Take a photo of the cotton leaf."
    )

    camera_file = st.camera_input(
        "Take a photo"
    )

    # ---------------------------------
    # PHOTO CAPTURED
    # ---------------------------------

    if camera_file is not None:

        st.session_state.camera_image = camera_file

        # Turn camera off after photo
        st.session_state.camera_active = False

        st.rerun()


# =====================================
# UPLOAD OPTION
# =====================================

uploaded_file = st.file_uploader(
    "📁 Or upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =====================================
# SELECT IMAGE
# =====================================

image_file = None


# Camera image has priority
if st.session_state.camera_image is not None:

    image_file = (
        st.session_state.camera_image
    )

elif uploaded_file is not None:

    image_file = uploaded_file


# =====================================
# PREDICTION
# =====================================

if image_file is not None:

    # ---------------------------------
    # OPEN IMAGE
    # ---------------------------------

    try:

        image = Image.open(
            image_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            "❌ Unable to open image."
        )

        st.code(str(e))

        st.stop()


    # ---------------------------------
    # DISPLAY IMAGE
    # ---------------------------------

    st.subheader(
        "🖼️ Selected Image"
    )

    st.image(
        image,
        caption="Cotton Leaf Image",
        use_container_width=True
    )


    # ---------------------------------
    # PREDICT
    # ---------------------------------

    with st.spinner(
        "🔍 Analyzing the cotton leaf..."
    ):

        try:

            result = model.predict(
                image
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.code(str(e))

            st.stop()


    # =================================
    # RESULT
    # =================================

    st.subheader(
        "🩺 Detection Result"
    )

    predicted_class = result[
        "class_label"
    ]

    st.success(
        f"🌿 Prediction: {predicted_class}"
    )


    # =================================
    # PROBABILITIES
    # =================================

    st.subheader(
        "📊 Class Probabilities"
    )

    probs = result[
        "probabilities"
    ]


    for i, prob in enumerate(probs):

        class_name = CLASS_NAMES[i]

        display_name = pretty_class_name(
            class_name
        )

        percentage = (
            float(prob) * 100
        )

        st.write(
            f"**{display_name}**: "
            f"{percentage:.2f}%"
        )

        st.progress(
            min(
                max(
                    float(prob),
                    0.0
                ),
                1.0
            )
        )


    # =================================
    # GRADCAM
    # =================================

    st.subheader(
        "🔥 GradCAM Visualization"
    )

    try:

        gradcam_image = result[
            "gradcam"
        ]

        st.image(
            gradcam_image,
            caption="GradCAM - Model Attention Visualization",
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            "⚠️ GradCAM visualization "
            "is not available."
        )

        st.code(str(e))


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "🌿 Cotton Leaf Disease Detection "
    "using Deep Learning & GradCAM"
)
