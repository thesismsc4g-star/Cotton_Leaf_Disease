# ===== app.py =====

import os
import sys

import torch

# Limit CPU threads
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
# LOAD MODEL WITH ERROR HANDLE
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

    st.code(
        str(e)
    )

    st.stop()


# =====================================
# IMAGE INPUT
# =====================================

st.subheader(
    "📷 Select Leaf Image"
)

st.write(
    "You can take a photo using your camera "
    "or upload an existing image."
)


# =====================================
# CAMERA INPUT
# =====================================

camera_file = st.camera_input(
    "Take a photo of the cotton leaf"
)


# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Or upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =====================================
# SELECT IMAGE
# =====================================

# Camera image gets priority
# if both camera and upload are available.

image_file = (
    camera_file
    if camera_file is not None
    else uploaded_file
)


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

        st.code(
            str(e)
        )

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
    # PREDICTION
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

            st.code(
                str(e)
            )

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
    # CLASS PROBABILITIES
    # =================================

    st.subheader(
        "📊 Class Probabilities"
    )

    probs = result[
        "probabilities"
    ]


    for i, prob in enumerate(probs):

        # Get class name
        class_name = CLASS_NAMES[i]

        # Convert to readable name
        display_name = pretty_class_name(
            class_name
        )

        # Percentage
        percentage = (
            float(prob) * 100
        )

        # Display probability
        st.write(
            f"**{display_name}**: "
            f"{percentage:.2f}%"
        )

        # Progress bar
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

        st.code(
            str(e)
        )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "🌿 Cotton Leaf Disease Detection "
    "using Deep Learning & GradCAM"
)
