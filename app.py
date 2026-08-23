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
# CONFIG: OOD GATE SETTINGS
# =====================================

# CLIP zero-shot candidate labels.
# NOTE: CLIP is only used for a COARSE check here (is this a leaf/plant
# at all, vs a human/animal/random object). Fine-grained "is this
# specifically a cotton leaf" is left to your own trained classifier,
# because CLIP's generic zero-shot text embeddings cannot reliably
# distinguish cotton leaves from other similarly-shaped lobed leaves
# (grape, hibiscus, maple, etc.) -- that caused real cotton leaf images
# to get wrongly rejected.
CLIP_CANDIDATE_LABELS = [
    "a close-up photo of a plant leaf",
    "a photo of a human face or a person",
    "a photo of an animal",
    "a random photo unrelated to plants or leaves (object, document, screenshot, etc.)",
]

# indices in CLIP_CANDIDATE_LABELS that count as "some kind of leaf/plant"
VALID_LEAF_INDICES = {0}

# minimum CLIP confidence to accept the image as a leaf/plant.
# Kept low on purpose: we only want to catch CLEAR non-leaf images
# (human, animal, random objects). Any actual leaf -- cotton or not --
# should pass this gate; cotton-specific validation happens later
# using your own trained classifier's confidence.
CLIP_ACCEPT_THRESHOLD = 0.30

# minimum classifier confidence to trust the disease prediction
CLASSIFIER_ACCEPT_THRESHOLD = 0.55


# =====================================
# LOAD DISEASE MODEL
# =====================================

@st.cache_resource
def get_model():

    return load_predictor(
        config.MODEL_WEIGHTS
    )


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
# LOAD CLIP GATE MODEL (cached, separate from disease model)
# =====================================

@st.cache_resource
def get_clip_gate():
    """
    Loads a small CLIP model purely for zero-shot
    'is this actually a cotton leaf?' filtering.
    Uses HuggingFace transformers so it works out-of-the-box
    on Streamlit Cloud with just `transformers` in requirements.txt.
    """

    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_model.eval()

    return clip_model, clip_processor


try:

    clip_model, clip_processor = get_clip_gate()
    clip_gate_available = True

except Exception as e:

    # If CLIP gate fails to load (e.g. no internet on first run),
    # don't crash the whole app — just skip the OOD check.
    clip_gate_available = False
    st.warning(
        "⚠️ Image-validity check (CLIP gate) could not be loaded. "
        "Predictions will run without the extra safety filter."
    )
    st.code(str(e))


# =====================================
# OOD GATE FUNCTION
# =====================================

def is_leaf_or_plant(image: Image.Image):
    """
    Coarse OOD check only: is this image a leaf/plant at all,
    vs a human, animal, or random unrelated object?
    Returns (is_valid: bool, confidence: float, best_label: str)
    """

    inputs = clip_processor(
        text=CLIP_CANDIDATE_LABELS,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)[0]

    best_idx = int(torch.argmax(probs).item())
    confidence = float(probs[best_idx].item())
    best_label = CLIP_CANDIDATE_LABELS[best_idx]

    is_valid = (
        best_idx in VALID_LEAF_INDICES
        and confidence >= CLIP_ACCEPT_THRESHOLD
    )

    return is_valid, confidence, best_label


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
        caption="Selected Image",
        use_container_width=True
    )


    # ---------------------------------
    # OOD GATE CHECK
    # ---------------------------------

    leaf_check_passed = True

    if clip_gate_available:

        with st.spinner(
            "🔎 Checking if this looks like a leaf/plant image..."
        ):

            is_valid, clip_conf, best_label = is_leaf_or_plant(image)

        if not is_valid:

            leaf_check_passed = False

            st.error(
                "🚫 This does not look like a leaf/plant image. "
                "Please upload or capture a clear cotton leaf photo."
            )

            st.caption(
                f"Detected as: *{best_label}* "
                f"(confidence: {clip_conf*100:.1f}%)"
            )


    # ---------------------------------
    # PREDICT (only if gate passed)
    # ---------------------------------

    if leaf_check_passed:

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


        probs = result["probabilities"]
        max_prob = float(max(probs))

        # Hard reject: classifier itself is very unsure -> likely not
        # a cotton leaf (or a cotton leaf CLIP couldn't rule out, but
        # the disease model doesn't recognize it either).
        CLASSIFIER_REJECT_THRESHOLD = 0.35

        if max_prob < CLASSIFIER_REJECT_THRESHOLD:

            st.error(
                "🚫 The model could not confidently match this image "
                f"to any known cotton leaf class (max confidence: "
                f"{max_prob*100:.1f}%). This may not be a cotton leaf, "
                "or the photo quality/angle needs improvement."
            )

            leaf_check_passed = False


    if leaf_check_passed and image_file is not None:

        # -----------------------------
        # LOW-CONFIDENCE CHECK
        # -----------------------------

        if max_prob < CLASSIFIER_ACCEPT_THRESHOLD:

            st.warning(
                "⚠️ Model is not confident enough about this image "
                f"(max confidence: {max_prob*100:.1f}%). "
                "Try taking a clearer, closer photo of the cotton leaf "
                "with good lighting."
            )


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
