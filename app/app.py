import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="SceneAI | Image Classifier",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/image_classifier.keras")
METADATA_PATH = Path("models/metadata.json")

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

CONFIDENCE_THRESHOLD = 60.0
VALIDATION_ACCURACY = "82.86%"
PAGES = ["Dashboard", "Model Info", "About"]


# =========================================================
# HTML HELPER  (yehi asal fix hai)
# Markdown 4+ spaces indent ya khali line ko code block bana deta hai.
# Ye helper har line ke leading spaces aur khali lines hata deta hai.
# =========================================================

def html(markup: str):
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# =========================================================
# CUSTOM CSS
# =========================================================

html(
    """
<style>

.stApp { background: #f5f8f6; color: #173b40; }

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] { background: #1f6f78; border-right: none; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: white !important; }

.brand { padding: 10px 10px 30px 10px; }
.brand-title { color: white; font-size: 30px; font-weight: 800; letter-spacing: -1px; margin-bottom: 4px; }
.brand-subtitle { color: #93e4c1; font-size: 13px; font-weight: 500; }
.nav-label {
    color: #93e4c1; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin: 5px 10px 8px 10px;
}
.side-note { color: #e7fffa; font-size: 12px; line-height: 1.7; padding: 10px; }

/* ---------- TOP BAR ---------- */
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 22px; }
.topbar-title { font-size: 15px; font-weight: 700; color: #173b40; }
.status {
    display: inline-flex; align-items: center; gap: 7px; background: #e6f6ef;
    color: #118a7e; padding: 7px 13px; border-radius: 30px; font-size: 12px; font-weight: 700;
}
.status.offline { background: #fdeaea; color: #c0392b; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #3baea0; }
.status.offline .status-dot { background: #c0392b; }

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, #1f6f78 0%, #118a7e 55%, #3baea0 100%);
    border-radius: 24px; padding: 38px 42px; margin-bottom: 22px; color: white;
    box-shadow: 0 12px 35px rgba(31, 111, 120, 0.16);
}
.hero-small { color: #93e4c1; font-size: 12px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }
.hero-title { font-size: 42px; line-height: 1.05; font-weight: 800; letter-spacing: -1.5px; margin: 0; }
.hero-description { color: #e7fffa; font-size: 15px; line-height: 1.7; max-width: 650px; margin-top: 14px; }

/* ---------- STAT CARDS ---------- */
.stat-card {
    background: white; border: 1px solid #e1ebe8; border-radius: 18px; padding: 22px;
    min-height: 120px; box-shadow: 0 6px 20px rgba(31, 111, 120, 0.05);
}
.stat-label { color: #719093; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { color: #173b40; font-size: 27px; font-weight: 800; margin-top: 8px; }
.stat-description { color: #8aa0a2; font-size: 12px; margin-top: 5px; }

/* ---------- CARDS ---------- */
/* st.container(border=True) ko card banane ke liye */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #e1ebe8 !important;
    border-radius: 20px !important;
    padding: 12px 14px;
    box-shadow: 0 6px 20px rgba(31, 111, 120, 0.05);
}
/* html card (About page) */
.card {
    background: white; border: 1px solid #e1ebe8; border-radius: 20px; padding: 25px;
    box-shadow: 0 6px 20px rgba(31, 111, 120, 0.05); height: 100%;
}
.card-title { color: #173b40; font-size: 18px; font-weight: 800; margin-bottom: 5px; }
.card-subtitle { color: #8aa0a2; font-size: 13px; margin-bottom: 18px; }

/* ---------- RESULT ---------- */
.result-box {
    background: linear-gradient(135deg, #0f4c54 0%, #14695f 100%);
    border-radius: 18px; padding: 28px; margin-top: 8px;
    box-shadow: 0 10px 28px rgba(15, 76, 84, 0.35);
}
.result-label { color: #93e4c1; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 800; }
.result-class { color: #ffffff; font-size: 48px; font-weight: 900; line-height: 1.1; margin: 6px 0 22px 0; text-transform: capitalize; }
.confidence-number { color: #ffffff; font-size: 34px; font-weight: 900; margin-top: 4px; }
.bar-bg { background: rgba(255, 255, 255, 0.2); border-radius: 10px; height: 12px; margin-top: 12px; overflow: hidden; }
.bar-fill { background: linear-gradient(90deg, #93e4c1, #ffffff); height: 100%; border-radius: 10px; }

.top-row { display: flex; justify-content: space-between; font-size: 13px; color: #49696c; margin-top: 12px; }
.top-row b { color: #173b40; text-transform: capitalize; }
.mini-bg { background: #e8f1ee; border-radius: 8px; height: 6px; margin-top: 4px; overflow: hidden; }
.mini-fill { background: #3baea0; height: 100%; }

.list-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 0; border-bottom: 1px solid #edf2f0;
}
.list-name { color: #173b40; font-weight: 600; text-transform: capitalize; }
.list-num { color: #118a7e; font-weight: 800; }

/* ---------- INFO BOX ---------- */
.info-box {
    background: #f0f7f5; border-left: 4px solid #3baea0; border-radius: 10px;
    padding: 14px 16px; color: #49696c; font-size: 13px; line-height: 1.7; margin-top: 10px;
}

/* ---------- FILE UPLOADER ---------- */
[data-testid="stFileUploader"] section {
    background: #f7fbfa; border: 2px dashed #93e4c1; border-radius: 16px;
}
/* dropzone ka text hamesha dark */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div {
    color: #173b40 !important;
}
/* Browse files button: shuru se hi dark teal + white text (hover ka wait nahi) */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    background: #118a7e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #0f4c54 !important;
    color: #ffffff !important;
}

/* ---------- ALERTS (success / warning / error text black) ---------- */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #000000 !important;
    font-weight: 600;
}

/* ---------- BUTTON ---------- */
.stButton > button { background: #118a7e; color: white; border: none; border-radius: 10px; font-weight: 700; }
.stButton > button:hover { background: #1f6f78; color: white; }

[data-testid="stImage"] img { border-radius: 16px; }

/* ---------- FOOTER ---------- */
.footer {
    text-align: center; color: #8aa0a2; font-size: 11px; margin-top: 45px;
    padding-top: 20px; border-top: 1px solid #dfe9e6;
}

@media (max-width: 800px) {
    .hero { padding: 28px; }
    .hero-title { font-size: 32px; }
}

</style>
"""
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def has_builtin_rescaling(model):
    return any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers)


def get_input_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height, width = shape[1], shape[2]
        if height is None or width is None:
            return (150, 150)
        return (int(height), int(width))
    except Exception:
        return (150, 150)


def preprocess(image, size, normalize):
    image = image.convert("RGB").resize(size)
    array = np.array(image).astype("float32")
    if normalize:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model, image):
    """Returns (predicted_class, confidence, top3) where top3 = [(name, pct), ...]"""
    size = get_input_size(model)
    processed = preprocess(image, size, not has_builtin_rescaling(model))
    predictions = model.predict(processed, verbose=0)[0]

    order = np.argsort(predictions)[::-1]
    top_index = int(order[0])
    top3 = [(CLASS_NAMES[int(i)], float(predictions[int(i)]) * 100) for i in order[:3]]

    return CLASS_NAMES[top_index], float(predictions[top_index]) * 100, top3


def stat_card(label, value, description):
    html(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-description">{description}</div>
        </div>
        """
    )


def topbar(title, online=True):
    status_class = "status" if online else "status offline"
    status_text = "MODEL ONLINE" if online else "MODEL OFFLINE"
    html(
        f"""
        <div class="topbar">
            <div class="topbar-title">{title}</div>
            <div class="{status_class}">
                <span class="status-dot"></span>{status_text}
            </div>
        </div>
        """
    )


def hero(small, title, description):
    html(
        f"""
        <div class="hero">
            <div class="hero-small">{small}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-description">{description}</div>
        </div>
        """
    )


def card_header(title, subtitle):
    html(
        f"""
        <div class="card-title">{title}</div>
        <div class="card-subtitle">{subtitle}</div>
        """
    )


def footer():
    html(
        """
        <div class="footer">
            SceneAI · MLOps Image Classification System · TensorFlow + Streamlit
        </div>
        """
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)

metadata = load_metadata()
model_online = model is not None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    html(
        """
        <div class="brand">
            <div class="brand-title">SceneAI</div>
            <div class="brand-subtitle">Intelligent Image Classification</div>
        </div>
        """
    )

    html('<div class="nav-label">Navigation</div>')

    selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")

    html(
        """
        <div class="side-note">
            <b>Model</b><br>
            CNN Image Classifier<br><br>
            <b>Classes</b><br>
            Buildings · Forest · Glacier<br>
            Mountain · Sea · Street
        </div>
        """
    )


# =========================================================
# DASHBOARD
# =========================================================

if selected_page == "Dashboard":

    topbar("Image Classification Dashboard", model_online)

    hero(
        "AI IMAGE ANALYSIS",
        "Understand your image.",
        "Upload an image and let the trained convolutional neural network "
        "identify the scene automatically. Get a prediction and confidence "
        "score in seconds.",
    )

    # ---------------- STATS ----------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stat_card("Model", "CNN", "Convolutional Neural Network")
    with col2:
        stat_card("Classes", f"{len(CLASS_NAMES):02d}", "Scene categories")
    with col3:
        size = get_input_size(model) if model_online else (150, 150)
        stat_card("Input", f"{size[0]}²", "Image resolution")
    with col4:
        stat_card("Validation", VALIDATION_ACCURACY, "Validation accuracy")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- MAIN AREA ----------------
    upload_col, result_col = st.columns([1.05, 0.95], gap="large")

    uploaded_file = None
    image = None

    with upload_col:
        with st.container(border=True):
            card_header(
                "Upload Image",
                "Select a JPG, JPEG or PNG image for classification.",
            )

            uploaded_file = st.file_uploader(
                "Choose image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.markdown("<br>", unsafe_allow_html=True)
                html('<div class="card-title">Image Preview</div>')
                st.image(image, use_container_width=True)

    with result_col:
        with st.container(border=True):
            card_header(
                "Classification Result",
                "AI-generated prediction from the trained model.",
            )

            if model_error:
                st.error("Unable to load the model.")
                st.code(model_error)

            elif uploaded_file is None:
                html(
                    """
                    <div class="info-box">
                        Upload an image from the left panel to see
                        the classification result here.
                    </div>
                    """
                )

            else:
                try:
                    with st.spinner("Analyzing image..."):
                        predicted_class, confidence, top3 = predict_image(model, image)

                    html(
                        f"""
                        <div class="result-box">
                            <div class="result-label">Predicted Scene</div>
                            <div class="result-class">{predicted_class}</div>
                            <div class="result-label">Confidence</div>
                            <div class="confidence-number">{confidence:.2f}%</div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width:{min(confidence, 100):.1f}%"></div>
                            </div>
                        </div>
                        """
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning(
                            "The model confidence is relatively low. "
                            "Try uploading a clearer image."
                        )
                    else:
                        st.success("The model is confident about this prediction.")

                except Exception as e:
                    st.error("Prediction failed.")
                    st.code(str(e))

    footer()


# =========================================================
# MODEL INFO
# =========================================================

elif selected_page == "Model Info":

    topbar("Model Information", model_online)

    hero(
        "MODEL DETAILS",
        "CNN Image Classifier",
        "A convolutional neural network trained to classify images into "
        "six different scene categories.",
    )

    if model is None:
        st.error("Model could not be loaded.")
        if model_error:
            st.code(model_error)
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            stat_card(
                "Parameters",
                f"{model.count_params():,}",
                "Total model parameters",
            )
        with col2:
            stat_card("Input Shape", str(model.input_shape), "Expected image input")
        with col3:
            stat_card("Output", f"{len(CLASS_NAMES)} Classes", "Classification categories")

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(2, gap="large")

        with left:
            with st.container(border=True):
                card_header(
                    "Classification Classes",
                    "Categories recognized by the model.",
                )
                rows = "".join(
                    f"""
                    <div class="list-row">
                        <span class="list-name">{name}</span>
                        <span class="list-num">{i:02d}</span>
                    </div>
                    """
                    for i, name in enumerate(CLASS_NAMES, start=1)
                )
                html(rows)

        with right:
            with st.container(border=True):
                card_header("Architecture", "Neural network configuration.")
                html(
                    """
                    <div class="info-box">
                        <b>Input</b><br>150 × 150 × 3 RGB image<br><br>
                        <b>Convolution Layers</b><br>32 → 64 → 128 filters<br><br>
                        <b>Pooling</b><br>MaxPooling2D<br><br>
                        <b>Dense Layer</b><br>128 neurons with ReLU<br><br>
                        <b>Regularization</b><br>Dropout 0.5<br><br>
                        <b>Output</b><br>6 neurons with Softmax
                    </div>
                    """
                )

    footer()


# =========================================================
# ABOUT
# =========================================================

elif selected_page == "About":

    topbar("About SceneAI", model_online)

    hero(
        "MLOPS PROJECT",
        "From Model to Deployment.",
        "SceneAI demonstrates an end-to-end machine learning workflow using "
        "TensorFlow, Streamlit and Docker. The project is designed as an MLOps "
        "portfolio project covering model training, inference and deployment.",
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        html(
            """
            <div class="card">
                <div class="card-title">Technology Stack</div>
                <div class="card-subtitle">Technologies used in this project.</div>
                <div class="info-box">
                    <b>Machine Learning</b><br>TensorFlow · Keras · CNN<br><br>
                    <b>Application</b><br>Python · Streamlit<br><br>
                    <b>Containerization</b><br>Docker<br><br>
                    <b>Dataset</b><br>Intel Image Classification
                </div>
            </div>
            """
        )

    with col2:
        html(
            """
            <div class="card">
                <div class="card-title">MLOps Workflow</div>
                <div class="card-subtitle">Current project pipeline.</div>
                <div class="info-box">
                    <b>01</b> · Dataset &amp; Training<br><br>
                    <b>02</b> · Model Evaluation<br><br>
                    <b>03</b> · Prediction Pipeline<br><br>
                    <b>04</b> · Streamlit Application<br><br>
                    <b>05</b> · Docker Container<br><br>
                    <b>06</b> · Kubernetes &amp; CI/CD<br><br>
                    <b>07</b> · Cloud Deployment
                </div>
            </div>
            """
        )

    footer()