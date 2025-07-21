import sys
import os

# Ensure .venv site-packages path is added
venv_path = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages")
if venv_path not in sys.path:
    sys.path.append(venv_path)

import streamlit as st
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

from text_model import detect_and_explain
from image_model import compare_faces
from video_model import predict_video_folder
from datetime import datetime
from PIL import Image
from streamlit_lottie import st_lottie
import json

# Prediction history
prediction_history = []

# Page config
st.set_page_config(
    page_title="Digital Doppelgänger Detector",
    layout="wide",
    page_icon="🤖"
)

# CSS Styling
st.markdown("""
    <style>
    body { background-color: #0f0f0f; }
    .block-container { padding-top: 2rem; }
    .title {
        font-size: 3em;
        font-weight: bold;
        color: #00ffe7;
        text-shadow: 0 0 10px #00ffe7;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.1em;
        color: #aaa;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 8px;
        font-weight: bold;
        color: white;
    }
    .real { background-color: #00c853; }
    .fake { background-color: #ff1744; }
    </style>
""", unsafe_allow_html=True)

# Load Lottie animation
@st.cache_data
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

loading_anim = load_lottie("scan.json") if os.path.exists("scan.json") else None

# Header
st.markdown('<div class="title">Digital Doppelgänger Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unmasking the Artificial in Real-Time</div>', unsafe_allow_html=True)

# --- Main Interface ---
col1, col2, col3 = st.columns([1, 1, 1])

# TEXT ANALYSIS
# TEXT ANALYSIS
with col1:
    st.subheader("📝 Text Analysis")
    text_input = st.text_area("Paste text here:")
    if st.button("Analyze Text") and text_input:
        with st.spinner("Scanning text..."):
            if loading_anim:
                st_lottie(loading_anim, height=150)

            result = detect_and_explain(text_input)
            label_clean = result['label'].strip().lower()

            # Confidence-based label adjustment
            if label_clean == "real" and result['confidence'] < 90:
                final_label = "AI-Generated"
                badge = '<span class="badge fake">\u26a0\ufe0f Possibly AI-Generated</span>'
            else:
                final_label = result['label']
                badge = (
                    '<span class="badge real">\u2705 Real</span>'
                    if label_clean == "real"
                    else '<span class="badge fake">\u26a0\ufe0f AI-Generated</span>'
                )

            st.markdown(f"**Prediction:** {final_label} {badge}", unsafe_allow_html=True)
            st.progress(result['confidence'] / 100)
            st.write(f"**Confidence:** {result['confidence']}%")
            st.write("**Key Phrases:**", result['highlighted_phrases'])
            prediction_history.insert(0, (datetime.now(), "Text", final_label, result['confidence']))

# IMAGE COMPARISON
with col2:
    st.subheader("🖼️ Image Comparison")
    img1 = st.file_uploader("Upload Real Image", type=["png", "jpg", "jpeg"], key="img1")
    img2 = st.file_uploader("Upload Suspected AI Image", type=["png", "jpg", "jpeg"], key="img2")
    if img1 and img2 and st.button("Compare Faces"):
        with open("img1.jpg", "wb") as f1:
            f1.write(img1.read())
        with open("img2.jpg", "wb") as f2:
            f2.write(img2.read())
        with st.spinner("Comparing faces..."):
            if loading_anim:
                st_lottie(loading_anim, height=150)
            result = compare_faces("img1.jpg", "img2.jpg")
            if "error" in result:
                st.error(result["error"])
            else:
                badge = '<span class="badge real">\u2705 Real</span>' if result['verified'] else '<span class="badge fake">\u26a0\ufe0f AI-Generated</span>'
                st.markdown(f"**Verified:** {result['verified']} {badge}", unsafe_allow_html=True)
                st.progress(result['similarity_score'] / 100)
                st.write(f"**Similarity Score:** {result['similarity_score']}%")
                prediction_history.insert(0, (datetime.now(), "Image", "Real" if result['verified'] else "AI-Generated", result['similarity_score']))

# VIDEO ANALYSIS (Single Upload)
with col3:
    st.subheader("🎥 Video Deepfake Detection")
    video_file = st.file_uploader("Upload video clip", type=["mp4", "mov", "avi"])
    if video_file and st.button("Analyze Video"):
        os.makedirs("test_videos", exist_ok=True)
        video_path = os.path.join("test_videos", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        with st.spinner("Analyzing video..."):
            if loading_anim:
                st_lottie(loading_anim, height=150)
            video_results = predict_video_folder("test_videos")
            if isinstance(video_results, dict) and "error" not in video_results:
                for fname, (label, confidence) in video_results.items():
                    badge = '<span class="badge real">\u2705 Real</span>' if label == "Real" else '<span class="badge fake">\u26a0\ufe0f AI-Generated</span>'
                    st.markdown(f"**Prediction for `{fname}`:** {label} {badge}", unsafe_allow_html=True)
                    st.progress(confidence)
                    st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
                    prediction_history.insert(0, (datetime.now(), "Video", label, round(confidence * 100, 2)))
            else:
                st.error(video_results.get("error", "Prediction failed or returned no result."))
        shutil.rmtree("test_videos", ignore_errors=True)

# --- History Section ---
st.markdown("---")
st.subheader("🕒 Last 5 Predictions")
if prediction_history:
    for time_stamp, content_type, label, conf in prediction_history[:5]:
        badge = '<span class="badge real">\u2705 Real</span>' if label.strip().lower() == "real" else '<span class="badge fake">\u26a0\ufe0f AI-Generated</span>'
        st.markdown(f"- [{time_stamp.strftime('%H:%M:%S')}] **{content_type}**: {label} {badge} — *Confidence:* {conf}%", unsafe_allow_html=True)
else:
    st.info("No predictions made yet.")
