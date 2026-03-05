# ============================================
# 🐄 Cattle Ear-Tag Detection App
# Streamlit Cloud Compatible Version
# ============================================

import os
import cv2
import json
import zipfile
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import easyocr

# ============================================
# Page Config
# ============================================

st.set_page_config(
    page_title="Cow Ear-Tag AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# Model Paths (IMPORTANT)
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "YOLOv8n_best.pt")  # <-- YOUR MODEL

# ============================================
# Cached Model Loading
# ============================================

@st.cache_resource
def load_yolo():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(
        ['en'],
        gpu=False,
        model_storage_directory="/tmp",
        download_enabled=True
    )

# ============================================
# UI
# ============================================

st.title("🐄 Cattle Ear-Tag Detection System")

st.markdown("""
Upload cow images → Detect ear tags → Read numbers → Correct → Download results.
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload image or ZIP",
    type=["zip", "jpg", "jpeg", "png"]
)

confidence_level = st.slider(
    "Detection Confidence",
    0.1, 1.0, 0.4, 0.1
)

st.divider()

# ============================================
# Main Processing
# ============================================

if uploaded_file:

    st.subheader("🔍 Processing")

    model = load_yolo()
    ocr_reader = load_ocr()

    if model is None:
        st.error("❌ YOLOv8_best.pt not found in root directory.")
        st.stop()

    # Temp folder
    with tempfile.TemporaryDirectory() as temp_dir:

        image_paths = []

        # ZIP handling
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as z:
                z.extractall(temp_dir)

            for file in os.listdir(temp_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(temp_dir, file))
        else:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths = [temp_path]

        st.success(f"Found {len(image_paths)} image(s)")

        results_db = []

        for idx, image_path in enumerate(image_paths, 1):

            img_name = os.path.basename(image_path)
            original_img = cv2.imread(image_path)

            if original_img is None:
                st.warning(f"Could not read {img_name}")
                continue

            results = model(image_path, conf=confidence_level)

            with st.expander(f"📷 {img_name}", expanded=True):

                col1, col2 = st.columns([2, 1])

                # Show detection image
                with col1:
                    plotted = results[0].plot()
                    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    st.image(plotted, use_container_width=True)

                # Process tags
                with col2:

                    boxes = results[0].boxes

                    if boxes is None or len(boxes) == 0:
                        st.warning("No tags detected")
                        continue

                    for tag_id, box in enumerate(boxes, 1):

                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        crop = original_img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, width=180)

                        # OCR
                        ocr_text = ""
                        try:
                            ocr_result = ocr_reader.readtext(crop, detail=0)
                            ocr_text = " ".join(ocr_result).strip()
                        except:
                            pass

                        if ocr_text:
                            st.success(f"OCR: {ocr_text}")
                        else:
                            st.error("OCR Failed")

                        user_input = st.text_input(
                            f"Edit Tag {tag_id}",
                            value=ocr_text,
                            key=f"{idx}_{tag_id}"
                        )

                        final_value = user_input if user_input else ocr_text

                        st.progress(conf, text=f"Confidence: {conf:.1%}")

                        results_db.append({
                            "image": img_name,
                            "tag": tag_id,
                            "ocr": ocr_text,
                            "final": final_value,
                            "confidence": conf
                        })

        # ============================================
        # Download Section
        # ============================================

        if results_db:

            st.divider()
            st.subheader("💾 Download Results")

            json_data = json.dumps(results_db, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                "results.json",
                "application/json"
            )

            csv_lines = ["Image,Tag,OCR,Final,Confidence"]
            for r in results_db:
                csv_lines.append(
                    f'{r["image"]},{r["tag"]},"{r["ocr"]}","{r["final"]}",{r["confidence"]:.2%}'
                )

            st.download_button(
                "Download CSV",
                "\n".join(csv_lines),
                "results.csv",
                "text/csv"
            )

else:
    st.info("Upload an image or ZIP file to begin.")
