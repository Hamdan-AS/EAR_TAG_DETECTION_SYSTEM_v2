# ============================================
# 🐄 Cattle Ear-Tag Detection App
# Streamlit Cloud Compatible Version
# ============================================

import sys
import streamlit as st

# Page config first
st.set_page_config(
    page_title="Cow Ear-Tag AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# Safe Imports with Error Handling
# ============================================

def safe_import():
    """Import all packages with error handling"""
    try:
        import cv2
    except ImportError:
        st.error("""
        ❌ OpenCV failed to import. This is a Streamlit Cloud issue.
        
        **Try this:**
        1. Go to your GitHub repo
        2. Delete the app from Streamlit Cloud
        3. Recreate it fresh
        4. Wait 5-10 minutes
        
        Or contact Streamlit support.
        """)
        st.stop()
    
    try:
        import numpy as np
    except ImportError:
        st.error("❌ NumPy import failed")
        st.stop()
    
    try:
        from PIL import Image
    except ImportError:
        st.error("❌ Pillow import failed")
        st.stop()
    
    try:
        import easyocr
    except ImportError:
        st.error("❌ EasyOCR import failed. Try: pip install easyocr")
        st.stop()
    
    return cv2, np, Image, easyocr

# Try to import
try:
    cv2, np, Image, easyocr = safe_import()
except:
    st.stop()

# Standard library imports
import zipfile
import os
import tempfile
from datetime import datetime
import json

# ============================================
# Styling
# ============================================

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stExpander"] { 
        border: 2px solid #4CAF50; 
        border-radius: 10px; 
        background-color: white; 
    }
    .stMetric { 
        background-color: #e8f5e9; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #4CAF50;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Model Loading
# ============================================

@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        if not os.path.exists(model_path):
            return None
        return YOLO(model_path)
    except Exception as e:
        st.warning(f"⚠️ YOLO load failed: {e}")
        return None

@st.cache_resource
def load_ocr_model():
    """Load OCR model"""
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.warning(f"⚠️ OCR load failed: {e}")
        return None

# ============================================
# Title
# ============================================

st.title("🐄 Cattle Ear-Tag Detection System")
st.markdown("""
This app helps you:
1. **Upload** photos of cows
2. **Detect** ear tags
3. **Read** the numbers
4. **Correct** if needed
5. **Save** results
""")

st.divider()

# ============================================
# Upload Section
# ============================================

st.subheader("📂 Upload Your Images")

uploaded_file = st.file_uploader(
    "Choose image(s)",
    type=["zip", "jpg", "jpeg", "png"],
    help="Single image or ZIP folder"
)

# ============================================
# Settings
# ============================================

st.subheader("⚙️ Settings")

confidence_level = st.slider(
    "Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.1
)

model_file = "cow_eartag_yolov8n_100ep_clean_best.pt"

st.divider()

# ============================================
# Main Processing
# ============================================

if uploaded_file:
    st.subheader("🔍 Processing")
    
    # Load models
    progress_bar = st.progress(0)
    status = st.empty()
    
    status.text("⏳ Loading YOLO model...")
    progress_bar.progress(25)
    model = load_yolo_model(model_file)
    
    status.text("⏳ Loading OCR model...")
    progress_bar.progress(50)
    ocr_reader = load_ocr_model()
    
    progress_bar.progress(75)
    
    if model is None:
        st.error("❌ YOLO model not found. Make sure cow_eartag_yolov8n_100ep_clean_best.pt is in the app folder.")
        st.stop()
    
    if ocr_reader is None:
        st.error("❌ OCR model failed to load.")
        st.stop()
    
    status.text("✅ Models loaded!")
    progress_bar.progress(100)
    st.success("Ready to process!")
    
    # Process files
    with tempfile.TemporaryDirectory() as temp_folder:
        image_list = []
        
        # Extract images
        if uploaded_file.name.endswith(".zip"):
            st.info("📦 Extracting ZIP...")
            with zipfile.ZipFile(uploaded_file, "r") as z:
                z.extractall(temp_folder)
            
            for filename in os.listdir(temp_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_list.append(os.path.join(temp_folder, filename))
        else:
            temp_path = os.path.join(temp_folder, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_list = [temp_path]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Images", len(image_list))
        col2.metric("🎯 Confidence", f"{confidence_level:.0%}")
        col3.metric("✅ Status", "Processing")
        
        # Initialize results
        if 'results_db' not in st.session_state:
            st.session_state.results_db = []
        st.session_state.results_db = []
        
        # Process images
        for img_idx, image_path in enumerate(image_list, 1):
            img_name = os.path.basename(image_path)
            
            try:
                original_img = cv2.imread(image_path)
                if original_img is None:
                    st.warning(f"⚠️ Could not read: {img_name}")
                    continue
                
                detection_results = model(image_path, conf=confidence_level)
                
                with st.expander(f"📷 Image {img_idx}/{len(image_list)}: {img_name}", expanded=True):
                    left_col, right_col = st.columns([2, 1])
                    
                    # Full image
                    with left_col:
                        st.markdown("**Full Image**")
                        try:
                            img_with_boxes = detection_results[0].plot()
                            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True)
                        except:
                            st.warning("Could not display image")
                    
                    # Tags
                    with right_col:
                        st.markdown("**Tags**")
                        boxes = detection_results[0].boxes
                        
                        if len(boxes) > 0:
                            for tag_idx, box in enumerate(boxes, 1):
                                confidence = float(box.conf[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                tag_crop = original_img[y1:y2, x1:x2]
                                
                                if tag_crop.size == 0:
                                    continue
                                
                                tag_rgb = cv2.cvtColor(tag_crop, cv2.COLOR_BGR2RGB)
                                st.image(tag_rgb, caption=f"Tag {tag_idx}", width=180)
                                
                                # OCR
                                ocr_text = ""
                                try:
                                    ocr_result = ocr_reader.readtext(tag_crop, detail=0)
                                    ocr_text = " ".join(ocr_result) if ocr_result else ""
                                except:
                                    pass
                                
                                if ocr_text:
                                    st.markdown(
                                        f'<div class="success-box">✅ OCR: <code>{ocr_text}</code></div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        '<div class="error-box">⚠️ OCR Failed</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Manual input
                                st.markdown(
                                    '<div class="info-box"><strong>✏️ Enter ID</strong></div>',
                                    unsafe_allow_html=True
                                )
                                user_input = st.text_input(
                                    "Tag ID",
                                    value=ocr_text,
                                    key=f"tag_{img_idx}_{tag_idx}",
                                    label_visibility="collapsed"
                                )
                                
                                final_value = user_input if user_input else (ocr_text if ocr_text else "EMPTY")
                                st.markdown(f"**Final ID:** `{final_value}`")
                                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                                
                                # Store
                                st.session_state.results_db.append({
                                    "image": img_name,
                                    "tag": tag_idx,
                                    "ocr": ocr_text,
                                    "user": user_input,
                                    "final": final_value,
                                    "conf": confidence,
                                    "time": datetime.now().isoformat()
                                })
                                
                                st.divider()
                        else:
                            st.warning("❌ No tags found")
            
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Save section
        st.divider()
        st.subheader("💾 Save Results")
        
        if st.session_state.results_db:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Download JSON"):
                    json_str = json.dumps(st.session_state.results_db, indent=2)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "⬇️ Get JSON File",
                        json_str,
                        f"results_{timestamp}.json",
                        "application/json"
                    )
            
            with col2:
                if st.button("📊 Download CSV"):
                    csv_lines = ["Image,Tag,OCR,User,Final,Confidence"]
                    for r in st.session_state.results_db:
                        line = f'{r["image"]},{r["tag"]},"{r["ocr"]}","{r["user"]}","{r["final"]}",{r["conf"]:.2%}'
                        csv_lines.append(line)
                    csv_str = "\n".join(csv_lines)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "⬇️ Get CSV File",
                        csv_str,
                        f"results_{timestamp}.csv",
                        "text/csv"
                    )
            
            st.divider()
            st.subheader("📊 Summary")
            summary = []
            for r in st.session_state.results_db:
                summary.append({
                    "Image": r["image"],
                    "Tag": r["tag"],
                    "Final": r["final"],
                    "Conf": f"{r['conf']:.1%}"
                })
            st.dataframe(summary, use_container_width=True)

else:
    st.info("👆 Upload an image or ZIP to get started!")
