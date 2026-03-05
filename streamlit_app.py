# ============================================
# 🐄 Cattle Ear-Tag Detection App
# A beginner-friendly app to detect and read cow ear tags
# Compatible with Python 3.8 - 3.13
# ============================================

import sys
import streamlit as st

# Check Python version
if sys.version_info < (3, 8):
    st.error("❌ This app requires Python 3.8 or higher")
    st.stop()

import cv2
import numpy as np
from PIL import Image
import zipfile
import os
import tempfile
import easyocr
from datetime import datetime
import json

# ============================================
# STEP 1: Set up the page
# ============================================
st.set_page_config(
    page_title="Cow Ear-Tag AI",  # Browser tab title
    layout="wide",  # Use full width
    initial_sidebar_state="collapsed"  # Hide sidebar
)

# ============================================
# STEP 2: Add some colors and styling
# ============================================
st.markdown("""
    <style>
    /* Make the background light */
    .main { background-color: #f8f9fa; }
    
    /* Style the boxes that expand/collapse */
    div[data-testid="stExpander"] { 
        border: 2px solid #4CAF50; 
        border-radius: 10px; 
        background-color: white; 
    }
    
    /* Style the numbers (metrics) */
    .stMetric { 
        background-color: #e8f5e9; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #4CAF50;
    }
    
    /* Green box for successful OCR */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    /* Red box for failed OCR */
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    /* Yellow box for manual entry */
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
# STEP 3: Load models (only once to save time)
# ============================================

@st.cache_resource  # This means "remember this, don't load it again"
def load_yolo_model(model_path):
    """Load the YOLO detection model (finds the ear tags in images)"""
    try:
        from ultralytics import YOLO
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}\nMake sure the .pt file is in the same folder as app.py")
            st.stop()
        return YOLO(model_path)
    except ImportError:
        st.error("❌ YOLO not installed properly. Try: pip install --upgrade ultralytics")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        st.stop()

@st.cache_resource
def load_ocr_model():
    """Load the OCR model (reads text from the ear tags)"""
    try:
        return easyocr.Reader(['en'], gpu=False)
    except ImportError:
        st.error("❌ EasyOCR not installed properly. Try: pip install --upgrade easyocr")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading OCR model: {e}")
        st.stop()

# ============================================
# STEP 4: Create the title and instructions
# ============================================
st.title("🐄 Cattle Ear-Tag Detection System")
st.markdown("""
This app helps you:
1. **Upload** photos of cows with ear tags
2. **Detect** where the ear tags are in the photos
3. **Read** the numbers on the ear tags automatically
4. **Fix** if the reading was wrong
5. **Save** all the results

Let's get started! 👇
""")

st.divider()  # Add a line to separate sections

# ============================================
# STEP 5: Create upload section
# ============================================
st.subheader("📂 Step 1: Upload Your Images")

uploaded_file = st.file_uploader(
    "Choose image(s) to upload",
    type=["zip", "jpg", "jpeg", "png"],
    help="Upload a single image or a ZIP file with multiple images"
)

# ============================================
# STEP 6: Create settings section
# ============================================
st.subheader("⚙️ Step 2: Adjust Settings")

confidence_level = st.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.1,
    help="Lower = finds more tags (but might be wrong), Higher = only finds clear tags"
)

model_file = "cow_eartag_yolov8n_100ep_clean_best.pt"

st.divider()

# ============================================
# STEP 7: Main processing (only if user uploaded something)
# ============================================
if uploaded_file:
    st.subheader("🔍 Step 3: Processing Your Images")
    
    # Try to load the models
    try:
        st.info("🔄 Loading AI models... (this takes a few seconds on first run)")
        model = load_yolo_model(model_file)
        ocr_reader = load_ocr_model()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()  # Stop the app if models don't load
    
    # Create a temporary folder to work in
    with tempfile.TemporaryDirectory() as temp_folder:
        image_list = []
        
        # ============================================
        # STEP 8a: Handle ZIP files
        # ============================================
        if uploaded_file.name.endswith(".zip"):
            st.info("📦 Extracting images from ZIP file...")
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                zip_ref.extractall(temp_folder)
            
            # Find all image files in the ZIP
            image_extensions = (".jpg", ".jpeg", ".png")
            for filename in os.listdir(temp_folder):
                if filename.lower().endswith(image_extensions):
                    image_list.append(os.path.join(temp_folder, filename))
        
        # ============================================
        # STEP 8b: Handle single image
        # ============================================
        else:
            temp_path = os.path.join(temp_folder, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_list = [temp_path]
        
        # ============================================
        # STEP 9: Show how many images we found
        # ============================================
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Total Images", len(image_list))
        col2.metric("🎯 Confidence Level", f"{confidence_level:.0%}")
        col3.metric("✅ Status", "Ready to Process")
        
        # ============================================
        # STEP 10: Initialize storage for results
        # ============================================
        if 'results_database' not in st.session_state:
            st.session_state.results_database = []
        
        # Clear old results
        st.session_state.results_database = []
        
        # ============================================
        # STEP 11: Process each image
        # ============================================
        for image_number, image_path in enumerate(image_list, start=1):
            image_name = os.path.basename(image_path)
            
            # Read the image
            original_image = cv2.imread(image_path)
            
            # Detect ear tags using YOLO
            detection_results = model(image_path, conf=confidence_level)
            
            # Create an expandable section for each image
            with st.expander(f"📷 Image {image_number}/{len(image_list)}: {image_name}", expanded=True):
                
                # Split into two columns
                left_column, right_column = st.columns([2, 1])
                
                # ============================================
                # LEFT SIDE: Show the full image
                # ============================================
                with left_column:
                    st.markdown("**Full Image with Detections**")
                    
                    # Draw boxes around detected tags
                    image_with_boxes = detection_results[0].plot()
                    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                    st.image(image_with_boxes_rgb, use_container_width=True)
                
                # ============================================
                # RIGHT SIDE: Show the tags
                # ============================================
                with right_column:
                    st.markdown("**Detected Tags**")
                    
                    # Check if any tags were found
                    boxes = detection_results[0].boxes
                    
                    if len(boxes) > 0:
                        # Process each detected tag
                        for tag_number, box in enumerate(boxes, start=1):
                            confidence = float(box.conf[0])
                            
                            # Get the exact location of the tag
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            tag_crop = original_image[y1:y2, x1:x2]
                            tag_crop_rgb = cv2.cvtColor(tag_crop, cv2.COLOR_BGR2RGB)
                            
                            # Try to read the tag number using OCR
                            ocr_results = ocr_reader.readtext(tag_crop, detail=0)
                            ocr_text = " ".join(ocr_results) if ocr_results else ""
                            
                            # Show the cropped tag
                            st.image(tag_crop_rgb, caption=f"Tag {tag_number}", width=180)
                            
                            # Show if OCR worked
                            if ocr_text:
                                st.markdown(
                                    f'<div class="success-box">✅ <strong>OCR Read:</strong> <code>{ocr_text}</code></div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    '<div class="error-box">⚠️ <strong>OCR Could Not Read</strong> - Please enter manually below</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Let user enter or correct the tag number
                            st.markdown('<div class="info-box"><strong>✏️ Your Entry</strong></div>', unsafe_allow_html=True)
                            
                            user_input = st.text_input(
                                "Enter the tag number (or correct the OCR)",
                                value=ocr_text,
                                key=f"tag_input_{image_number}_{tag_number}",
                                label_visibility="collapsed"
                            )
                            
                            # Show the final value
                            final_value = user_input if user_input else ocr_text if ocr_text else "EMPTY"
                            st.markdown(f"**Final Tag ID:** `{final_value}`")
                            
                            # Show confidence
                            st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                            
                            # Save this result
                            st.session_state.results_database.append({
                                "image_name": image_name,
                                "tag_number": tag_number,
                                "ocr_result": ocr_text,
                                "user_entry": user_input,
                                "final_value": final_value,
                                "confidence": confidence,
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            st.divider()
                    
                    else:
                        st.warning("❌ No tags found in this image. Try lowering the confidence level.")
        
        # ============================================
        # STEP 12: Save results section
        # ============================================
        st.divider()
        st.subheader("💾 Step 4: Save Your Results")
        
        if st.session_state.results_database:
            
            # Create two buttons for different save options
            save_col1, save_col2 = st.columns(2)
            
            # ============================================
            # Option 1: Save as JSON (for computers/databases)
            # ============================================
            with save_col1:
                st.markdown("**Option 1: Save as JSON**")
                st.markdown("Good for: Importing into databases or Excel")
                
                if st.button("📋 Prepare JSON File", key="json_button"):
                    json_data = json.dumps(st.session_state.results_database, indent=2)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ear_tag_results_{timestamp}.json"
                    
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json_data,
                        file_name=filename,
                        mime="application/json"
                    )
                    st.success("✅ JSON file is ready!")
            
            # ============================================
            # Option 2: Save as CSV (easier to open in Excel)
            # ============================================
            with save_col2:
                st.markdown("**Option 2: Save as CSV**")
                st.markdown("Good for: Opening in Excel or Google Sheets")
                
                if st.button("📊 Prepare CSV File", key="csv_button"):
                    # Convert to CSV format
                    csv_lines = ["Image Name,Tag Number,OCR Result,Your Entry,Final Value,Confidence,Time"]
                    
                    for result in st.session_state.results_database:
                        line = f'{result["image_name"]},{result["tag_number"]},"{result["ocr_result"]}","{result["user_entry"]}","{result["final_value"]}",{result["confidence"]:.2%},{result["time"]}'
                        csv_lines.append(line)
                    
                    csv_data = "\n".join(csv_lines)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ear_tag_results_{timestamp}.csv"
                    
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                    st.success("✅ CSV file is ready!")
            
            st.divider()
            
            # ============================================
            # Show a summary table
            # ============================================
            st.subheader("📊 Quick Summary")
            
            summary_data = []
            for result in st.session_state.results_database:
                summary_data.append({
                    "Image": result["image_name"],
                    "Tag #": result["tag_number"],
                    "Final Value": result["final_value"],
                    "Confidence": f"{result['confidence']:.1%}"
                })
            
            st.dataframe(summary_data, use_container_width=True)

# ============================================
# STEP 13: Show instructions if no file uploaded
# ============================================
else:
    st.info("""
    👆 **Start here:**
    1. Click "Browse files" above
    2. Choose your image(s) - single or in a ZIP folder
    3. Adjust the confidence level if needed
    4. Let the AI detect the tags
    5. Correct any mistakes
    6. Download your results!
    """)
