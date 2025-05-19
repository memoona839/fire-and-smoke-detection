import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="🔥 Fire & Smoke Detector", layout="centered")

# Load YOLO model
model = YOLO('model_- 8 may 2025 8_52.pt')  # Update with your correct model path

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 38px;
        font-weight: bold;
        color: #FF4500;
        text-align: center;
    }
    .subtitle {
        font-size: 22px;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #333333;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and instruction
st.markdown('<div class="title">🔥 Fire & Smoke Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to detect fire or smoke </div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    st.markdown('<div class="section-header">📷 Original Image:</div>', unsafe_allow_html=True)

    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.image(image, caption="Uploaded Image", channels="BGR")

    # YOLO inference
    results = model(image)
    result = results[0]  # Get first result

    # Display detected image
    detected_image = result.plot()
    st.markdown('<div class="section-header">🧠 Detection Output:</div>', unsafe_allow_html=True)
    st.image(detected_image, caption='Detected Objects (Fire/Smoke)', channels='BGR')

    # Get detection details
    boxes = result.boxes
    if boxes is not None and boxes.xyxy is not None:
        data = {
            'x1': boxes.xyxy[:, 0].cpu().numpy(),
            'y1': boxes.xyxy[:, 1].cpu().numpy(),
            'x2': boxes.xyxy[:, 2].cpu().numpy(),
            'y2': boxes.xyxy[:, 3].cpu().numpy(),
            'Confidence': boxes.conf.cpu().numpy(),
            'Class ID': boxes.cls.cpu().numpy()
        }

        df = pd.DataFrame(data)
        st.markdown('<div class="section-header">📋 Detection Details:</div>', unsafe_allow_html=True)
        st.dataframe(df)
    else:
        st.warning("⚠️ No fire or smoke detected in the image.")
