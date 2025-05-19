import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Load the YOLO model
model = YOLO('model_- 8 may 2025 8_52.pt')  # Replace with your actual model path

# Streamlit UI
st.title("YOLO Object Detection")
st.write("Upload an image to detect objects:")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.image(image, caption="Original Image", channels="BGR")

    # Run YOLO inference
    results = model(image)
    result = results[0]  # Get the first (and only) result

    # Display detected image
    detected_image = result.plot()
    st.image(detected_image, caption='Detected Image', channels='BGR')

    # Get detection details and display them as a DataFrame
    boxes = result.boxes

    if boxes is not None and boxes.xyxy is not None:
        data = {
            'x1': boxes.xyxy[:, 0].cpu().numpy(),
            'y1': boxes.xyxy[:, 1].cpu().numpy(),
            'x2': boxes.xyxy[:, 2].cpu().numpy(),
            'y2': boxes.xyxy[:, 3].cpu().numpy(),
            'confidence': boxes.conf.cpu().numpy(),
            'class_id': boxes.cls.cpu().numpy()
        }

        df = pd.DataFrame(data)
        st.write("Detected Objects:")
        st.dataframe(df)
    else:
        st.write("No objects detected.")