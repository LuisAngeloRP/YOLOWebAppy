import cv2
import streamlit as st
from ultralytics import YOLO
import os

UPLOAD_FOLDER = 'uploads'

st.set_page_config(page_title="Upload Image or Video")

def generate_frames(image_path):
    model = YOLO("tomate2.pt")

    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    results = model.predict(image, conf=0.2)
    result = results[0]

    image_cv = cv2.cvtColor(result.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)

    st.image(image_cv, channels="BGR", use_column_width=True)

def generate_video_frames(video_path):
    model = YOLO("tomate2.pt")

    cap = cv2.VideoCapture(video_path)

    st.warning("Displaying video frames:")

    frame_container = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))
        results = model.predict(frame, conf=0.2)
        result = results[0]

        image_cv = cv2.cvtColor(result.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)

        frame_container.image(image_cv, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    st.title("Upload Image or Video")

    choice = st.sidebar.radio("Select Processing", ["Image", "Video"])

    uploaded_file = st.file_uploader(f"Choose a {choice.lower()}", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if choice == "Image":
            st.image(file_path, channels="BGR", use_column_width=True)
            st.warning("Displaying YOLO processed image:")
            generate_frames(file_path)
        elif choice == "Video":
            generate_video_frames(file_path)
