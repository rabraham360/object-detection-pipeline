import streamlit as st
import cv2 as cv
import torch
import torchvision as tv
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights as SSD

st.title("Real-Time Object Detection")


@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    weights = SSD.DEFAULT
    model = tv.models.detection.ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    model.eval()
    return model, device

model, device = load_model()


confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
run = st.checkbox("Run Webcam")
frame_placeholder = st.empty()

if run:
    cam = cv.VideoCapture(0)
    
    with torch.inference_mode():
        while cam.isOpened() and run:
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break


            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            batched_tensor = (torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)


            predictions = model(batched_tensor)[0]
            keep = predictions["scores"] >= confidence_threshold

            boxes = predictions["boxes"][keep].cpu().numpy()
            labels = predictions["labels"][keep].cpu().numpy()
            scores = predictions["scores"][keep].cpu().numpy()


            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                cv.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(rgb_frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            frame_placeholder.image(rgb_frame, channels="RGB")

    cam.release()