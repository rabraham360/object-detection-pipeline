import streamlit as st
import cv2 as cv
import torch
import torchvision as tv
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights as SSD
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.title("Real-Time Live Object Detection")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    weights = SSD.DEFAULT
    model = tv.models.detection.ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    model.eval()
    return model, device

model, device = load_model()

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ObjectDetector(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        batched_tensor = (torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)

        with torch.inference_mode():
            predictions = model(batched_tensor)[0]

        keep = predictions["scores"] >= self.confidence_threshold
        boxes = predictions["boxes"][keep].numpy()
        labels = predictions["labels"][keep].numpy()
        scores = predictions["scores"][keep].numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img, f"Class {label}: {score:.2f}", (x1, max(y1 - 10, 15)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="live-object-detection",
    video_processor_factory=ObjectDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    ctx.video_processor.confidence_threshold = confidence_threshold