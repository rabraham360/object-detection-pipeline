import torch
import cv2 as cv
import torchvision as tv
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights as SSD

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

weights = SSD.DEFAULT
model = tv.models.detection.ssdlite320_mobilenet_v3_large(weights=weights).to(device)

model.eval()

cam = cv.VideoCapture(0)

with torch.inference_mode():
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        batchedTensor = (((torch.from_numpy(cv.cvtColor(frame, cv.COLOR_BGR2RGB))).permute(2,0,1)).float()/255.0).unsqueeze(0).to(device)

        predictions = model(batchedTensor)[0]

        keep = predictions["scores"] >= 0.5

        boxes = predictions["boxes"][keep].cpu().numpy()
        labels = predictions["labels"][keep].cpu().numpy()
        scores = predictions["scores"][keep].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow("Detection Stream",frame)

        if(cv.waitKey(1)==ord("q")):
                break
        
        
cam.release()
cv.destroyAllWindows()