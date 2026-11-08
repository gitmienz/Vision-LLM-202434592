import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="src/models/YOLO/yolo11n.pt", conf_thresh=0.25, iou_thresh=0.5):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, frame):
        height, width, _ = frame.shape
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
            classes=None
        )[0]
        
        detected_boxes = []
        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            pos_y = "상단" if cy < height * 0.33 else ("중간" if cy < height * 0.66 else "하단")
            pos_x = "좌측" if cx < width * 0.33 else ("중앙" if cx < width * 0.66 else "우측")
            
            detected_boxes.append({
                "id": idx,
                "location": f"{pos_y} {pos_x}",
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy]
            })
        return detected_boxes
