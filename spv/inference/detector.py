# spv/inference/detector.py
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", device: str = "cpu"):
        self.weights = weights
        self.device = device
        self.model = YOLO(weights)

    def train(self, data: str, epochs: int = 30, imgsz: int = 640, batch: int = 8, run_name: str = "train"):
        self.model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=self.device, name=run_name, exist_ok=True)

    def predict(self, source: str, imgsz: int = 640, conf: float = 0.25, run_name: str = "predict", save: bool = True):
        return self.model.predict(source=source, imgsz=imgsz, conf=conf, device=self.device, save=save, name=run_name, exist_ok=True)

    def infer_frame(self, frame, imgsz: int = 640, conf: float = 0.25):
        results = self.model(frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False)
        return results[0] if len(results) else None
