# spv/inference/yolo_detector.py
# shim to keep backward compatibility with run.py which expects yolo_detector module.
# It simply re-exports YoloDetector from detector.py

from .detector import YoloDetector

__all__ = ["YoloDetector"]
