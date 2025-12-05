# spv/inference/__init__.py
# Re-export common symbols for convenient imports.
# This file should not import heavy modules on import-time except local helpers.

# try to import draw helper if present
try:
    from .draw import draw_boxes
except Exception:
    # fallback: define a stub
    def draw_boxes(img, boxes, labels=None, color=(0,255,0), thickness=2):
        return img

# import YoloDetector from detector.py (must exist)
try:
    from .detector import YoloDetector
except Exception:
    # define a minimal stub so other modules that import won't immediately fail with NameError
    class YoloDetector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("YoloDetector is not available (failed to import detector.py).")
