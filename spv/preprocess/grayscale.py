# spv/preprocess/grayscale.py
import cv2

def to_bgr_grayscale(frame):
    """Convert BGR to single-channel grayscale then back to 3-channel BGR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
