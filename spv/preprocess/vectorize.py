# spv/preprocess/vectorize.py
import cv2
import numpy as np

def vectorize_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    # morphology refine
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, 1)
    edges = cv2.erode(edges, kernel, 1)

    # convert to 3-channel for YOLO
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def sobel_vector_filter(img, ksize=3):
    """Sobel magnitude normalized and returned as 3-channel BGR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(sx*sx + sy*sy)
    if mag.max() > 0:
        mag = (mag / (mag.max() + 1e-9) * 255.0).astype('uint8')
    else:
        mag = mag.astype('uint8')
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
