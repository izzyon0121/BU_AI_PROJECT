# spv/preprocess/preprocess_utils.py
import cv2
import numpy as np
from .vectorize import sobel_vector_filter
from collections import deque
from config.settings import (
    DEFAULT_BRIGHT_LOW, DEFAULT_CONTRAST_LOW,
    DEFAULT_NOISE_HIGH, DEFAULT_SHARPEN_ALPHA
)

# DEFAULT_PARAMS exposed for import
DEFAULT_PARAMS = {
    'bright_low': DEFAULT_BRIGHT_LOW,
    'contrast_low': DEFAULT_CONTRAST_LOW,
    'noise_high': DEFAULT_NOISE_HIGH,
    'noise_low': 8.0,
    'clahe_clip': 2.0,
    'clahe_grid': (8,8),
    'sharpen_alpha': DEFAULT_SHARPEN_ALPHA,
    'color_lower': (0, 0, 0),
    'color_upper': (180, 255, 255),
    'use_vector': True
}

# ---------- basic ops ----------
def apply_clahe_bgr(img, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def color_filter_hsv(img, lower_hsv=(0,0,0), upper_hsv=(180,255,255), invert=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    if invert:
        mask = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    blended = cv2.addWeighted(img, 0.6, res, 0.4, 0)
    return blended

def laplacian_sharpen(img, alpha=0.7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    lap3 = cv2.cvtColor(lap_norm, cv2.COLOR_GRAY2BGR)
    sharp = cv2.addWeighted(img.astype('float32'), 1.0 + alpha, lap3.astype('float32'), -alpha, 0)
    sharp = np.clip(sharp, 0, 255).astype('uint8')
    return sharp

# ---------- metrics ----------
def brightness_score(img):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    return float(np.mean(y))

def contrast_score(img):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    return float(np.std(y))

def noise_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(lap))

# ---------- adaptive pipeline ----------
def adaptive_preprocess(img, params=None):
    """
    Apply adaptive preprocessing based on image metrics.
    Returns: processed_img, applied_steps(list)
    """
    if params is None:
        params = DEFAULT_PARAMS

    applied = []
    out = img.copy()

    b = brightness_score(out)
    c = contrast_score(out)
    n = noise_score(out)

    # dark -> CLAHE
    if b < params['bright_low']:
        out = apply_clahe_bgr(out, clip_limit=params['clahe_clip'], tile_grid_size=params['clahe_grid'])
        applied.append('clahe')

    # low contrast -> mild sharpen
    if c < params['contrast_low']:
        out = laplacian_sharpen(out, alpha=params['sharpen_alpha'])
        applied.append('sharpen')

    # high noise -> mild denoise (gauss)
    if n > params['noise_high']:
        out = cv2.GaussianBlur(out, (3,3), 0)
        applied.append('gauss_blur_for_noise')

    # color filter heuristic
    if (c < params['contrast_low']) and (b > params['bright_low']):
        out = color_filter_hsv(out, lower_hsv=params['color_lower'], upper_hsv=params['color_upper'], invert=False)
        applied.append('color_filter')

    # vector filter if not noisy
    if n < params['noise_low'] and params.get('use_vector', True):
        vf = sobel_vector_filter(out, ksize=3)
        out = cv2.addWeighted(out, 0.8, vf, 0.2, 0)
        applied.append('vector_filter')

    return out, applied
