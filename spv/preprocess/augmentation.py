# spv/preprocess/augmentation.py
import cv2
import numpy as np
import random

def random_augment(img, p=0.5):
    out = img.copy()
    if random.random() < 0.3:
        # random brightness/contrast
        alpha = 1.0 + (random.uniform(-0.2, 0.2))
        beta = random.uniform(-20, 20)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if random.random() < 0.2:
        # gaussian blur
        k = random.choice([3,5])
        out = cv2.GaussianBlur(out, (k,k), 0)
    if random.random() < 0.2:
        # small rotate
        h,w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2), random.uniform(-8,8), 1.0)
        out = cv2.warpAffine(out, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    return out
