# spv/preprocess/shift_register.py
from collections import deque
import numpy as np

class ShiftRegister:
    def __init__(self, maxlen=5):
        self.buf = deque(maxlen=maxlen)

    def push(self, frame):
        self.buf.append(frame.copy())

    def avg(self):
        if not self.buf:
            return None
        arr = np.stack([f.astype('float32') for f in self.buf], axis=0)
        return np.mean(arr, axis=0).astype('uint8')

    def clear(self):
        self.buf.clear()
