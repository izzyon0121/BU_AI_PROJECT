# spv/utils/timer.py
import time

class Timer:
    def __init__(self):
        self._t0 = None

    def start(self):
        self._t0 = time.time()

    def elapsed(self):
        if self._t0 is None:
            return 0.0
        return time.time() - self._t0
