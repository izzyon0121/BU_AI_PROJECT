# spv/utils/video_writer.py
import cv2
class VideoWriter:
    def __init__(self, path, fps=20.0, size=(640,480)):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
