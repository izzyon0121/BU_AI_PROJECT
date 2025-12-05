# spv/pipeline/pipeline.py
import cv2
import numpy as np
import time
from ..inference import YoloDetector
from ..preprocess.preprocess_utils import adaptive_preprocess
from ..preprocess.shift_register import ShiftRegister
from ..postprocess.analyze import analyze_detections
from ..inference.draw import draw_boxes
from config.settings import DEFAULT_IMG_SIZE

def train_model(data_yaml: str, weights: str = "yolov8n.pt", epochs: int = 30, imgsz: int = DEFAULT_IMG_SIZE, batch: int = 8, run_name: str = "train"):
    det = YoloDetector(weights=weights, device="cpu")
    det.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, run_name=run_name)
    return det

def infer_on_source(weights: str, source: str, imgsz: int = DEFAULT_IMG_SIZE, conf: float = 0.25, run_name: str = "predict"):
    det = YoloDetector(weights=weights, device="cpu")
    return det.predict(source=source, imgsz=imgsz, conf=conf, run_name=run_name, save=True)

def run_webcam(weights: str, cam_id: int = 0, imgsz: int = DEFAULT_IMG_SIZE, conf: float = 0.25,
               use_adaptive: bool = True, use_shift: bool = False, sr_len: int = 5, window_title: str = "SPV Kickboard (CPU)"):
    det = YoloDetector(weights=weights, device="cpu")
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {cam_id}")

    fps_hist = []
    sr = ShiftRegister(maxlen=sr_len) if use_shift else None
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            proc = frame.copy()
            applied = []
            if use_adaptive:
                proc, applied = adaptive_preprocess(proc)

            if sr is not None:
                sr.push(proc)
                avg = sr.avg()
                if avg is not None:
                    proc = avg

            # inference (ultralytics accepts BGR numpy)
            t0 = time.time()
            res = det.infer_frame(proc, imgsz=imgsz, conf=conf)
            info = analyze_detections(res, conf_thres=conf)

            # draw detections (we draw on the original frame for display)
            draw_boxes(frame, info.get("persons", []), labels=None, colors=(0,200,0))
            draw_boxes(frame, info.get("two_person_boxes", []), labels=None, colors=(0,0,255))

            # overlay status & applied preprocessing steps
            status = f"People: {info.get('headcount',0)} | 2-person: {info.get('two_person_detected')}"
            cv2.putText(frame, status, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
            if applied:
                cv2.putText(frame, ",".join(applied), (10,56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,255), 1)

            # FPS (smooth over history)
            fps = 1.0 / max(1e-6, time.time() - t0)
            fps_hist.append(fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            avg_fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10,92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
