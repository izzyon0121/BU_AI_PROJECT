# run.py
"""
Interactive runner for SPV Kickboard project.

Usage:
    python run.py
Then enter a menu number:
    1) Make dataset.yaml template
    2) Train model
    3) Predict (file/folder/video)
    4) Run webcam (real-time)
    0) Exit
"""
import sys
import traceback
import time
import os
import glob
import shutil
import cv2
import numpy as np
import traceback

from spv.inference.yolo_detector import YoloDetector
from spv.inference.draw import draw_boxes
from spv.preprocess.preprocess_utils import adaptive_preprocess   # returns (img, applied_steps)          # returns img
from spv.preprocess.shift_register import ShiftRegister
from config.paths import DATASET_PATH, WEIGHTS_DIR
from config.settings import DEFAULT_IMG_SIZE, CONF_THRES

# prefer high-level pipeline functions if available
try:
    from spv.pipeline.pipeline import train_model, infer_on_source, run_webcam
    has_pipeline = True
except Exception:
    has_pipeline = False

# fallback: try to call YoloDetector directly if pipeline missing
try:
    from spv.inference.yolo_detector import YoloDetector
    has_detector = True
except Exception:
    has_detector = False

# attempt to import dataset yaml maker
try:
    from spv.preprocess.dataset_preprocess import make_dataset_yaml
    has_make_yaml = True
except Exception:
    has_make_yaml = False

def input_with_default(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s != "" else default

def make_yaml_action():
    if not has_make_yaml:
        print("dataset yaml generator not found (make_dataset_yaml).")
        return
    out = input_with_default("Write dataset yaml to (path)", "config/dataset.yaml")
    try:
        p = make_dataset_yaml(out)
        print(f"✔ dataset yaml written to {p}")
    except Exception as e:
        print("✖ failed to make dataset yaml:", e)
        traceback.print_exc()

def train_action():
    default_weights = "yolov8n.pt"
    weights = input_with_default("Weights path", default_weights)
    data_yaml = input_with_default("Data YAML path", "config/dataset.yaml")
    epochs = int(input_with_default("Epochs", "30"))
    imgsz = int(input_with_default("Image size", str(DEFAULT_IMG_SIZE)))
    batch = int(input_with_default("Batch size", "8"))
    run_name = input_with_default("Run name", "kick_train")

    print("Starting training with:")
    print(f"  weights={weights}, data={data_yaml}, epochs={epochs}, imgsz={imgsz}, batch={batch}, run_name={run_name}")

    try:
        if has_pipeline:
            train_model(data_yaml, weights=weights, epochs=epochs, imgsz=imgsz, batch=batch, run_name=run_name)
        elif has_detector:
            det = YoloDetector(weights=weights, device="cpu")
            det.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, run_name=run_name)
        else:
            print("No training function available (pipeline or detector).")
    except Exception as e:
        print("Training failed:", e)
        traceback.print_exc()

def predict_action():
    default_weights = input_with_default("Weights path", "runs/detect/kick_train/weights/best.pt")
    source = input_with_default("Source (file/folder/video)", "samples")
    imgsz = int(input_with_default("Image size", str(DEFAULT_IMG_SIZE)))
    conf = float(input_with_default("Conf threshold", str(CONF_THRES)))
    run_name = input_with_default("Run name", "kick_predict")

    print("Starting prediction:")
    print(f"  weights={default_weights}, source={source}, imgsz={imgsz}, conf={conf}, run_name={run_name}")

    out_dir = os.path.abspath(os.path.join("runs", "detect", run_name))
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------
    # 전처리 & 추론 & 시각화 함수
    # ------------------------------------------
    def manual_predict_frame(det: YoloDetector, img_bgr):
        proc = img_bgr.copy()

        # 1) adaptive preprocess
        try:
            proc, _ = adaptive_preprocess(proc)
        except Exception:
            pass

        # 2) vectorize (선택적으로 적용)
        try:
            proc = vectorize_img(proc)
        except Exception:
            pass

        # 3) inference
        res = det.infer_frame(proc, imgsz=imgsz, conf=conf)

        # 4) box 정리
        persons = []
        if res and getattr(res, "boxes", None) is not None:
            for b in res.boxes:
                xyxy = b.xyxy.cpu().numpy()[0]
                cls = int(b.cls.cpu().numpy()[0])
                confb = float(b.conf.cpu().numpy()[0])
                if confb < conf:
                    continue
                if cls == 0:  # class 0 = person
                    persons.append(xyxy)

        # 5) 결과 시각화
        vis = img_bgr.copy()
        try:
            draw_boxes(vis, persons, labels=None)
        except Exception:
            pass

        return vis

    # ---------------------------------------------
    # Detector 준비
    # ---------------------------------------------
    det = YoloDetector(weights=default_weights, device="cpu")

    # ---------------------------------------------
    # 📂 1) source가 폴더인 경우
    # ---------------------------------------------
    if os.path.isdir(source):
        imgs = glob.glob(os.path.join(source, "*.jpg")) + \
               glob.glob(os.path.join(source, "*.png"))
        if len(imgs) == 0:
            print("No images found in folder.")
            return

        for img_path in imgs:
            img = cv2.imread(img_path)
            vis = manual_predict_frame(det, img)
            save_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, vis)

        print(f"Finished. Saved to {out_dir}")
        return

    # ---------------------------------------------
    # 🖼️ 2) source가 이미지 파일인 경우
    # ---------------------------------------------
    if os.path.isfile(source):
        img = cv2.imread(source)
        vis = manual_predict_frame(det, img)
        save_path = os.path.join(out_dir, os.path.basename(source))
        cv2.imwrite(save_path, vis)
        print(f"Saved: {save_path}")
        return

    # ---------------------------------------------
    # 🎥 3) source가 비디오인 경우
    # ---------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Failed to open video source:", source)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_video_path = os.path.join(out_dir, "prediction.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = manual_predict_frame(det, frame)
        writer.write(vis)

    writer.release()
    cap.release()

    print(f"Video saved to {out_video_path}")


def webcam_action():
    default_weights = input_with_default("Weights path", "runs/detect/kick_train/weights/best.pt")
    cam_id_str = input_with_default("Camera ID (0 for default)", "0")
    try:
        cam_id = int(cam_id_str)
    except:
        cam_id = cam_id_str  # allow RTSP/url
    imgsz = int(input_with_default("Image size", str(DEFAULT_IMG_SIZE)))
    conf = float(input_with_default("Conf threshold", str(CONF_THRES)))
    use_adaptive = input_with_default("Use adaptive preprocess? (y/n)", "y").lower().startswith("y")
    use_shift = input_with_default("Use shift register? (y/n)", "y").lower().startswith("y")
    sr_len = int(input_with_default("Shift-register length", "5"))

    print("Starting webcam mode:")
    print(f"  weights={default_weights}, cam_id={cam_id}, imgsz={imgsz}, conf={conf}, adaptive={use_adaptive}, shift={use_shift}")

    try:
        if has_pipeline:
            run_webcam(weights=default_weights, cam_id=cam_id, imgsz=imgsz, conf=conf,
                       use_adaptive=use_adaptive, use_shift=use_shift, sr_len=sr_len)
        elif has_detector:
            # minimal webcam loop using YoloDetector + optional preprocess
            from spv.preprocess.preprocess_utils import adaptive_preprocess
            from spv.preprocess.shift_register import ShiftRegister
            from spv.postprocess.analyze import analyze_detections
            from spv.inference.draw import draw_boxes
            det = YoloDetector(weights=default_weights, device="cpu")
            cap = None
            import cv2
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                print(f"Failed to open camera {cam_id}")
                return
            sr = ShiftRegister(maxlen=sr_len) if use_shift else None
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
                    res = det.infer_frame(proc, imgsz=imgsz, conf=conf)
                    info = analyze_detections(res, conf_thres=conf)
                    draw_boxes(frame, info.get("persons", []), labels=None, colors=(0,200,0))
                    draw_boxes(frame, info.get("two_person_boxes", []), labels=None, colors=(0,0,255))
                    status = f"People: {info.get('headcount',0)} | 2-person: {info.get('two_person_detected')}"
                    cv2.putText(frame, status, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
                    if applied:
                        cv2.putText(frame, ",".join(applied), (10,56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,255), 1)
                    cv2.imshow("SPV Kickboard (Webcam)", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
        else:
            print("No pipeline or detector available to run webcam.")
    except Exception as e:
        print("Webcam mode failed:", e)
        traceback.print_exc()

def main_menu():
    print("\nSPV Kickboard - Interactive Runner")
    print("1) Make dataset.yaml template")
    print("2) Train model")
    print("3) Predict (file/folder/video)")
    print("4) Run webcam (real-time)")
    print("0) Exit")

def main():
    while True:
        main_menu()
        choice = input("Select option (0-4): ").strip()
        if choice == "0":
            print("Exit.")
            break
        elif choice == "1":
            make_yaml_action()
        elif choice == "2":
            train_action()
        elif choice == "3":
            predict_action()
        elif choice == "4":
            webcam_action()
        else:
            print("Invalid choice. Enter 0-4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)

