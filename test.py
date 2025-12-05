# test_import.py
import importlib, traceback
import numpy as np
print("== test yolo_detector import ==")
try:
    importlib.import_module("spv.inference.yolo_detector")
    print("yolo_detector import OK")
except Exception:
    print("yolo_detector import FAIL")
    traceback.print_exc()

print("== test pipeline import ==")
try:
    importlib.import_module("spv.pipeline.pipeline")
    print("pipeline.pipeline import OK")
except Exception:
    print("pipeline.pipeline import FAIL")
    traceback.print_exc()
# quick_test.py — 빠른 모델/이미지 점검
# quick_test_verbose.py
from ultralytics import YOLO
import os, sys, cv2

WEIGHTS = "runs/detect/kick_train/weights/best.pt"
IMG = "samples"   # 너가 테스트한 이미지 경로로 바꿔

if not os.path.exists(WEIGHTS):
    print("Weights not found:", WEIGHTS); sys.exit(1)
if not os.path.exists(IMG):
    print("Image not found:", IMG); sys.exit(1)

model = YOLO(WEIGHTS)
# use low conf and larger img size to be permissive
res = model(IMG, imgsz=1280, conf=0.05)  
r = res[0]

# print raw boxes info
if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
    print("Boxes count: 0")
else:
    print("Boxes count:", len(r.boxes))
    for i, b in enumerate(r.boxes):
        xyxy = b.xyxy.cpu().numpy()[0]
        cls = int(b.cls.cpu().numpy()[0])
        conf = float(b.conf.cpu().numpy()[0])
        print(f"box[{i}] cls={cls}, conf={conf:.3f}, xyxy={xyxy}")

# save plotted image using OpenCV (r.plot() returns numpy array)
out = "quick_test_out.jpg"
try:
    img_plot = r.plot()  # numpy ndarray (BGR)
    cv2.imwrite(out, img_plot)
    print("Saved plotted result to", out)
except Exception as e:
    print("Failed to save plotted result:", e)
from ultralytics import YOLO
m = YOLO("runs/detect/kick_train/weights/best.pt")
try:
    names = m.model.names
except Exception:
    names = getattr(m, "names", None)
print("model.names:", names)
import os
p = "runs/detect/kick_train/weights/best.pt"
print(os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else None)
from ultralytics import YOLO
import cv2, os
IMG = "samples"
if not os.path.exists(IMG):
    print("image missing:", IMG); raise SystemExit
model = YOLO("yolov8n.pt")  # 로컬에 yolov8n.pt 있어야 함
res = model(IMG, imgsz=1280, conf=0.05)
r = res[0]
print("pretrained boxes:", 0 if r.boxes is None else len(r.boxes))
if r.boxes is not None:
    for b in r.boxes:
        print(" cls", int(b.cls.cpu().numpy()[0]), "conf", float(b.conf.cpu().numpy()[0]))
cv2.imwrite("pretrained_out.jpg", r.plot())
print("saved pretrained_out.jpg")
import cv2, os
img="samples"
lbl=os.path.splitext(img)[0]+".txt"
img0=cv2.imread(img)
h,w=img0.shape[:2]
if not os.path.exists(lbl):
    print("label missing:", lbl); raise SystemExit
with open(lbl) as f:
    for line in f:
        cls, xc, yc, ww, hh = map(float, line.split())
        x1=int((xc-ww/2)*w); y1=int((yc-hh/2)*h)
        x2=int((xc+ww/2)*w); y2=int((yc+hh/2)*h)
        cv2.rectangle(img0,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite("gt_overlay.jpg", img0)
print("Saved gt_overlay.jpg")
