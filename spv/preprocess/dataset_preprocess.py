# spv/preprocess/dataset_preprocess.py
from pathlib import Path
import yaml

TEMPLATE = """# YOLO dataset YAML (kick_safety)
path: {root}
train: images/train
val: images/val
names:
  0: 2-person_with_kickboard
  1: helmet_O
  2: helmet_X
  3: person_with_kickboard
"""

def make_dataset_yaml(yaml_path: str):
    p = Path(yaml_path)
    base = p.parent
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE.format(root=base.as_posix()), encoding="utf-8")
    return p