# spv/postprocess/analyze.py
import numpy as np
from config.settings import CONF_THRES

def analyze_detections(detections, conf_thres=CONF_THRES):
    """
    Simplified: use class index 0 (2-person_with_kickboard) as two-person overload detection.
    Returns dict with headcount (person boxes), two_person_detected (bool), lists of boxes.
    """
    persons = []
    two_person_boxes = []
    other_boxes = []

    if detections is None:
        return {"headcount": 0, "two_person_detected": False, "persons": [], "two_person_boxes": [], "other": []}

    if getattr(detections, "boxes", None) is None:
        return {"headcount": 0, "two_person_detected": False, "persons": [], "two_person_boxes": [], "other": []}

    for b in detections.boxes:
        conf = float(b.conf.cpu().numpy()[0])
        if conf < conf_thres:
            continue
        cls = int(b.cls.cpu().numpy()[0])
        xyxy = b.xyxy.cpu().numpy()[0].tolist()
        if cls == 3:
            persons.append(xyxy)
        elif cls == 0:
            two_person_boxes.append(xyxy)
        else:
            other_boxes.append({"cls": cls, "box": xyxy})

    headcount = len(persons)
    two_person_detected = len(two_person_boxes) > 0

    return {
        "headcount": headcount,
        "two_person_detected": two_person_detected,
        "persons": persons,
        "two_person_boxes": two_person_boxes,
        "other": other_boxes
    }
