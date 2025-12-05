# spv/inference/draw.py
import cv2

def draw_boxes(img, boxes, labels=None, color=(0,255,0), colors=None, thickness=2):
    """
    Draw bounding boxes on image.
    - Accepts both `color` and `colors` keyword names (compatibility).
    - boxes: iterable of [x1,y1,x2,y2] (pixel coords) OR dicts like {"box": [x1,y1,x2,y2], ...}
    - labels: iterable of strings (optional)
    - color/colors: BGR tuple or single tuple. If you pass a list of colors per box, it will try to use per-box color.
    """
    # prefer explicit 'colors' if provided (backcompat), else use 'color'
    if colors is not None:
        use_color = colors
    else:
        use_color = color

    if boxes is None:
        return img

    # support single color tuple or iterable of colors
    per_box_colors = None
    if isinstance(use_color, (list, tuple)) and len(use_color) > 0 and not isinstance(use_color[0], (int, float)):
        # assume it's a list of colors if first element looks like a tuple/list
        per_box_colors = use_color

    for i, b in enumerate(boxes):
        # normalize different box formats
        try:
            if isinstance(b, dict) and "box" in b:
                x1, y1, x2, y2 = map(int, b["box"])
            else:
                # handle torch tensors or numpy arrays
                x1, y1, x2, y2 = map(int, b)
        except Exception:
            # skip malformed entries
            continue

        # choose color for this box
        if per_box_colors:
            try:
                box_color = tuple(map(int, per_box_colors[i]))
            except Exception:
                box_color = tuple(map(int, per_box_colors[0]))
        else:
            # ensure BGR tuple of ints
            try:
                box_color = tuple(map(int, use_color))
            except Exception:
                box_color = (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

        # draw label if provided
        if labels:
            try:
                txt = str(labels[i])
            except Exception:
                txt = str(labels)
            # background for readability
            (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), box_color, -1)
            cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return img
