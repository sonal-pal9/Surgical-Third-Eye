import math

# -------------------------------
# Helper functions
# -------------------------------

def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# -------------------------------
# ALERT RULES
# -------------------------------

def tool_tool_collision(tools, iou_threshold=0.3):
    """
    tools: list of dicts with key 'bbox'
    """
    alerts = []
    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            overlap = iou(tools[i]["bbox"], tools[j]["bbox"])
            if overlap > iou_threshold:
                alerts.append({
                    "type": "tool_tool_collision",
                    "confidence": round(overlap, 2)
                })
    return alerts


def fast_tool_motion(prev_tools, curr_tools, speed_threshold=40):
    """
    prev_tools, curr_tools: list of dicts with key 'bbox'
    """
    alerts = []

    for p, c in zip(prev_tools, curr_tools):
        cx1, cy1 = box_center(p["bbox"])
        cx2, cy2 = box_center(c["bbox"])

        speed = math.dist((cx1, cy1), (cx2, cy2))
        if speed > speed_threshold:
            alerts.append({
                "type": "fast_tool_motion",
                "confidence": round(min(speed / 100, 1.0), 2)
            })

    return alerts


def tool_critical_region(tools, frame_shape, margin_ratio=0.2):
    """
    Critical region = center area of frame
    """
    h, w = frame_shape[:2]

    cx1 = int(w * margin_ratio)
    cy1 = int(h * margin_ratio)
    cx2 = int(w * (1 - margin_ratio))
    cy2 = int(h * (1 - margin_ratio))

    critical_box = (cx1, cy1, cx2, cy2)

    alerts = []
    for t in tools:
        overlap = iou(t["bbox"], critical_box)
        if overlap > 0.1:
            alerts.append({
                "type": "tool_near_critical_region",
                "confidence": round(overlap, 2)
            })

    return alerts


def visibility_alert(smoke_score, blood_ratio):
    if smoke_score < 40 or blood_ratio > 0.02:
        return "POOR"
    elif smoke_score < 60:
        return "MODERATE"
    return "GOOD"
