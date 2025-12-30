import cv2
import math
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

def detect_tools(frame, conf_thresh=0.4):
    results = model(frame, verbose=False)
    tools = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > conf_thresh:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tools.append((x1, y1, x2, y2))

    return tools


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def tool_distance(box1, box2):
    return math.dist(center(box1), center(box2))


def assess_tool_safety(tools, threshold=80):
    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            if tool_distance(tools[i], tools[j]) < threshold:
                return "RISK"
    return "SAFE"


def draw_tool_alerts(frame, tools, safety):
    for box in tools:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (255, 255, 0), 2)

    if safety == "RISK":
        cv2.putText(frame, "SAFETY RISK ⚠️",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
