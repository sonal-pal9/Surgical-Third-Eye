# import cv2
# import numpy as np
# import math

# # --- 1. TOOL DETECTION ---
# def detect_tools(frame, model):
#     results = model(frame, verbose=False)
#     tools = []
#     for box in results[0].boxes:
#         conf = float(box.conf[0])
#         if conf > 0.4:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             tools.append({
#                 "bbox": [x1, y1, x2, y2], # List format for JSON compatibility
#                 "confidence": round(conf, 2)
#             })
#     return tools

# # --- 2. BLOOD DETECTION ---
# def blood_ratio(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
#     lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
#     mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
#     return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

# # --- 3. SMOKE DETECTION ---
# def smoke_score(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return np.std(cv2.GaussianBlur(gray, (5, 5), 0))

# # --- 4. HELPERS ---
# def iou(boxA, boxB):
#     xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
#     xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter <= 0: return 0.0
#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     return inter / float(areaA + areaB - inter)

# # --- MAIN PIPELINE ---
# def run_inference(frame, models):
#     alerts = []
    
#     # Run Detections
#     tools = detect_tools(frame, models["tool_model"])
#     blood = blood_ratio(frame)
#     smoke = smoke_score(frame)

#     # Critical Region Logic (Center 25% of screen)
#     h, w = frame.shape[:2]
#     # We define the box here for logic, but frontend will draw the circle
#     critical_box = [int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]
    
#     for t in tools:
#         if iou(t['bbox'], critical_box) > 0.1:
#             alerts.append({"type": "Critical Region Entry", "confidence": 0.95})

#     # Visibility Logic
#     if smoke < 45 or blood > 0.03: visibility = "POOR"
#     elif smoke < 60: visibility = "MODERATE"
#     else: visibility = "GOOD"

#     return {
#         "visibility": visibility,
#         "blood_ratio": blood,
#         "smoke_score": smoke,
#         "tool_count": len(tools),
#         "raw_tools": tools,  # CRITICAL: Sends coordinates to frontend
#         "alerts": alerts
#     }

import cv2
import numpy as np
import math

# --- CONFIGURATION PROFILES ---
PROCEDURE_PROFILES = {
    "laparoscopy": { "blur_th": 55, "blood_th": 0.12, "smoke_th": 30, "speed_th": 45, "critical_radius_factor": 0.25 },
    "endoscopy":   { "blur_th": 70, "blood_th": 0.18, "smoke_th": 25, "speed_th": 60, "critical_radius_factor": 0.20 },
    "robotic":     { "blur_th": 50, "blood_th": 0.10, "smoke_th": 35, "speed_th": 30, "critical_radius_factor": 0.15 }
}

# --- 1. DETECTION FUNCTIONS ---
def detect_tools(frame, model):
    results = model(frame, verbose=False)
    tools = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tools.append({"bbox": [x1, y1, x2, y2], "confidence": round(conf, 2)})
    return tools

def blood_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

def smoke_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.std(cv2.GaussianBlur(gray, (5, 5), 0))

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter / float((boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter) if inter > 0 else 0

# --- 2. RISK LOGIC ---
def assess_contextual_risk(visibility, safety_flags):
    # Logic: Bad visibility amplifies other risks
    if "CRITICAL_REGION" in safety_flags: return "CRITICAL"
    
    if visibility == "POOR":
        if safety_flags: return "HIGH" # Poor vis + any issue = High Risk
        else: return "MODERATE"
        
    if visibility == "MODERATE":
        if len(safety_flags) >= 2: return "HIGH"
        if safety_flags: return "MODERATE"
        
    if safety_flags: return "LOW"
    return "SAFE"

# --- MAIN PIPELINE ---
prev_tools = []

def run_inference(frame, models, procedure_type="laparoscopy"):
    global prev_tools
    profile = PROCEDURE_PROFILES.get(procedure_type, PROCEDURE_PROFILES["laparoscopy"])
    
    # 1. Measurements
    tools = detect_tools(frame, models["tool_model"])
    blood = blood_ratio(frame)
    smoke = smoke_score(frame)
    
    # 2. Visibility Assessment
    if smoke < profile["smoke_th"] or blood > profile["blood_th"]: visibility = "POOR"
    elif smoke < (profile["smoke_th"] + 20): visibility = "MODERATE"
    else: visibility = "GOOD"

    # 3. Safety Flags
    safety_flags = []
    
    # A. Critical Region
    h, w = frame.shape[:2]
    rf = profile["critical_radius_factor"]
    crit_box = [int(w*rf), int(h*rf), int(w*(1-rf)), int(h*(1-rf))]
    for t in tools:
        if iou(t['bbox'], crit_box) > 0.1:
            if "CRITICAL_REGION" not in safety_flags: safety_flags.append("CRITICAL_REGION")

    # B. Tool Speed
    if prev_tools and tools:
        # Simple speed check (center movement)
        c_prev = [(t['bbox'][0]+t['bbox'][2])//2 for t in prev_tools]
        c_curr = [(t['bbox'][0]+t['bbox'][2])//2 for t in tools]
        # Compare first detected tool for simplicity in demo
        if len(c_prev) > 0 and len(c_curr) > 0:
            dist = abs(c_prev[0] - c_curr[0])
            if dist > profile["speed_th"]: safety_flags.append("HIGH_TOOL_SPEED")

    # C. Collision/Proximity
    if len(tools) >= 2:
        # Check distance between tools
        c1 = ((tools[0]['bbox'][0]+tools[0]['bbox'][2])//2, (tools[0]['bbox'][1]+tools[0]['bbox'][3])//2)
        c2 = ((tools[1]['bbox'][0]+tools[1]['bbox'][2])//2, (tools[1]['bbox'][1]+tools[1]['bbox'][3])//2)
        dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        if dist < (w * 0.15): safety_flags.append("TOOL_PROXIMITY")

    prev_tools = tools
    
    # 4. Contextual Risk
    risk_level = assess_contextual_risk(visibility, safety_flags)

    # 1. Define Critical Box based on profile
    h, w = frame.shape[:2]
    rf = profile["critical_radius_factor"]
    
    # Calculate the box coordinates (x1, y1, x2, y2)
    crit_box = [int(w*rf), int(h*rf), int(w*(1-rf)), int(h*(1-rf))]

    # ... existing tool detection logic ...

    return {
        # ... other returns ...
        "critical_box": crit_box, # ADD THIS LINE
        "visibility": visibility,
        # ...
    }

    return {
        "visibility": visibility,
        "blood_ratio": blood,
        "smoke_score": smoke,
        "tool_count": len(tools),
        "raw_tools": tools,
        "risk_level": risk_level,
        "safety_flags": safety_flags,
        "procedure": procedure_type
    }