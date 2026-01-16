import cv2
import numpy as np
import shutil
import os
import math
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# IMPORT YOLO (CRITICAL)
from ultralytics import YOLO

# --- CONFIGURATION ---
PROCEDURE_PROFILES = {
    "laparoscopy": {
        "blur_th": 100.0,    
        "blood_th": 0.05,
        "smoke_th": 30.0,
        "speed_th": 20.0,
        "critical_radius": 120
    }
}
CURRENT_PROCEDURE = "laparoscopy"
profile = PROCEDURE_PROFILES[CURRENT_PROCEDURE]

# --- 1. INITIALIZE APP & MODEL ---
app = FastAPI()

# Enable CORS (Allows Frontend to talk to Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SAFETY CHECK: Create temp folder if missing
if not os.path.exists("temp_videos"):
    os.makedirs("temp_videos")

# LOAD MODEL SAFELY
print("⏳ Loading YOLO Model... (This might take 30s the first time)")
try:
    model = YOLO("yolov8n.pt") 
    print("✅ YOLO Model Loaded Successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load YOLO. {e}")
    print("Run 'pip install ultralytics' in terminal.")

# --- 2. IMAGE PROCESSING LOGIC ---
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def assess_visibility(frame):
    # BLUR (Laplacian Variance)
    gray = preprocess(frame)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # BLOOD (Red HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
           cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
    blood = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
    
    # SMOKE (Std Dev)
    smoke = np.std(gray)

    reasons = []
    if blur < profile["blur_th"]: reasons.append("BLUR")
    if blood > profile["blood_th"]: reasons.append("BLOOD")
    if smoke < profile["smoke_th"]: reasons.append("SMOKE")

    if (blur < profile["blur_th"] * 0.7 or blood > profile["blood_th"] * 1.5):
        visibility = "POOR"
    elif (blur < profile["blur_th"] or blood > profile["blood_th"]):
        visibility = "MODERATE"
    else:
        visibility = "GOOD"

    # Calc Score (0-100)
    score = int(100 - (len(reasons) * 25))
    return visibility, reasons, score, blood, smoke

# --- 3. YOLO TOOL DETECTION ---
def detect_tools(frame):
    results = model(frame, verbose=False)
    tools = []
    for box in results[0].boxes:
        if float(box.conf[0]) > 0.3: # Confidence Threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tools.append([x1, y1, x2, y2, float(box.conf[0])])
    return tools

def get_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

def assess_tool_safety(tools, threshold):
    for i in range(len(tools)):
        for j in range(i+1, len(tools)):
            dist = math.dist(get_center(tools[i]), get_center(tools[j]))
            if dist < threshold: return "RISK"
    return "SAFE"

def contextual_risk(visibility, safety_flags):
    if visibility == "POOR": return "CRITICAL" if safety_flags else "HIGH"
    if visibility == "MODERATE": return "HIGH" if safety_flags else "MODERATE"
    return "LOW" if safety_flags else "SAFE"

# --- 4. API ENDPOINT ---
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.filename}")
    temp_path = f"temp_videos/{file.filename}"
    
    try:
        # Save Video
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        events = []
        stats = {
            "good_vis": 0, "moderate_vis": 0, "poor_vis": 0,
            "tool_collision": 0, "critical_region": 0, "fast_tool": 0,
            "reduced_visibility_risk": 0, "total_risk_events": 0, "total_frames": 0,
            "total_blood": 0.0, "total_smoke": 0.0
        }

        frame_index = 0
        skip_rate = 5 # Analyze every 5th frame to prevent timeouts
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_index += 1
            if frame_index % skip_rate != 0: continue 
            
            # A. VISIBILITY
            vis, reasons, conf, b_val, s_val = assess_visibility(frame)
            stats["total_frames"] += 1
            stats["total_blood"] += b_val
            stats["total_smoke"] += s_val
            
            if vis == "GOOD": stats["good_vis"] += 1
            elif vis == "MODERATE": stats["moderate_vis"] += 1
            else: stats["poor_vis"] += 1

            # B. YOLO TOOL DETECTION
            tools = detect_tools(frame)
            safety_flags = []
            crit_region_box = None

            # C. SAFETY RULES
            # 1. Collision
            if assess_tool_safety(tools, profile["critical_radius"]) == "RISK":
                safety_flags.append("TOOL_PROXIMITY")
                stats["tool_collision"] += 1
                stats["total_risk_events"] += 1
                if vis != "GOOD": stats["reduced_visibility_risk"] += 1
                if tools: crit_region_box = tools[0][:4] # Use first tool as alert box

            # 2. Critical Region (Simulated by tighter proximity)
            if assess_tool_safety(tools, profile["critical_radius"] * 0.6) == "RISK":
                safety_flags.append("CRITICAL_REGION")
                stats["critical_region"] += 1
                stats["total_risk_events"] += 1
                if tools: crit_region_box = tools[0][:4]

            # D. RISK LEVEL
            risk = contextual_risk(vis, safety_flags)

            # E. SAVE EVENT
            event = {
                "time": frame_index / fps,
                "risk_level": risk,
                "visibility": vis,
                "vis_score": conf,
                "vis_reason": ", ".join(reasons),
                "safety_flags": safety_flags,
                "tools": [{"bbox": t[:4], "confidence": t[4]} for t in tools],
                "crit_region": crit_region_box
            }
            events.append(event)

        cap.release()
        os.remove(temp_path) # Clean up

        # F. GENERATE REPORT
        total = max(1, stats["total_frames"])
        overall_risk = "LOW"
        if stats["total_risk_events"] > 2: overall_risk = "MODERATE"
        if stats["total_risk_events"] > 5: overall_risk = "HIGH"

        report = {
            "duration_sec": int(frame_index / fps),
            "overall_risk": overall_risk,
            "vis_breakdown": {
                "good": int((stats["good_vis"]/total)*100),
                "moderate": int((stats["moderate_vis"]/total)*100),
                "poor": int((stats["poor_vis"]/total)*100)
            },
            "event_counts": {
                "TOOL_PROXIMITY": stats["tool_collision"],
                "CRITICAL_REGION": stats["critical_region"],
                "HIGH_TOOL_SPEED": stats["fast_tool"]
            },
            "reduced_visibility_risk": stats["reduced_visibility_risk"],
            "total_risk_events": stats["total_risk_events"]
        }

        print("✅ Analysis Complete. Sending Data...")
        return JSONResponse(content={
            "status": "success",
            "blood_ratio": int(stats["total_blood"] / total * 100),
            "smoke_score": int(stats["total_smoke"] / total * 100),
            "events": events,
            "report": report
        })

    except Exception as e:
        print(f"❌ SERVER CRASH ERROR: {e}")
        # Return error as JSON so Frontend can see it
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)