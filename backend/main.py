from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import tempfile
import os

from model_loader import load_models
from inference import run_inference

app = FastAPI()
@app.get("/")
def root():
    return {"status": "Backend is running"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = load_models()

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    # 1. Use async/await to read the file content safely
    content = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(content)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    events_list = [] 
    frame_no = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_no += 1

            # Run inference every 40 frames
            if frame_no % 40 == 0:
                result = run_inference(frame, models)
                timestamp = round(frame_no / fps, 2)

                if result.get("alerts"):
                    # This matches the 'data.events' structure in your JS
                    events_list.append({
                        "time": timestamp,
                        "alerts": result["alerts"], 
                        "visibility": result.get("visibility", "UNKNOWN")
                    })
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # 2. Always cleanup resources
        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

    # 3. Return the response outside the while loop
    return {
        "fps": fps,
        "events": events_list, 
        "overall_visibility": "Clear",
        "blood_ratio": 0.0,
        "smoke_score": 0.0,
        "tool_count": 0

    }
