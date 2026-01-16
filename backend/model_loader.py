from ultralytics import YOLO

# Load once at server start
tool_model = YOLO("yolov8n.pt")

def load_models():
    return {
        "tool_model": tool_model
    }
