import os
import sys
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

# Add both DetectionModel and Sequential to safe globals
print("Adding DetectionModel and Sequential to safe globals...")
torch.serialization.add_safe_globals([DetectionModel, Sequential])

def test_model_loading():
    try:
        print("Attempting to load the model...")
        model_path = "runs/train/pothole_detector/weights/best.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            # Try alternative paths
            alternative_paths = ["models/best.pt", "best.pt", "yolov8n.pt"]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found model at alternative path: {alt_path}")
                    model_path = alt_path
                    break
            
        model = YOLO(model_path)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model task: {model.task}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 