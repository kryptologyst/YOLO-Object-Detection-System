# YOLO Object Detection Configuration
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATABASE_PATH = PROJECT_ROOT / "detections.db"

# Model configuration
MODEL_CONFIG = {
    "default_model": "yolov8n.pt",  # nano version for speed
    "available_models": [
        "yolov8n.pt",  # nano - fastest
        "yolov8s.pt",  # small
        "yolov8m.pt",  # medium
        "yolov8l.pt",  # large
        "yolov8x.pt",  # extra large - most accurate
        "yolov11n.pt", # latest nano
        "yolov11s.pt", # latest small
    ],
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 1000,
}

# Detection settings
DETECTION_CONFIG = {
    "save_results": True,
    "save_crops": False,
    "save_txt": False,
    "save_conf": True,
    "line_width": 2,
    "font_size": 0.5,
    "box_color": (0, 255, 0),  # Green
    "text_color": (255, 0, 255),  # Magenta
}

# Web UI configuration
UI_CONFIG = {
    "title": "YOLO Object Detection System",
    "description": "Real-time object detection using YOLOv8/YOLOv11",
    "theme": "default",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".mp4", ".avi", ".mov"],
}

# Database configuration
DATABASE_CONFIG = {
    "table_name": "detections",
    "max_records": 10000,
    "cleanup_days": 30,
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)
