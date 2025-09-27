# Project 125. YOLO implementation for object detection
# Description:
# YOLO (You Only Look Once) is a high-speed, real-time object detection system that detects multiple objects in a single pass of a convolutional neural network. 
# This project has been modernized to use the latest YOLOv8/YOLOv11 models with enhanced features.

# 🚀 Modern Implementation Features:
# - Latest YOLOv8/YOLOv11 models
# - Web interface with Streamlit
# - Database integration for analytics
# - Batch processing capabilities
# - Video processing support
# - Comprehensive testing suite

# Quick Start Examples:

# 1. Basic Detection (Legacy - for compatibility)
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pre-trained YOLOv8 model (modernized from YOLOv5)
model = YOLO("yolov8n.pt")  # Automatically downloads the model if not present

# Load input image
image_path = "test_image.jpg"  # Replace with your own image
image = cv2.imread(image_path)

if image is not None:
    # Run inference
    results = model(image)
    
    # Parse results and draw bounding boxes
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
        
                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Show image
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.title("YOLOv8 Object Detection")
    plt.axis("off")
    plt.show()
else:
    print("❌ Could not load image. Please check the file path.")

# 2. Modern Implementation (Recommended)
# For the full-featured implementation, use the new modules:

# from yolo_detector import YOLODetector
# detector = YOLODetector(model_name="yolov8n.pt")
# result = detector.detect_image("test_image.jpg", show_results=True)

# 3. Web Interface
# Run: streamlit run app.py

# 4. Command Line Tools
# Run: python utils.py --mode process --input /path/to/images

# 📦 What This Project Demonstrates:
# ✅ Modern YOLOv8/YOLOv11 implementation
# ✅ Real-time object detection with confidence scores
# ✅ Bounding box visualization
# ✅ Web interface for interactive detection
# ✅ Database integration for analytics
# ✅ Batch processing capabilities
# ✅ Video processing support
# ✅ Comprehensive testing and validation