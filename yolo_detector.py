"""
Modern YOLO Object Detection Module
Supports YOLOv8/YOLOv11 with enhanced features
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
from datetime import datetime
import sqlite3
import pandas as pd

from ultralytics import YOLO
from config import MODEL_CONFIG, DETECTION_CONFIG, DATABASE_CONFIG, OUTPUT_DIR, DATABASE_PATH


class YOLODetector:
    """Modern YOLO Object Detection Class with enhanced features"""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize YOLO detector
        
        Args:
            model_name: Model to use (default: from config)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name or MODEL_CONFIG["default_model"]
        self.device = device
        self.model = None
        self.class_names = {}
        self.detection_history = []
        
        # Initialize database
        self._init_database()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading {self.model_name}...")
            self.model = YOLO(self.model_name)
            self.class_names = self.model.names
            print(f"Model loaded successfully! Classes: {len(self.class_names)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _init_database(self):
        """Initialize SQLite database for storing detection results"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DATABASE_CONFIG['table_name']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                model_name TEXT,
                detections TEXT,
                total_objects INTEGER,
                processing_time REAL,
                confidence_threshold REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def detect_image(self, image_path: Union[str, Path], 
                    save_results: bool = True,
                    show_results: bool = False) -> Dict:
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            save_results: Whether to save results to database
            show_results: Whether to display results
            
        Returns:
            Dictionary containing detection results
        """
        start_time = datetime.now()
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.model(image, 
                           conf=MODEL_CONFIG["confidence_threshold"],
                           iou=MODEL_CONFIG["iou_threshold"],
                           max_det=MODEL_CONFIG["max_detections"])
        
        # Process results
        detections = []
        annotated_image = image.copy()
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.class_names[cls]
                    
                    detection = {
                        "class": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "area": (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), 
                                DETECTION_CONFIG["box_color"], DETECTION_CONFIG["line_width"])
                    
                    # Draw label
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, DETECTION_CONFIG["font_size"],
                              DETECTION_CONFIG["text_color"], 2)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare results
        result = {
            "image_path": str(image_path),
            "model_name": self.model_name,
            "detections": detections,
            "total_objects": len(detections),
            "processing_time": processing_time,
            "timestamp": start_time.isoformat(),
            "annotated_image": annotated_image
        }
        
        # Save to database if requested
        if save_results:
            self._save_to_database(result)
        
        # Display results if requested
        if show_results:
            self._display_results(annotated_image, detections)
        
        return result
    
    def detect_video(self, video_path: Union[str, Path], 
                    output_path: Optional[Union[str, Path]] = None,
                    show_preview: bool = False) -> Dict:
        """
        Detect objects in video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            show_preview: Whether to show preview during processing
            
        Returns:
            Dictionary containing video processing results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        start_time = datetime.now()
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on frame
            results = self.model(frame, 
                               conf=MODEL_CONFIG["confidence_threshold"],
                               iou=MODEL_CONFIG["iou_threshold"],
                               max_det=MODEL_CONFIG["max_detections"])
            
            # Annotate frame
            annotated_frame = frame.copy()
            frame_detections = 0
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.class_names[cls]
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    DETECTION_CONFIG["box_color"], DETECTION_CONFIG["line_width"])
                        
                        label_text = f"{label} {conf:.2f}"
                        cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, DETECTION_CONFIG["font_size"],
                                  DETECTION_CONFIG["text_color"], 2)
                        
                        frame_detections += 1
            
            total_detections += frame_detections
            
            # Write frame if output specified
            if writer:
                writer.write(annotated_frame)
            
            # Show preview if requested
            if show_preview:
                cv2.imshow('YOLO Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "video_path": str(video_path),
            "output_path": str(output_path) if output_path else None,
            "total_frames": frame_count,
            "total_detections": total_detections,
            "processing_time": processing_time,
            "fps": fps,
            "average_detections_per_frame": total_detections / frame_count if frame_count > 0 else 0
        }
    
    def batch_detect(self, image_dir: Union[str, Path], 
                    output_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Detect objects in multiple images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save annotated images
            
        Returns:
            List of detection results
        """
        image_dir = Path(image_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.detect_image(image_file, save_results=True, show_results=False)
                
                # Save annotated image if output directory specified
                if output_dir:
                    output_path = output_dir / f"annotated_{image_file.name}"
                    cv2.imwrite(str(output_path), result["annotated_image"])
                    result["output_path"] = str(output_path)
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        return results
    
    def _save_to_database(self, result: Dict):
        """Save detection result to database"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {DATABASE_CONFIG['table_name']} 
            (timestamp, image_path, model_name, detections, total_objects, 
             processing_time, confidence_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["timestamp"],
            result["image_path"],
            result["model_name"],
            json.dumps(result["detections"]),
            result["total_objects"],
            result["processing_time"],
            MODEL_CONFIG["confidence_threshold"]
        ))
        
        conn.commit()
        conn.close()
    
    def _display_results(self, image: np.ndarray, detections: List[Dict]):
        """Display detection results"""
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f"YOLO Detection Results - {len(detections)} objects detected")
        plt.axis('off')
        plt.show()
        
        # Print detection summary
        if detections:
            print(f"\nDetection Summary:")
            print(f"Total objects detected: {len(detections)}")
            
            # Count by class
            class_counts = {}
            for det in detections:
                class_name = det["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("\nObjects by class:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")
    
    def get_detection_history(self, limit: int = 100) -> pd.DataFrame:
        """Get detection history from database"""
        conn = sqlite3.connect(DATABASE_PATH)
        
        query = f"""
            SELECT * FROM {DATABASE_CONFIG['table_name']} 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get total detections
        cursor.execute(f"SELECT COUNT(*) FROM {DATABASE_CONFIG['table_name']}")
        total_detections = cursor.fetchone()[0]
        
        # Get average processing time
        cursor.execute(f"SELECT AVG(processing_time) FROM {DATABASE_CONFIG['table_name']}")
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # Get most detected classes
        cursor.execute(f"SELECT detections FROM {DATABASE_CONFIG['table_name']}")
        all_detections = cursor.fetchall()
        
        class_counts = {}
        for row in all_detections:
            detections = json.loads(row[0])
            for det in detections:
                class_name = det["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        conn.close()
        
        return {
            "total_detections": total_detections,
            "average_processing_time": avg_processing_time,
            "most_detected_classes": dict(sorted(class_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:10])
        }


def main():
    """Example usage of YOLODetector"""
    # Initialize detector
    detector = YOLODetector(model_name="yolov8n.pt")
    
    # Example: Detect objects in an image
    # Replace with your image path
    image_path = "test_image.jpg"
    
    if Path(image_path).exists():
        result = detector.detect_image(image_path, show_results=True)
        print(f"Detection completed in {result['processing_time']:.2f} seconds")
        print(f"Found {result['total_objects']} objects")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")


if __name__ == "__main__":
    main()
