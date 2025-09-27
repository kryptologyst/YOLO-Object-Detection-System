"""
Utility Scripts for YOLO Object Detection System
Additional tools and scripts for enhanced functionality
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from yolo_detector import YOLODetector
from mock_database import MockDatabase
from config import MODEL_CONFIG, OUTPUT_DIR


class YOLOUtils:
    """Utility class for YOLO operations"""
    
    def __init__(self):
        self.detector = None
        self.mock_db = MockDatabase()
    
    def initialize_detector(self, model_name: str = None, device: str = "cpu"):
        """Initialize YOLO detector"""
        self.detector = YOLODetector(model_name=model_name, device=device)
        print(f"✅ Detector initialized with {model_name or MODEL_CONFIG['default_model']}")
    
    def process_directory(self, input_dir: str, output_dir: str = None, 
                         export_format: str = "json") -> Dict:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path (optional)
            export_format: Export format ('json', 'csv', 'xml')
        
        Returns:
            Processing results summary
        """
        if not self.detector:
            raise ValueError("Detector not initialized. Call initialize_detector() first.")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = OUTPUT_DIR
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"📁 Found {len(image_files)} images to process")
        
        results = []
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Detect objects
                result = self.detector.detect_image(image_file, save_results=True)
                
                # Save annotated image
                annotated_path = output_path / f"annotated_{image_file.name}"
                cv2.imwrite(str(annotated_path), result["annotated_image"])
                
                # Export detection data
                if export_format == "json":
                    json_path = output_path / f"{image_file.stem}_detections.json"
                    self._export_json(result, json_path)
                elif export_format == "csv":
                    csv_path = output_path / f"{image_file.stem}_detections.csv"
                    self._export_csv(result, csv_path)
                elif export_format == "xml":
                    xml_path = output_path / f"{image_file.stem}_detections.xml"
                    self._export_xml(result, xml_path)
                
                results.append({
                    "file": image_file.name,
                    "objects": result["total_objects"],
                    "processing_time": result["processing_time"],
                    "output_path": str(annotated_path)
                })
                
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
                continue
        
        # Generate summary report
        summary = self._generate_summary_report(results, output_path)
        return summary
    
    def _export_json(self, result: Dict, output_path: Path):
        """Export detection results to JSON"""
        export_data = {
            "image_path": result["image_path"],
            "model_name": result["model_name"],
            "timestamp": result["timestamp"],
            "processing_time": result["processing_time"],
            "total_objects": result["total_objects"],
            "detections": result["detections"]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, result: Dict, output_path: Path):
        """Export detection results to CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'area'])
            
            for det in result["detections"]:
                bbox = det["bbox"]
                writer.writerow([
                    det["class"],
                    det["confidence"],
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    det["area"]
                ])
    
    def _export_xml(self, result: Dict, output_path: Path):
        """Export detection results to XML (Pascal VOC format)"""
        import xml.etree.ElementTree as ET
        
        root = ET.Element("annotation")
        
        # Add image info
        ET.SubElement(root, "filename").text = Path(result["image_path"]).name
        ET.SubElement(root, "model").text = result["model_name"]
        ET.SubElement(root, "timestamp").text = result["timestamp"]
        
        # Add detections
        objects_elem = ET.SubElement(root, "objects")
        for det in result["detections"]:
            obj_elem = ET.SubElement(objects_elem, "object")
            ET.SubElement(obj_elem, "name").text = det["class"]
            ET.SubElement(obj_elem, "confidence").text = str(det["confidence"])
            
            bbox_elem = ET.SubElement(obj_elem, "bndbox")
            bbox = det["bbox"]
            ET.SubElement(bbox_elem, "xmin").text = str(bbox[0])
            ET.SubElement(bbox_elem, "ymin").text = str(bbox[1])
            ET.SubElement(bbox_elem, "xmax").text = str(bbox[2])
            ET.SubElement(bbox_elem, "ymax").text = str(bbox[3])
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _generate_summary_report(self, results: List[Dict], output_path: Path) -> Dict:
        """Generate processing summary report"""
        if not results:
            return {"error": "No results to summarize"}
        
        total_images = len(results)
        total_objects = sum(r["objects"] for r in results)
        total_time = sum(r["processing_time"] for r in results)
        avg_time = total_time / total_images if total_images > 0 else 0
        
        summary = {
            "total_images_processed": total_images,
            "total_objects_detected": total_objects,
            "total_processing_time": total_time,
            "average_processing_time": avg_time,
            "average_objects_per_image": total_objects / total_images if total_images > 0 else 0,
            "output_directory": str(output_path),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary report
        summary_path = output_path / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Processing Summary:")
        print(f"  Images processed: {total_images}")
        print(f"  Objects detected: {total_objects}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per image: {avg_time:.2f}s")
        print(f"  Summary saved to: {summary_path}")
        
        return summary
    
    def compare_models(self, image_path: str, models: List[str]) -> Dict:
        """
        Compare detection results across different models
        
        Args:
            image_path: Path to test image
            models: List of model names to compare
        
        Returns:
            Comparison results
        """
        if not Path(image_path).exists():
            raise ValueError(f"Image not found: {image_path}")
        
        results = {}
        
        for model_name in models:
            print(f"Testing {model_name}...")
            
            try:
                detector = YOLODetector(model_name=model_name)
                result = detector.detect_image(image_path, save_results=False)
                
                results[model_name] = {
                    "total_objects": result["total_objects"],
                    "processing_time": result["processing_time"],
                    "detections": result["detections"]
                }
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def generate_analytics_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive analytics report"""
        if not output_path:
            output_path = OUTPUT_DIR / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get database statistics
        stats = self.mock_db.get_performance_stats()
        class_stats = self.mock_db.get_class_statistics()
        
        # Get recent activity
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_df = self.mock_db.get_detections_by_date_range(week_ago, datetime.now().isoformat())
        
        # Generate report
        report = {
            "generated_at": datetime.now().isoformat(),
            "performance_statistics": stats,
            "class_statistics": class_stats,
            "recent_activity": {
                "total_detections_last_week": len(recent_df),
                "average_objects_per_detection": recent_df['total_objects'].mean() if not recent_df.empty else 0,
                "average_processing_time": recent_df['processing_time'].mean() if not recent_df.empty else 0
            },
            "recommendations": self._generate_recommendations(stats, class_stats)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📈 Analytics report saved to: {output_path}")
        return report
    
    def _generate_recommendations(self, stats: Dict, class_stats: Dict) -> List[str]:
        """Generate recommendations based on statistics"""
        recommendations = []
        
        if stats['avg_processing_time'] > 2.0:
            recommendations.append("Consider using a smaller model (yolov8n) for faster processing")
        
        if stats['avg_objects_per_detection'] > 10:
            recommendations.append("High object density detected - consider adjusting confidence threshold")
        
        if len(class_stats['class_counts']) < 5:
            recommendations.append("Limited class diversity - consider testing with more varied images")
        
        if not recommendations:
            recommendations.append("System performance is optimal - no specific recommendations")
        
        return recommendations


def main():
    """Command-line interface for YOLO utilities"""
    parser = argparse.ArgumentParser(description="YOLO Object Detection Utilities")
    parser.add_argument("--mode", choices=["process", "compare", "analytics", "test"], 
                       required=True, help="Operation mode")
    parser.add_argument("--input", help="Input file or directory path")
    parser.add_argument("--output", help="Output directory path")
    parser.add_argument("--model", help="YOLO model name")
    parser.add_argument("--models", nargs="+", help="Multiple models for comparison")
    parser.add_argument("--format", choices=["json", "csv", "xml"], default="json",
                       help="Export format")
    parser.add_argument("--device", default="cpu", help="Processing device")
    
    args = parser.parse_args()
    
    utils = YOLOUtils()
    
    if args.mode == "process":
        if not args.input:
            print("❌ Input directory required for process mode")
            return
        
        utils.initialize_detector(args.model, args.device)
        result = utils.process_directory(args.input, args.output, args.format)
        print(f"✅ Processing completed: {result}")
    
    elif args.mode == "compare":
        if not args.input or not args.models:
            print("❌ Input image and models list required for compare mode")
            return
        
        result = utils.compare_models(args.input, args.models)
        print(f"📊 Model comparison results: {result}")
    
    elif args.mode == "analytics":
        report = utils.generate_analytics_report(args.output)
        print(f"📈 Analytics report generated")
    
    elif args.mode == "test":
        print("🧪 Running system tests...")
        from test_suite import run_validation_tests
        run_validation_tests()


if __name__ == "__main__":
    main()
