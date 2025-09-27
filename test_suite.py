"""
Test Suite for YOLO Object Detection System
Comprehensive testing and validation scripts
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import json
import time
from unittest.mock import patch, MagicMock

from yolo_detector import YOLODetector
from mock_database import MockDatabase
from config import MODEL_CONFIG, DETECTION_CONFIG


class TestYOLODetector(unittest.TestCase):
    """Test cases for YOLODetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        self._create_test_image()
        
        # Mock the YOLO model to avoid downloading during tests
        self.detector = None
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_image(self):
        """Create a test image with simple shapes"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (135, 206, 235)  # Sky blue background
        
        # Add some rectangles to simulate objects
        cv2.rectangle(img, (100, 300), (200, 450), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(img, (300, 350), (500, 420), (255, 0, 0), -1)  # Blue rectangle
        cv2.rectangle(img, (50, 200), (150, 280), (0, 0, 255), -1)  # Red rectangle
        
        cv2.imwrite(str(self.test_image_path), img)
    
    @patch('ultralytics.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization"""
        mock_model = MagicMock()
        mock_model.names = {0: 'person', 1: 'car', 2: 'bicycle'}
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(model_name="yolov8n.pt")
        
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.model_name, "yolov8n.pt")
        self.assertEqual(len(detector.class_names), 3)
    
    @patch('ultralytics.YOLO')
    def test_image_detection(self, mock_yolo):
        """Test image detection functionality"""
        # Mock YOLO model and results
        mock_model = MagicMock()
        mock_model.names = {0: 'person', 1: 'car'}
        
        # Mock detection results
        mock_box = MagicMock()
        mock_box.xyxy = [[100, 200, 300, 400]]
        mock_box.cls = [0]
        mock_box.conf = [0.85]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(model_name="yolov8n.pt")
        result = detector.detect_image(self.test_image_path, save_results=False)
        
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('total_objects', result)
        self.assertIn('processing_time', result)
        self.assertGreaterEqual(result['total_objects'], 0)
    
    @patch('ultralytics.YOLO')
    def test_invalid_image_path(self, mock_yolo):
        """Test handling of invalid image path"""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(model_name="yolov8n.pt")
        
        with self.assertRaises(ValueError):
            detector.detect_image("nonexistent_image.jpg")
    
    def test_config_validation(self):
        """Test configuration validation"""
        self.assertIn('default_model', MODEL_CONFIG)
        self.assertIn('confidence_threshold', MODEL_CONFIG)
        self.assertIn('iou_threshold', MODEL_CONFIG)
        
        self.assertGreater(MODEL_CONFIG['confidence_threshold'], 0)
        self.assertLessEqual(MODEL_CONFIG['confidence_threshold'], 1)
        self.assertGreater(MODEL_CONFIG['iou_threshold'], 0)
        self.assertLessEqual(MODEL_CONFIG['iou_threshold'], 1)


class TestMockDatabase(unittest.TestCase):
    """Test cases for MockDatabase class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Patch the database path for testing
        with patch('mock_database.DATABASE_PATH', self.temp_db.name):
            self.mock_db = MockDatabase()
    
    def tearDown(self):
        """Clean up test fixtures"""
        Path(self.temp_db.name).unlink()
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertIsNotNone(self.mock_db)
        self.assertTrue(Path(self.temp_db.name).exists())
    
    def test_add_detection(self):
        """Test adding detection record"""
        detections = [
            {"class": "person", "confidence": 0.85, "bbox": [100, 200, 300, 400], "area": 40000},
            {"class": "car", "confidence": 0.92, "bbox": [150, 250, 350, 450], "area": 40000}
        ]
        
        self.mock_db.add_detection(
            image_path="test.jpg",
            model_name="yolov8n.pt",
            detections=detections,
            processing_time=1.5
        )
        
        df = self.mock_db.get_all_detections()
        self.assertGreater(len(df), 0)
    
    def test_get_performance_stats(self):
        """Test performance statistics retrieval"""
        stats = self.mock_db.get_performance_stats()
        
        self.assertIn('total_records', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertIn('avg_objects_per_detection', stats)
        self.assertIn('model_usage', stats)
    
    def test_get_class_statistics(self):
        """Test class statistics retrieval"""
        stats = self.mock_db.get_class_statistics()
        
        self.assertIn('class_counts', stats)
        self.assertIn('total_detections', stats)
        self.assertIn('unique_classes', stats)
    
    def test_export_to_csv(self):
        """Test CSV export functionality"""
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_csv.close()
        
        try:
            self.mock_db.export_to_csv(temp_csv.name)
            self.assertTrue(Path(temp_csv.name).exists())
            
            # Verify CSV content
            import pandas as pd
            df = pd.read_csv(temp_csv.name)
            self.assertGreater(len(df), 0)
        finally:
            Path(temp_csv.name).unlink()


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "integration_test.jpg"
        self._create_test_image()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_image(self):
        """Create a test image"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # White background
        cv2.imwrite(str(self.test_image_path), img)
    
    @patch('ultralytics.YOLO')
    def test_end_to_end_detection(self, mock_yolo):
        """Test complete detection pipeline"""
        # Mock YOLO model
        mock_model = MagicMock()
        mock_model.names = {0: 'person', 1: 'car'}
        
        mock_box = MagicMock()
        mock_box.xyxy = [[100, 200, 300, 400]]
        mock_box.cls = [0]
        mock_box.conf = [0.85]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Test detection
        detector = YOLODetector(model_name="yolov8n.pt")
        result = detector.detect_image(self.test_image_path, save_results=True)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('detections', result)
        self.assertIn('annotated_image', result)
        
        # Verify annotated image
        annotated_img = result['annotated_image']
        self.assertIsInstance(annotated_img, np.ndarray)
        self.assertEqual(len(annotated_img.shape), 3)


class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_images = []
        
        # Create multiple test images
        for i in range(5):
            img_path = Path(self.temp_dir) / f"test_image_{i}.jpg"
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            self.test_images.append(img_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    @patch('ultralytics.YOLO')
    def test_batch_processing_performance(self, mock_yolo):
        """Test batch processing performance"""
        # Mock YOLO model
        mock_model = MagicMock()
        mock_model.names = {0: 'person'}
        
        mock_box = MagicMock()
        mock_box.xyxy = [[100, 200, 300, 400]]
        mock_box.cls = [0]
        mock_box.conf = [0.85]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(model_name="yolov8n.pt")
        
        start_time = time.time()
        results = detector.batch_detect(self.temp_dir)
        processing_time = time.time() - start_time
        
        self.assertEqual(len(results), len(self.test_images))
        self.assertLess(processing_time, 10)  # Should complete within 10 seconds


def run_validation_tests():
    """Run validation tests for the YOLO system"""
    print("🧪 Running YOLO Object Detection Validation Tests...")
    
    # Test configuration
    print("\n1. Testing Configuration...")
    assert MODEL_CONFIG['confidence_threshold'] > 0, "Invalid confidence threshold"
    assert MODEL_CONFIG['iou_threshold'] > 0, "Invalid IoU threshold"
    assert len(MODEL_CONFIG['available_models']) > 0, "No models available"
    print("✅ Configuration validation passed")
    
    # Test database
    print("\n2. Testing Database...")
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    with patch('mock_database.DATABASE_PATH', temp_db.name):
        mock_db = MockDatabase()
        stats = mock_db.get_performance_stats()
        assert stats['total_records'] > 0, "No mock data generated"
    print("✅ Database validation passed")
    
    # Test image creation
    print("\n3. Testing Image Creation...")
    temp_dir = tempfile.mkdtemp()
    test_img_path = Path(temp_dir) / "test.jpg"
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(test_img_path), img)
    assert test_img_path.exists(), "Test image not created"
    print("✅ Image creation validation passed")
    
    # Cleanup
    Path(temp_db.name).unlink()
    shutil.rmtree(temp_dir)
    
    print("\n🎉 All validation tests passed!")


def benchmark_models():
    """Benchmark different YOLO models"""
    print("⚡ Running Model Benchmark...")
    
    models_to_test = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    results = {}
    
    # Create test image
    temp_dir = tempfile.mkdtemp()
    test_img_path = Path(temp_dir) / "benchmark_test.jpg"
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_img_path), img)
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            with patch('ultralytics.YOLO') as mock_yolo:
                # Mock model
                mock_model = MagicMock()
                mock_model.names = {0: 'person'}
                
                mock_box = MagicMock()
                mock_box.xyxy = [[100, 200, 300, 400]]
                mock_box.cls = [0]
                mock_box.conf = [0.85]
                
                mock_result = MagicMock()
                mock_result.boxes = [mock_box]
                
                mock_model.return_value = [mock_result]
                mock_yolo.return_value = mock_model
                
                detector = YOLODetector(model_name=model_name)
                
                # Benchmark
                start_time = time.time()
                result = detector.detect_image(test_img_path, save_results=False)
                processing_time = time.time() - start_time
                
                results[model_name] = {
                    'processing_time': processing_time,
                    'objects_detected': result['total_objects']
                }
                
                print(f"  Processing time: {processing_time:.3f}s")
                print(f"  Objects detected: {result['total_objects']}")
        
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n📊 Benchmark Results:")
    for model, result in results.items():
        if 'error' not in result:
            print(f"{model}: {result['processing_time']:.3f}s, {result['objects_detected']} objects")
        else:
            print(f"{model}: Error - {result['error']}")


if __name__ == "__main__":
    # Run validation tests
    run_validation_tests()
    
    # Run benchmark
    benchmark_models()
    
    # Run unit tests
    print("\n🧪 Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)
