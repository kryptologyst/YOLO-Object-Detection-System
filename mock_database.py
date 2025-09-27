"""
Mock Database Module for YOLO Detection Results
Provides sample data and database utilities
"""

import sqlite3
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional

from config import DATABASE_PATH, DATABASE_CONFIG


class MockDatabase:
    """Mock database for storing and retrieving detection results"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.table_name = DATABASE_CONFIG["table_name"]
        self._init_database()
        self._populate_mock_data()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
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
    
    def _populate_mock_data(self):
        """Populate database with mock detection data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        
        if count > 0:
            conn.close()
            return
        
        # Mock data
        mock_images = [
            "street_scene.jpg", "office_meeting.jpg", "park_family.jpg",
            "traffic_intersection.jpg", "shopping_mall.jpg", "beach_vacation.jpg",
            "kitchen_cooking.jpg", "living_room.jpg", "garden_party.jpg",
            "construction_site.jpg", "airport_terminal.jpg", "restaurant_dining.jpg"
        ]
        
        models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
        
        # COCO class names (first 20 most common)
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow"
        ]
        
        # Generate mock detections
        for i in range(50):  # Generate 50 mock records
            timestamp = datetime.now() - timedelta(days=random.randint(0, 30))
            image_path = random.choice(mock_images)
            model_name = random.choice(models)
            
            # Generate random detections
            num_detections = random.randint(1, 15)
            detections = []
            
            for _ in range(num_detections):
                detection = {
                    "class": random.choice(coco_classes),
                    "confidence": round(random.uniform(0.3, 0.95), 2),
                    "bbox": [
                        random.randint(0, 400),
                        random.randint(0, 300),
                        random.randint(400, 800),
                        random.randint(300, 600)
                    ],
                    "area": random.randint(1000, 50000)
                }
                detections.append(detection)
            
            processing_time = round(random.uniform(0.1, 2.0), 2)
            confidence_threshold = round(random.uniform(0.3, 0.7), 2)
            
            cursor.execute(f"""
                INSERT INTO {self.table_name} 
                (timestamp, image_path, model_name, detections, total_objects, 
                 processing_time, confidence_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp.isoformat(),
                image_path,
                model_name,
                json.dumps(detections),
                num_detections,
                processing_time,
                confidence_threshold
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Populated database with 50 mock detection records")
    
    def get_all_detections(self) -> pd.DataFrame:
        """Get all detection records"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        conn.close()
        return df
    
    def get_detections_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get detections within date range"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT * FROM {self.table_name} 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        return df
    
    def get_detections_by_model(self, model_name: str) -> pd.DataFrame:
        """Get detections by specific model"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT * FROM {self.table_name} 
            WHERE model_name = ?
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=[model_name])
        conn.close()
        return df
    
    def get_class_statistics(self) -> Dict:
        """Get statistics about detected classes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT detections FROM {self.table_name}")
        all_detections = cursor.fetchall()
        
        class_counts = {}
        total_detections = 0
        
        for row in all_detections:
            detections = json.loads(row[0])
            for det in detections:
                class_name = det["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_detections += 1
        
        conn.close()
        
        return {
            "class_counts": class_counts,
            "total_detections": total_detections,
            "unique_classes": len(class_counts)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Average processing time
        cursor.execute(f"SELECT AVG(processing_time) FROM {self.table_name}")
        avg_processing_time = cursor.fetchone()[0] or 0
        
        # Total detections
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        total_records = cursor.fetchone()[0]
        
        # Average objects per detection
        cursor.execute(f"SELECT AVG(total_objects) FROM {self.table_name}")
        avg_objects = cursor.fetchone()[0] or 0
        
        # Model usage statistics
        cursor.execute(f"""
            SELECT model_name, COUNT(*) as usage_count 
            FROM {self.table_name} 
            GROUP BY model_name
        """)
        model_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_records": total_records,
            "avg_processing_time": avg_processing_time,
            "avg_objects_per_detection": avg_objects,
            "model_usage": model_stats
        }
    
    def add_detection(self, image_path: str, model_name: str, 
                     detections: List[Dict], processing_time: float,
                     confidence_threshold: float = 0.5):
        """Add a new detection record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            INSERT INTO {self.table_name} 
            (timestamp, image_path, model_name, detections, total_objects, 
             processing_time, confidence_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            image_path,
            model_name,
            json.dumps(detections),
            len(detections),
            processing_time,
            confidence_threshold
        ))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """Remove records older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cursor.execute(f"""
            DELETE FROM {self.table_name} 
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"🗑️ Cleaned up {deleted_count} old records")
        return deleted_count
    
    def export_to_csv(self, output_path: str):
        """Export all data to CSV"""
        df = self.get_all_detections()
        df.to_csv(output_path, index=False)
        print(f"📊 Exported {len(df)} records to {output_path}")
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive data for dashboard"""
        stats = self.get_performance_stats()
        class_stats = self.get_class_statistics()
        
        # Get recent activity (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_df = self.get_detections_by_date_range(week_ago, datetime.now().isoformat())
        
        # Daily activity
        if not recent_df.empty:
            recent_df['date'] = pd.to_datetime(recent_df['timestamp']).dt.date
            daily_activity = recent_df.groupby('date').agg({
                'total_objects': 'sum',
                'processing_time': 'mean',
                'id': 'count'
            }).reset_index()
            daily_activity.columns = ['date', 'total_objects', 'avg_processing_time', 'detection_count']
        else:
            daily_activity = pd.DataFrame()
        
        return {
            "performance_stats": stats,
            "class_statistics": class_stats,
            "daily_activity": daily_activity.to_dict('records') if not daily_activity.empty else [],
            "recent_detections": recent_df.head(10).to_dict('records')
        }


def create_sample_images():
    """Create sample images for testing"""
    import cv2
    import numpy as np
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample image 1: Street scene
    street_img = np.zeros((480, 640, 3), dtype=np.uint8)
    street_img[:] = (135, 206, 235)  # Sky blue background
    
    # Add some rectangles to simulate objects
    cv2.rectangle(street_img, (100, 300), (200, 450), (0, 255, 0), -1)  # Person
    cv2.rectangle(street_img, (300, 350), (500, 420), (255, 0, 0), -1)  # Car
    cv2.rectangle(street_img, (50, 200), (150, 280), (0, 0, 255), -1)  # Building
    
    cv2.imwrite(str(data_dir / "street_scene.jpg"), street_img)
    
    # Sample image 2: Office scene
    office_img = np.zeros((480, 640, 3), dtype=np.uint8)
    office_img[:] = (255, 255, 255)  # White background
    
    # Add office objects
    cv2.rectangle(office_img, (200, 200), (300, 400), (128, 128, 128), -1)  # Person
    cv2.rectangle(office_img, (400, 300), (600, 350), (139, 69, 19), -1)  # Desk
    cv2.rectangle(office_img, (100, 100), (200, 200), (0, 0, 0), -1)  # Computer
    
    cv2.imwrite(str(data_dir / "office_meeting.jpg"), office_img)
    
    # Sample image 3: Park scene
    park_img = np.zeros((480, 640, 3), dtype=np.uint8)
    park_img[:] = (34, 139, 34)  # Forest green background
    
    # Add park objects
    cv2.rectangle(park_img, (150, 250), (250, 400), (0, 255, 0), -1)  # Person
    cv2.rectangle(park_img, (300, 200), (400, 300), (139, 69, 19), -1)  # Tree
    cv2.rectangle(park_img, (500, 350), (600, 400), (255, 255, 0), -1)  # Bench
    
    cv2.imwrite(str(data_dir / "park_family.jpg"), park_img)
    
    print("✅ Created sample test images in data/ directory")


if __name__ == "__main__":
    # Initialize mock database
    mock_db = MockDatabase()
    
    # Create sample images
    create_sample_images()
    
    # Display some statistics
    print("\n📊 Database Statistics:")
    stats = mock_db.get_performance_stats()
    print(f"Total records: {stats['total_records']}")
    print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
    print(f"Average objects per detection: {stats['avg_objects_per_detection']:.1f}")
    
    class_stats = mock_db.get_class_statistics()
    print(f"\n🏆 Top 5 detected classes:")
    top_classes = sorted(class_stats['class_counts'].items(), 
                        key=lambda x: x[1], reverse=True)[:5]
    for class_name, count in top_classes:
        print(f"  {class_name}: {count}")
    
    print("\n✅ Mock database setup complete!")
