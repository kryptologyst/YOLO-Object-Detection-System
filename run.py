#!/usr/bin/env python3
"""
YOLO Object Detection System - Startup Script
Quick setup and launch script for the YOLO detection system
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'ultralytics', 'opencv-python', 'matplotlib', 'streamlit',
        'pandas', 'numpy', 'plotly', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True


def setup_project():
    """Initialize project setup"""
    print("🚀 Setting up YOLO Object Detection System...")
    
    # Create necessary directories
    directories = ['data', 'models', 'outputs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Initialize mock database
    try:
        from mock_database import MockDatabase, create_sample_images
        print("🗄️ Initializing database...")
        mock_db = MockDatabase()
        print("🖼️ Creating sample images...")
        create_sample_images()
        print("✅ Database and sample data initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize database: {e}")
    
    print("🎉 Project setup complete!")


def launch_web_ui():
    """Launch the Streamlit web interface"""
    print("🌐 Launching web interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Web interface closed")
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")


def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    try:
        subprocess.run([sys.executable, "test_suite.py"])
    except Exception as e:
        print(f"❌ Error running tests: {e}")


def run_demo():
    """Run a quick demo"""
    print("🎬 Running YOLO detection demo...")
    try:
        from yolo_detector import YOLODetector
        from pathlib import Path
        
        # Check if sample images exist
        sample_images = list(Path("data").glob("*.jpg"))
        if not sample_images:
            print("❌ No sample images found. Run setup first.")
            return
        
        # Initialize detector
        detector = YOLODetector(model_name="yolov8n.pt")
        
        # Process first sample image
        sample_image = sample_images[0]
        print(f"🔍 Processing: {sample_image}")
        
        result = detector.detect_image(sample_image, show_results=True)
        print(f"✅ Detected {result['total_objects']} objects in {result['processing_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO Object Detection System Launcher")
    parser.add_argument("--mode", choices=["setup", "web", "test", "demo", "check"], 
                       default="check", help="Operation mode")
    parser.add_argument("--skip-check", action="store_true", 
                       help="Skip dependency check")
    
    args = parser.parse_args()
    
    print("🎯 YOLO Object Detection System")
    print("=" * 40)
    
    # Check dependencies unless skipped
    if not args.skip_check and not check_dependencies():
        print("\n💡 Run with --skip-check to continue anyway")
        return
    
    # Execute based on mode
    if args.mode == "setup":
        setup_project()
    elif args.mode == "web":
        launch_web_ui()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "check":
        print("\n📋 Available commands:")
        print("  python run.py --mode setup    # Initialize project")
        print("  python run.py --mode web       # Launch web interface")
        print("  python run.py --mode test      # Run test suite")
        print("  python run.py --mode demo     # Run quick demo")
        print("  python run.py --mode check     # Check dependencies")


if __name__ == "__main__":
    main()
