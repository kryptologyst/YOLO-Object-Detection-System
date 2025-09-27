# YOLO Object Detection System

A modern, comprehensive object detection system built with YOLOv8/YOLOv11, featuring a web interface, database integration, and advanced analytics.

## Features

- **Latest YOLO Models**: Support for YOLOv8 and YOLOv11 (nano, small, medium, large, extra-large)
- **Web Interface**: Interactive Streamlit-based UI for real-time detection
- **Database Integration**: SQLite database for storing and analyzing detection results
- **Batch Processing**: Process multiple images and videos efficiently
- **Export Capabilities**: Export results in JSON, CSV, and XML formats
- **Analytics Dashboard**: Comprehensive statistics and performance metrics
- **Mock Database**: Pre-populated sample data for testing and demonstration
- **Comprehensive Testing**: Full test suite with validation and benchmarking

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yolo-object-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the system**
   ```bash
   python mock_database.py
   ```

## Quick Start

### Web Interface
Launch the interactive web application:
```bash
streamlit run app.py
```

### Command Line Usage
```bash
# Process a single image
python yolo_detector.py

# Process a directory of images
python utils.py --mode process --input /path/to/images --output /path/to/output

# Compare different models
python utils.py --mode compare --input test_image.jpg --models yolov8n.pt yolov8s.pt

# Generate analytics report
python utils.py --mode analytics
```

### Python API
```python
from yolo_detector import YOLODetector

# Initialize detector
detector = YOLODetector(model_name="yolov8n.pt")

# Detect objects in image
result = detector.detect_image("path/to/image.jpg", show_results=True)

# Process video
video_result = detector.detect_video("path/to/video.mp4", output_path="output.mp4")

# Batch processing
results = detector.batch_detect("path/to/images/", "path/to/output/")
```

## Project Structure

```
yolo-object-detection/
├── app.py                 # Streamlit web interface
├── yolo_detector.py       # Core YOLO detection class
├── config.py             # Configuration settings
├── mock_database.py      # Database and sample data
├── utils.py              # Utility scripts and CLI
├── test_suite.py         # Comprehensive test suite
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── data/                # Sample images and test data
├── models/              # Downloaded YOLO models
├── outputs/             # Detection results and exports
└── detections.db        # SQLite database
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Selection**: Choose from available YOLO models
- **Detection Parameters**: Confidence and IoU thresholds
- **UI Settings**: Web interface configuration
- **Database Settings**: Storage and cleanup preferences

## Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.pt | Nano | Fastest | Good | Real-time applications |
| yolov8s.pt | Small | Fast | Better | Balanced performance |
| yolov8m.pt | Medium | Moderate | Good | General purpose |
| yolov8l.pt | Large | Slow | Better | High accuracy needed |
| yolov8x.pt | Extra Large | Slowest | Best | Maximum accuracy |
| yolov11n.pt | Latest Nano | Fastest | Good | Latest technology |
| yolov11s.pt | Latest Small | Fast | Better | Latest balanced |

## Testing

Run the comprehensive test suite:
```bash
python test_suite.py
```

The test suite includes:
- Unit tests for all components
- Integration tests
- Performance benchmarks
- Validation tests

## Analytics

The system provides detailed analytics including:
- Detection statistics by class
- Processing performance metrics
- Model comparison results
- Usage patterns and trends

Access analytics through:
- Web interface analytics tab
- Command line: `python utils.py --mode analytics`
- Database queries via `MockDatabase` class

## Video Processing

Process videos with object detection:
```python
detector = YOLODetector()
result = detector.detect_video(
    "input_video.mp4",
    output_path="annotated_video.mp4",
    show_preview=True
)
```

Features:
- Real-time processing
- Preview during processing
- Configurable output formats
- Performance metrics

## 📁 Batch Processing

Process multiple files efficiently:
```python
results = detector.batch_detect(
    input_dir="images/",
    output_dir="results/"
)
```

Supports:
- Multiple image formats
- Progress tracking
- Error handling
- Export in various formats

## Export Formats

Export detection results in multiple formats:

- **JSON**: Complete detection data with metadata
- **CSV**: Tabular format for analysis
- **XML**: Pascal VOC format for annotation tools

## Database

The system uses SQLite for storing detection results:

- Automatic data storage
- Query capabilities
- Analytics and reporting
- Data export options

## Performance

Typical performance on different hardware:

| Hardware | Model | Images/sec | Video FPS |
|----------|-------|------------|-----------|
| CPU (Intel i7) | yolov8n | 15-20 | 8-12 |
| CPU (Intel i7) | yolov8s | 8-12 | 5-8 |
| GPU (RTX 3080) | yolov8n | 60-80 | 30-40 |
| GPU (RTX 3080) | yolov8s | 40-60 | 20-30 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [Streamlit](https://streamlit.io/) for web interface framework
- [OpenCV](https://opencv.org/) for computer vision operations

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Run the test suite for troubleshooting

## Updates

This project is regularly updated with:
- Latest YOLO model versions
- Performance improvements
- New features and capabilities
- Bug fixes and optimizations


# YOLO-Object-Detection-System
