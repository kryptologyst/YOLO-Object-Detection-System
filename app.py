"""
Modern Streamlit Web UI for YOLO Object Detection
Interactive interface with real-time detection capabilities
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import json
from datetime import datetime
import time

from yolo_detector import YOLODetector
from config import MODEL_CONFIG, UI_CONFIG, DATABASE_CONFIG


# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["title"],
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-result {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []


def initialize_detector(model_name: str, device: str):
    """Initialize YOLO detector with progress bar"""
    with st.spinner(f"Loading {model_name}..."):
        try:
            detector = YOLODetector(model_name=model_name, device=device)
            st.session_state.detector = detector
            st.success(f"✅ Model {model_name} loaded successfully!")
            return detector
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown(f'<h1 class="main-header">{UI_CONFIG["title"]}</h1>', 
                unsafe_allow_html=True)
    st.markdown(f"**{UI_CONFIG['description']}**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Select YOLO Model",
            MODEL_CONFIG["available_models"],
            index=0,
            help="Choose model size: n=nano (fastest), s=small, m=medium, l=large, x=extra large (most accurate)"
        )
        
        # Device selection
        device = st.selectbox(
            "Select Device",
            ["cpu", "cuda", "mps"],
            index=0,
            help="Choose processing device. CUDA for NVIDIA GPUs, MPS for Apple Silicon"
        )
        
        # Detection parameters
        st.subheader("🎯 Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=MODEL_CONFIG["confidence_threshold"],
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=MODEL_CONFIG["iou_threshold"],
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        # Initialize detector button
        if st.button("🚀 Initialize Detector", type="primary"):
            detector = initialize_detector(model_name, device)
            if detector:
                # Update model config
                MODEL_CONFIG["confidence_threshold"] = confidence_threshold
                MODEL_CONFIG["iou_threshold"] = iou_threshold
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📊 Analytics", "📁 Batch Processing"])
    
    with tab1:
        st.header("📸 Single Image Detection")
        
        if st.session_state.detector is None:
            st.warning("⚠️ Please initialize the detector in the sidebar first.")
        else:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload an image for object detection"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📤 Original Image")
                    # Display original image
                    image = cv2.imread(tmp_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Original Image", use_column_width=True)
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    
                    if st.button("🔍 Detect Objects", type="primary"):
                        with st.spinner("Running detection..."):
                            start_time = time.time()
                            result = st.session_state.detector.detect_image(
                                tmp_path, 
                                save_results=True, 
                                show_results=False
                            )
                            processing_time = time.time() - start_time
                        
                        # Display annotated image
                        annotated_image = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                        st.image(annotated_image, caption="Detection Results", use_column_width=True)
                        
                        # Display detection summary
                        st.markdown('<div class="detection-result">', unsafe_allow_html=True)
                        st.write(f"**🎯 Objects Detected:** {result['total_objects']}")
                        st.write(f"**⏱️ Processing Time:** {processing_time:.2f} seconds")
                        st.write(f"**🤖 Model:** {result['model_name']}")
                        
                        # Show individual detections
                        if result['detections']:
                            st.subheader("📋 Detection Details")
                            detections_df = pd.DataFrame(result['detections'])
                            st.dataframe(detections_df, use_container_width=True)
                            
                            # Detection statistics
                            class_counts = detections_df['class'].value_counts()
                            if len(class_counts) > 0:
                                fig = px.pie(
                                    values=class_counts.values, 
                                    names=class_counts.index,
                                    title="Objects by Class"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to session history
                        st.session_state.detection_history.append(result)
                
                # Clean up temporary file
                Path(tmp_path).unlink()
    
    with tab2:
        st.header("🎥 Video Detection")
        
        if st.session_state.detector is None:
            st.warning("⚠️ Please initialize the detector in the sidebar first.")
        else:
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video for object detection"
            )
            
            if uploaded_video is not None:
                st.subheader("📤 Video Preview")
                
                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_video.getvalue())
                    tmp_video_path = tmp_file.name
                
                # Display video
                st.video(uploaded_video)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("⚙️ Processing Options")
                    show_preview = st.checkbox("Show preview during processing", value=False)
                    save_output = st.checkbox("Save annotated video", value=True)
                
                with col2:
                    st.subheader("🚀 Process Video")
                    if st.button("🎬 Process Video", type="primary"):
                        with st.spinner("Processing video..."):
                            output_path = None
                            if save_output:
                                output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            
                            result = st.session_state.detector.detect_video(
                                tmp_video_path,
                                output_path=output_path,
                                show_preview=show_preview
                            )
                        
                        # Display results
                        st.success("✅ Video processing completed!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Frames", result['total_frames'])
                        with col_b:
                            st.metric("Total Detections", result['total_detections'])
                        with col_c:
                            st.metric("Processing Time", f"{result['processing_time']:.1f}s")
                        
                        st.metric("Average Detections/Frame", f"{result['average_detections_per_frame']:.1f}")
                        
                        if save_output and Path(output_path).exists():
                            st.subheader("📥 Download Processed Video")
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Annotated Video",
                                    data=file.read(),
                                    file_name=output_path,
                                    mime="video/mp4"
                                )
                
                # Clean up temporary file
                Path(tmp_video_path).unlink()
    
    with tab3:
        st.header("📊 Detection Analytics")
        
        if st.session_state.detector is None:
            st.warning("⚠️ Please initialize the detector in the sidebar first.")
        else:
            # Get statistics
            stats = st.session_state.detector.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Detections", stats['total_detections'])
            
            with col2:
                st.metric("Avg Processing Time", f"{stats['average_processing_time']:.2f}s")
            
            with col3:
                st.metric("Unique Classes", len(stats['most_detected_classes']))
            
            # Most detected classes chart
            if stats['most_detected_classes']:
                st.subheader("🏆 Most Detected Classes")
                
                classes_df = pd.DataFrame(
                    list(stats['most_detected_classes'].items()),
                    columns=['Class', 'Count']
                )
                
                fig = px.bar(
                    classes_df, 
                    x='Count', 
                    y='Class',
                    orientation='h',
                    title="Detection Count by Class"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detection history
            st.subheader("📈 Recent Detection History")
            history_df = st.session_state.detector.get_detection_history(limit=50)
            
            if not history_df.empty:
                # Convert timestamp to datetime
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                
                # Processing time over time
                fig = px.line(
                    history_df, 
                    x='timestamp', 
                    y='processing_time',
                    title="Processing Time Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Objects detected over time
                fig = px.line(
                    history_df, 
                    x='timestamp', 
                    y='total_objects',
                    title="Objects Detected Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display recent detections table
                st.subheader("📋 Recent Detections")
                display_df = history_df[['timestamp', 'image_path', 'total_objects', 'processing_time']].copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No detection history available yet.")
    
    with tab4:
        st.header("📁 Batch Processing")
        
        if st.session_state.detector is None:
            st.warning("⚠️ Please initialize the detector in the sidebar first.")
        else:
            st.subheader("📂 Upload Multiple Images")
            
            uploaded_files = st.file_uploader(
                "Choose multiple image files",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if uploaded_files:
                st.write(f"📊 {len(uploaded_files)} files selected for processing")
                
                if st.button("🚀 Process All Images", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    temp_files = []
                    
                    try:
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {uploaded_file.name}...")
                            
                            # Save file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                                temp_files.append(tmp_path)
                            
                            # Process image
                            result = st.session_state.detector.detect_image(
                                tmp_path, 
                                save_results=True, 
                                show_results=False
                            )
                            results.append(result)
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.text("✅ Batch processing completed!")
                        
                        # Display batch results
                        st.subheader("📊 Batch Processing Results")
                        
                        batch_df = pd.DataFrame([
                            {
                                'File': Path(r['image_path']).name,
                                'Objects': r['total_objects'],
                                'Processing Time': f"{r['processing_time']:.2f}s"
                            }
                            for r in results
                        ])
                        
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Images", len(results))
                        with col2:
                            st.metric("Total Objects", sum(r['total_objects'] for r in results))
                        with col3:
                            st.metric("Avg Processing Time", f"{np.mean([r['processing_time'] for r in results]):.2f}s")
                        
                    finally:
                        # Clean up temporary files
                        for tmp_file in temp_files:
                            Path(tmp_file).unlink()


if __name__ == "__main__":
    main()
