"""
Enhanced Streamlit Dashboard for Real-time CV Queue Management Visualization

This dashboard provides comprehensive real-time visualization capabilities including
video streaming, interactive zone configuration, and live event monitoring.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time
import cv2
import threading
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from io import BytesIO
import base64

# Import platform components
from premise_cv_platform.config.settings import settings
from premise_cv_platform.config.zone_config import ZoneManager, Zone, ZoneType, create_default_zone_manager
from premise_cv_platform.storage.csv_manager import CSVManager
from premise_cv_platform.storage.data_schemas import ProcessingSummary, Detection
from premise_cv_platform.utils.async_processor import processing_manager, ProcessingStatus
from premise_cv_platform.utils.file_handler import TempFileManager
from premise_cv_platform.utils.logging_config import get_zone_logger

# Import visualization components from main directory
from visualize_IDEAL import UpdatedZoneEventDetector, VisualUpdatedDetector
from premise_cv_platform.inference.track_people import PersonTracker

# Import interface components
try:
    from .video_processor import StreamlitVideoProcessor
    from .controls import InteractiveZoneControls
    from .event_visualizer import RealTimeEventVisualizer
except ImportError:
    # Handle direct execution
    from premise_cv_platform.interface.video_processor import StreamlitVideoProcessor
    from premise_cv_platform.interface.controls import InteractiveZoneControls
    from premise_cv_platform.interface.event_visualizer import RealTimeEventVisualizer

# Performance monitoring imports
import psutil
import gc
import resource
from collections import defaultdict


class EnhancedPremiseDashboard:
    """Enhanced Streamlit dashboard with real-time video streaming capabilities."""

    def __init__(self):
        self.csv_manager = CSVManager()
        self.setup_page_config()
        
        # Initialize real-time components
        self.zone_logger = get_zone_logger()
        self.tracker = PersonTracker()
        self.detector = UpdatedZoneEventDetector()
        
        # Real-time video streaming state
        if "video_stream_active" not in st.session_state:
            st.session_state.video_stream_active = False
            st.session_state.current_frame = None
            st.session_state.frame_queue = deque(maxlen=10)
            st.session_state.live_events = []
            st.session_state.performance_metrics = {
                'fps': 0,
                'processing_time': 0,
                'memory_usage': 0,
                'frame_count': 0,
                'cpu_usage': 0,
                'gpu_usage': 0,
                'disk_usage': 0,
                'network_usage': 0
            }
        
        # Performance monitoring state
        if "performance_history" not in st.session_state:
            st.session_state.performance_history = {
                'timestamps': deque(maxlen=100),
                'fps_history': deque(maxlen=100),
                'memory_history': deque(maxlen=100),
                'cpu_history': deque(maxlen=100),
                'processing_time_history': deque(maxlen=100)
            }
        
        # System health monitoring
        if "system_health" not in st.session_state:
            st.session_state.system_health = {
                'status': 'healthy',
                'alerts': [],
                'uptime_start': datetime.now(),
                'total_processed_frames': 0,
                'total_processing_time': 0,
                'error_count': 0,
                'last_health_check': datetime.now()
            }
        
        # Initialize component instances
        self.video_processor = StreamlitVideoProcessor()
        self.zone_controls = InteractiveZoneControls()
        self.event_visualizer = RealTimeEventVisualizer()

        # Initialize session state for upload and processing
        if "upload_processor" not in st.session_state:
            session_id = st.session_state.get("session_id", f"session_{int(time.time())}")
            st.session_state.session_id = session_id
            st.session_state.upload_processor = processing_manager.get_processor(session_id)
            st.session_state.processing_status = ProcessingStatus(
                status="idle", message="Ready for upload"
            )
            st.session_state.uploaded_file_info = None

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="PREMISE CV - Real-time Dashboard",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render_navigation(self):
        """Render main navigation tabs."""
        return st.tabs([
            "🎬 Live Video Stream", 
            "📤 Video Upload", 
            "📊 Analytics", 
            "🗺️ Zone Config",
            "📋 Event Monitor",
            "📈 Performance Monitor",
            "⚙️ System Status"
        ])

    def render_live_video_tab(self):
        """Render real-time video streaming interface."""
        st.header("🎬 Real-time Video Stream & Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Video stream controls
            control_col1, control_col2, control_col3 = st.columns(3)
            
            with control_col1:
                if st.button("🎥 Start Live Stream", type="primary", disabled=st.session_state.video_stream_active):
                    self.start_live_stream()
            
            with control_col2:
                if st.button("⏹️ Stop Stream", disabled=not st.session_state.video_stream_active):
                    self.stop_live_stream()
            
            with control_col3:
                video_source = st.selectbox(
                    "Video Source",
                    ["Test Video (bank_sample.MOV)", "Webcam", "Uploaded File"],
                    help="Select video source for live streaming"
                )
            
            # Video display area
            video_placeholder = st.empty()
            
            if st.session_state.video_stream_active and st.session_state.current_frame is not None:
                # Display current frame with annotations
                video_placeholder.image(
                    st.session_state.current_frame,
                    caption="Live Video Analysis with Zone Detection",
                    use_column_width=True
                )
            else:
                video_placeholder.info("🎬 Click 'Start Live Stream' to begin real-time video analysis")
        
        with col2:
            # Real-time performance metrics
            self.render_live_performance_metrics()
            
            # Live event stream
            self.render_live_event_stream()

    def start_live_stream(self):
        """Start real-time video streaming."""
        st.session_state.video_stream_active = True
        
        # Initialize video processing thread
        video_path = "videos/bank_sample.MOV"  # Default test video
        
        def video_processing_thread():
            """Background thread for video processing."""
            try:
                cap = cv2.VideoCapture(video_path)
                frame_number = 0
                
                while st.session_state.video_stream_active and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                        continue
                    
                    start_time = time.time()
                    
                    # Run detection and tracking
                    detections = self.tracker.detect_and_track(frame, frame_number, datetime.now())
                    
                    # Process zone events
                    zone_events = self.detector.detect_zone_events(detections)
                    
                    # Create visualization
                    vis_frame = self.create_live_visualization(frame, detections, zone_events, frame_number)
                    
                    # Update session state
                    st.session_state.current_frame = vis_frame
                    st.session_state.live_events.extend(zone_events)
                    
                    # Update performance metrics
                    processing_time = time.time() - start_time
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    
                    st.session_state.performance_metrics.update({
                        'fps': fps,
                        'processing_time': processing_time * 1000,  # ms
                        'frame_count': frame_number,
                        'memory_usage': self.get_memory_usage()
                    })
                    
                    frame_number += 1
                    time.sleep(0.1)  # Control frame rate
                
                cap.release()
                
            except Exception as e:
                self.zone_logger.error(f"Error in video processing thread: {e}")
                st.session_state.video_stream_active = False
        
        # Start processing thread
        if not hasattr(st.session_state, 'video_thread') or not st.session_state.video_thread.is_alive():
            st.session_state.video_thread = threading.Thread(target=video_processing_thread, daemon=True)
            st.session_state.video_thread.start()
        
        st.success("🎬 Live stream started!")
        time.sleep(1)
        st.rerun()

    def stop_live_stream(self):
        """Stop real-time video streaming."""
        st.session_state.video_stream_active = False
        st.session_state.current_frame = None
        st.info("⏹️ Live stream stopped")
        st.rerun()

    def create_live_visualization(self, frame, detections, zone_events, frame_number):
        """Create real-time visualization with zones and detections."""
        vis_frame = frame.copy()
        
        # Draw zones
        vis_frame = self.draw_zones_live(vis_frame)
        
        # Draw detections and events
        vis_frame = self.draw_detections_live(vis_frame, detections, zone_events)
        
        # Add live overlay
        vis_frame = self.add_live_overlay(vis_frame, frame_number)
        
        return vis_frame

    def draw_zones_live(self, frame):
        """Draw zone overlays on live video."""
        # Draw Teller Access Line (Cyan)
        line_y = 600
        cv2.line(frame, (300, line_y), (800, line_y), (255, 255, 0), 3)
        cv2.putText(frame, "Teller Access Line", (300, line_y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw Teller Interaction Zone (Magenta rectangle)
        zone_points = np.array([[400, 100], [700, 100], [700, 300], [400, 300]], np.int32)
        cv2.polylines(frame, [zone_points], True, (255, 0, 255), 2)
        cv2.putText(frame, "Teller Interaction Zone", (400, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame

    def draw_detections_live(self, frame, detections, zone_events):
        """Draw live detections and highlight events."""
        event_people = set()
        for event in zone_events:
            person_id = event.get('person_id', '')
            event_people.add(person_id)
        
        for detection in detections:
            person_id = f"person_{detection.person_id:03d}"
            
            # Create bounding box
            bbox = (int(detection.bbox_x1), int(detection.bbox_y1), 
                   int(detection.bbox_x2), int(detection.bbox_y2))
            
            # Choose color based on events
            if person_id in event_people:
                color = (0, 0, 255)  # Red for events
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw person ID
            cv2.putText(frame, f"ID:{detection.person_id}", 
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Highlight events
            if person_id in event_people:
                cv2.circle(frame, (int(detection.center_x), int(detection.center_y)), 8, (0, 0, 255), -1)
                cv2.putText(frame, "EVENT!", (bbox[0], bbox[3]+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

    def add_live_overlay(self, frame, frame_number):
        """Add live information overlay."""
        overlay = frame.copy()
        
        # Add frame info
        cv2.putText(overlay, f"Live Frame: {frame_number}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add FPS
        fps = st.session_state.performance_metrics.get('fps', 0)
        cv2.putText(overlay, f"FPS: {fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add live status
        cv2.putText(overlay, "🔴 LIVE", 
                   (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return overlay

    def render_live_performance_metrics(self):
        """Render real-time performance metrics."""
        st.subheader("⚡ Live Performance")
        
        metrics = st.session_state.performance_metrics
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("FPS", f"{metrics['fps']:.1f}")
            st.metric("Frame Count", metrics['frame_count'])
        
        with col2:
            st.metric("Processing Time", f"{metrics['processing_time']:.1f}ms")
            st.metric("Memory Usage", f"{metrics['memory_usage']:.1f}MB")
        
        # Performance chart
        if st.session_state.video_stream_active:
            # Create a simple performance chart
            fps_data = [metrics['fps']] * 10  # Simplified for demo
            fig = px.line(y=fps_data, title="Live FPS", range_y=[0, 30])
            st.plotly_chart(fig, use_container_width=True, key="live_fps")

    def render_live_event_stream(self):
        """Render live event stream."""
        st.subheader("📡 Live Events")
        
        # Display recent events
        recent_events = st.session_state.live_events[-5:] if st.session_state.live_events else []
        
        if recent_events:
            for i, event in enumerate(reversed(recent_events)):
                event_type = event.get('event_type', 'Unknown')
                person_id = event.get('person_id', 'Unknown')
                timestamp = event.get('timestamp', datetime.now())
                
                # Format timestamp
                time_str = timestamp.strftime("%H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp)
                
                # Choose emoji based on event type
                if 'line' in str(event_type).lower():
                    emoji = "🚶"
                elif 'teller' in str(event_type).lower():
                    emoji = "✅"
                elif 'abandon' in str(event_type).lower():
                    emoji = "❌"
                else:
                    emoji = "📝"
                
                st.write(f"{emoji} **{time_str}** - {event_type} ({person_id})")
        else:
            st.info("No live events detected yet")
        
        # Auto-refresh during live streaming
        if st.session_state.video_stream_active:
            time.sleep(2)
            st.rerun()

    def get_memory_usage(self):
        """Get current memory usage (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0

    def render_video_upload_tab(self):
        """Render enhanced video upload interface with pipeline integration."""
        st.header("📤 Video Upload & Analysis")
        
        # Processing options
        self._render_processing_options()
        
        # Get current processing status  
        processor = st.session_state.upload_processor
        current_status = processor.get_current_status()
        st.session_state.processing_status = current_status

        # Upload section
        upload_col, status_col = st.columns([1, 1])

        with upload_col:
            st.subheader("📁 Upload Video File")

            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mov", "mp4", "mpg", "avi", "mkv"],
                help="Upload a video file for computer vision analysis",
                disabled=current_status.status == "processing",
            )

            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                save_visualization = st.checkbox(
                    "Save Visualization",
                    value=True,
                    help="Save annotated video output"
                )
                
                use_enhanced_processor = st.checkbox(
                    "Enhanced Processing",
                    value=True,
                    help="Use web-optimized video processor"
                )
            
            with col2:
                processing_mode = st.selectbox(
                    "Processing Mode",
                    ["Full Pipeline", "Detection Only", "Zone Analysis Only"],
                    help="Select processing mode"
                )
                
                frame_skip = st.slider(
                    "Frame Skip",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="Process every Nth frame (higher = faster but less accurate)"
                )

            # Start analysis button
            start_analysis = st.button(
                "🚀 Start Analysis",
                type="primary",
                disabled=uploaded_file is None or current_status.status == "processing",
                help="Begin video processing and analysis",
            )

            # Display upload file info
            if uploaded_file is not None:
                file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
                st.info(f"📋 **File:** {uploaded_file.name}\n\n📏 **Size:** {file_size_mb:.1f} MB")
                
                # Estimated processing time
                estimated_time = self._estimate_processing_time(file_size_mb, frame_skip)
                st.caption(f"⏱️ Estimated processing time: {estimated_time}")

        with status_col:
            self.render_enhanced_processing_status(current_status)

        # Handle start analysis with enhanced pipeline integration
        if start_analysis and uploaded_file is not None:
            try:
                st.info("🔄 Starting enhanced video analysis...")
                
                # Process using enhanced pipeline
                processing_options = {
                    'save_visualization': save_visualization,
                    'processing_mode': processing_mode,
                    'frame_skip': frame_skip,
                    'use_enhanced_processor': use_enhanced_processor
                }
                
                self._start_enhanced_video_processing(uploaded_file, processing_options)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to start processing: {str(e)}")
                self.logger.error(f"Video processing start error: {e}")
    
    def _render_processing_options(self):
        """Render processing configuration options."""
        with st.expander("⚙️ Processing Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Detection Settings")
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum confidence for person detection"
                )
                
                nms_threshold = st.slider(
                    "NMS Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.4,
                    step=0.05,
                    help="Non-maximum suppression threshold"
                )
            
            with col2:
                st.subheader("Zone Settings")
                zone_sensitivity = st.slider(
                    "Zone Sensitivity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Sensitivity for zone event detection"
                )
                
                dwell_time_threshold = st.number_input(
                    "Dwell Time (seconds)",
                    min_value=1,
                    max_value=60,
                    value=3,
                    help="Minimum time in zone to trigger event"
                )
            
            with col3:
                st.subheader("Output Settings")
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "JSON", "Both"],
                    help="Data export format"
                )
                
                include_debug_info = st.checkbox(
                    "Include Debug Info",
                    value=False,
                    help="Include detailed debug information"
                )
            
            # Store settings in session state
            st.session_state.processing_config = {
                'confidence_threshold': confidence_threshold,
                'nms_threshold': nms_threshold,
                'zone_sensitivity': zone_sensitivity,
                'dwell_time_threshold': dwell_time_threshold,
                'export_format': export_format,
                'include_debug_info': include_debug_info
            }
    
    def _estimate_processing_time(self, file_size_mb: float, frame_skip: int) -> str:
        """Estimate processing time based on file size and settings."""
        # Simple estimation: ~1 second per MB with frame skip adjustment
        base_time = file_size_mb * 1.0  # seconds
        adjusted_time = base_time / frame_skip
        
        if adjusted_time < 60:
            return f"{adjusted_time:.0f} seconds"
        elif adjusted_time < 3600:
            return f"{adjusted_time/60:.1f} minutes"
        else:
            return f"{adjusted_time/3600:.1f} hours"
    
    def _start_enhanced_video_processing(self, uploaded_file, options: Dict[str, Any]):
        """Start enhanced video processing with pipeline integration."""
        try:
            # Create temporary file for processing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_video_path = temp_file.name
            
            try:
                # Initialize enhanced processing
                if options.get('use_enhanced_processor', True):
                    # Use our enhanced video processor
                    self._process_with_enhanced_processor(temp_video_path, options)
                else:
                    # Use main pipeline
                    self._process_with_main_pipeline(temp_video_path, options)
                
            finally:
                # Clean up temporary file will be handled by processing thread
                st.session_state.temp_video_path = temp_video_path
                    
        except Exception as e:
            st.error(f"Error starting enhanced processing: {e}")
            self.logger.error(f"Enhanced processing error: {e}")
    
    def render_enhanced_processing_status(self, status: ProcessingStatus):
        """Render enhanced processing status with detailed feedback."""
        st.subheader("⚡ Processing Status")

        status_colors = {
            "idle": "gray", "uploading": "blue", "processing": "orange",
            "completed": "green", "error": "red"
        }

        status_icons = {
            "idle": "⏸️", "uploading": "📤", "processing": "⚙️",
            "completed": "✅", "error": "❌"
        }

        icon = status_icons.get(status.status, "❓")
        st.markdown(f"**Status:** {icon} {status.status.title()}")
        
        # Enhanced progress display
        if status.status == "processing":
            progress_col1, progress_col2 = st.columns([3, 1])
            
            with progress_col1:
                progress_bar = st.progress(status.progress / 100.0)
                st.markdown(f"**Message:** {status.message}")
            
            with progress_col2:
                st.metric("Progress", f"{status.progress:.1f}%")
            
            # Auto-refresh during processing
            if status.progress < 100:
                time.sleep(2)
                st.rerun()
        else:
            st.markdown(f"**Message:** {status.message}")

        # Enhanced results display
        if status.status == "completed" and status.results:
            results = status.results
            st.success("🎉 Processing completed successfully!")

            # Detailed results display
            if results.get("success"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Frames Processed", f"{results.get('total_frames', 0):,}")
                    st.metric("Processing FPS", f"{results.get('average_fps', 0):.1f}")
                
                with col2:
                    st.metric("Total Events", results.get('total_events', 0))
                    st.metric("Processing Time", f"{results.get('processing_time', 0):.1f}s")
                
                with col3:
                    events = results.get('events_by_type', {})
                    st.metric("Line Events", events.get('line_events', 0))
                    st.metric("Teller Events", events.get('teller_interactions', 0))
                
                # Clear cache to show new results
                st.cache_data.clear()

        # Enhanced error display
        elif status.status == "error" and status.error:
            st.error(f"**Error details:** {status.error}")
            
            # Error troubleshooting suggestions
            with st.expander("🔧 Troubleshooting", expanded=False):
                st.write("**Common solutions:**")
                st.write("• Check video file format and integrity")
                st.write("• Ensure sufficient disk space")
                st.write("• Try reducing frame skip or quality settings")
                st.write("• Check system memory usage")
    
    def _process_with_enhanced_processor(self, video_path: str, options: Dict[str, Any]):
        """Process video using enhanced web-optimized processor."""
        def progress_callback(progress_data: Dict[str, Any]):
            # Update session state with progress
            st.session_state.processing_progress = progress_data
            
            # Update performance metrics if available
            if 'stats' in progress_data:
                stats = progress_data['stats']
                st.session_state.performance_metrics.update({
                    'fps': stats.get('fps', 0),
                    'processing_time': stats.get('processing_time', 0) * 1000,
                    'frame_count': stats.get('processed_frames', 0)
                })
                
                # Update system health
                st.session_state.system_health['total_processed_frames'] = stats.get('processed_frames', 0)
                st.session_state.system_health['total_processing_time'] = stats.get('total_time', 0)
            
            # Add events to visualizer if available
            if 'events' in progress_data:
                for event in progress_data['events']:
                    self.event_visualizer.add_live_event(event)
        
        # Set up processor with callback
        self.video_processor.set_progress_callback(progress_callback)
        
        # Start processing in background thread
        import threading
        import os
        
        def process_video_thread():
            try:
                # Update status to processing
                st.session_state.processing_status = ProcessingStatus(
                    status="processing",
                    message="Processing video with enhanced pipeline...",
                    progress=0
                )
                
                # Process video
                results = self.video_processor.process_video_file(video_path)
                
                # Update status to completed
                st.session_state.processing_status = ProcessingStatus(
                    status="completed",
                    message="Video processing completed successfully!",
                    progress=100,
                    results=results
                )
                
            except Exception as e:
                st.session_state.processing_status = ProcessingStatus(
                    status="error",
                    message=f"Processing failed: {str(e)}",
                    error=str(e)
                )
                self.logger.error(f"Video processing thread error: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except:
                        pass
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_video_thread, daemon=True)
        processing_thread.start()
        st.session_state.processing_thread = processing_thread

    def render_processing_status(self, status: ProcessingStatus):
        """Render processing status display."""
        st.subheader("⚡ Processing Status")

        status_colors = {
            "idle": "gray", "uploading": "blue", "processing": "orange",
            "completed": "green", "error": "red"
        }

        status_icons = {
            "idle": "⏸️", "uploading": "📤", "processing": "⚙️",
            "completed": "✅", "error": "❌"
        }

        icon = status_icons.get(status.status, "❓")
        st.markdown(f"**Status:** {icon} {status.status.title()}")
        st.markdown(f"**Message:** {status.message}")

        if status.status == "processing":
            st.progress(status.progress / 100.0)
            st.caption(f"Progress: {status.progress:.1f}%")

            if status.progress < 100:
                time.sleep(2)
                st.rerun()

        if status.status == "completed" and status.results:
            st.success("🎉 Processing completed!")
            results = status.results
            if results.get("success"):
                stats = results.get("processing_stats", {})
                events = results.get("events_generated", {})
                
                st.markdown("**📊 Results Summary:**")
                st.write(f"• Frames processed: {stats.get('total_frames', 0):,}")
                st.write(f"• People detected: {stats.get('unique_persons', 0)}")
                st.write(f"• Line events: {events.get('line_events', 0)}")
                st.write(f"• Teller interactions: {events.get('teller_interactions', 0)}")
                st.write(f"• Abandonment events: {events.get('abandonment_events', 0)}")

    @st.cache_data(ttl=300)
    def load_csv_data(_self, file_type: str) -> pd.DataFrame:
        """Load CSV data with caching."""
        csv_dir = Path(settings.output_csv_dir)
        
        if not csv_dir.exists():
            return pd.DataFrame()
        
        pattern = f"{file_type}_*.csv"
        csv_files = list(csv_dir.glob(pattern))
        
        if not csv_files:
            return pd.DataFrame()
        
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        
        try:
            return pd.read_csv(latest_file, parse_dates=["timestamp"])
        except Exception as e:
            st.error(f"Error loading {file_type} data: {e}")
            return pd.DataFrame()

    def render_analytics_tab(self):
        """Render analytics and charts."""
        st.header("📊 Analytics Dashboard")
        
        # Load summary data
        summary_data = self.load_summary_data()
        
        if summary_data is None:
            st.warning("No processing data available. Please run video processing first.")
            return
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Individuals", summary_data.get("unique_individuals", 0))
        with col2:
            st.metric("🚶 Entered Line", summary_data.get("line_entries", 0))
        with col3:
            st.metric("✅ Teller Interactions", summary_data.get("teller_interactions", 0))
        with col4:
            abandonment_events = summary_data.get("abandonment_events", 0)
            line_entries = summary_data.get("line_entries", 0)
            rate = (abandonment_events / line_entries * 100) if line_entries > 0 else 0
            st.metric("❌ Abandonment Rate", f"{rate:.1f}%")
        
        # Timeline chart
        self.render_timeline_chart()
        
        # Zone analysis
        col1, col2 = st.columns(2)
        with col1:
            self.render_zone_analysis()
        with col2:
            self.render_abandonment_analysis()

    @st.cache_data(ttl=300)
    def load_summary_data(_self) -> Optional[Dict[str, Any]]:
        """Load processing summary data."""
        csv_dir = Path(settings.output_csv_dir)
        summary_files = list(csv_dir.glob("processing_summary_*.csv"))
        
        if not summary_files:
            return None
        
        latest_file = max(summary_files, key=lambda f: f.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file)
            if len(df) > 0:
                return df.iloc[0].to_dict()
        except Exception as e:
            st.error(f"Error loading summary data: {e}")
        
        return None

    def render_timeline_chart(self):
        """Render event timeline chart."""
        st.subheader("📈 Event Timeline")
        
        line_events_df = self.load_csv_data("line_events")
        teller_events_df = self.load_csv_data("teller_interaction_events")
        abandonment_df = self.load_csv_data("abandonment_events")
        
        if line_events_df.empty and teller_events_df.empty and abandonment_df.empty:
            st.info("No event data available for timeline.")
            return
        
        # Simplified timeline chart logic
        timeline_data = []
        
        if not line_events_df.empty:
            for _, row in line_events_df.iterrows():
                timeline_data.append({
                    "timestamp": row["timestamp"],
                    "event_type": "Line Event",
                    "count": 1
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df["time_bin"] = pd.to_datetime(timeline_df["timestamp"]).dt.floor("1min")
            
            agg_timeline = timeline_df.groupby(["time_bin", "event_type"]).agg({"count": "sum"}).reset_index()
            
            fig = px.line(agg_timeline, x="time_bin", y="count", color="event_type",
                         title="Events Over Time")
            st.plotly_chart(fig, use_container_width=True)

    def render_zone_analysis(self):
        """Render zone analysis charts."""
        st.write("**Line Zone Activity**")
        
        line_events_df = self.load_csv_data("line_events")
        
        if not line_events_df.empty:
            event_counts = line_events_df["event_type"].value_counts()
            fig = px.pie(values=event_counts.values, names=event_counts.index,
                        title="Line Zone Events")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No line zone data available.")

    def render_abandonment_analysis(self):
        """Render abandonment analysis."""
        st.write("**Abandonment Analysis**")
        
        abandonment_df = self.load_csv_data("abandonment_events")
        
        if abandonment_df.empty:
            st.info("No abandonment events recorded.")
            return
        
        # Simple abandonment chart
        if "timestamp" in abandonment_df.columns:
            abandonment_df["hour"] = pd.to_datetime(abandonment_df["timestamp"]).dt.hour
            hourly_abandonment = abandonment_df.groupby("hour").size()
            
            fig = px.bar(x=hourly_abandonment.index, y=hourly_abandonment.values,
                        title="Abandonment Events by Hour")
            st.plotly_chart(fig, use_container_width=True)

    def render_zone_config_tab(self):
        """Render zone configuration interface."""
        st.header("🗺️ Zone Configuration")
        
        st.info("🚧 Interactive zone configuration coming in next tasks...")
        
        # For now, show current configuration
        st.subheader("Current Zone Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Line Zone Points**")
            st.text_area("Line Zone", value=settings.line_zone_points, height=100)
        
        with col2:
            st.write("**Teller Zone Points**")
            st.text_area("Teller Zone", value=settings.teller_zone_points, height=100)

    def render_event_monitor_tab(self):
        """Render event monitoring interface."""
        st.header("📋 Event Monitor")
        
        tab1, tab2, tab3 = st.tabs(["Line Events", "Teller Interactions", "Abandonment Events"])
        
        with tab1:
            line_df = self.load_csv_data("line_events")
            if not line_df.empty:
                st.dataframe(line_df.sort_values("timestamp", ascending=False).head(20))
            else:
                st.info("No line events data available.")
        
        with tab2:
            teller_df = self.load_csv_data("teller_interaction_events")
            if not teller_df.empty:
                st.dataframe(teller_df.sort_values("timestamp", ascending=False).head(20))
            else:
                st.info("No teller interaction data available.")
        
        with tab3:
            abandonment_df = self.load_csv_data("abandonment_events")
            if not abandonment_df.empty:
                st.dataframe(abandonment_df.sort_values("timestamp", ascending=False).head(20))
            else:
                st.info("No abandonment events data available.")

    def render_system_status_tab(self):
        """Render system status interface."""
        st.header("⚙️ System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Health")
            st.metric("Live Stream Status", "Active" if st.session_state.video_stream_active else "Inactive")
            st.metric("Memory Usage", f"{self.get_memory_usage():.1f} MB")
            
        with col2:
            st.subheader("Configuration")
            st.write(f"- Model: {settings.model_name}")
            st.write(f"- GPU Enabled: {settings.gpu_enabled}")
            st.write(f"- Debug Mode: {settings.debug_mode}")

    def render_performance_monitor_tab(self):
        """Render comprehensive performance monitoring dashboard."""
        st.header("📈 Performance Monitor")
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Performance overview
        self._render_performance_overview()
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_fps_chart()
            self._render_memory_chart()
        
        with col2:
            self._render_cpu_chart()
            self._render_processing_time_chart()
        
        # System health indicators
        self._render_system_health_indicators()
        
        # Performance statistics table
        self._render_performance_statistics()
        
        # Auto-refresh for real-time monitoring
        if st.session_state.get("performance_auto_refresh", True):
            time.sleep(3)
            st.rerun()
    
    def _update_performance_metrics(self):
        """Update current performance metrics."""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update performance metrics
            current_metrics = {
                'fps': st.session_state.performance_metrics.get('fps', 0),
                'processing_time': st.session_state.performance_metrics.get('processing_time', 0),
                'memory_usage': memory.used / (1024**3),  # GB
                'cpu_usage': cpu_usage,
                'disk_usage': disk.used / disk.total * 100,
                'frame_count': st.session_state.performance_metrics.get('frame_count', 0)
            }
            
            # Try to get GPU usage if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    current_metrics['gpu_usage'] = gpus[0].load * 100
                else:
                    current_metrics['gpu_usage'] = 0
            except:
                current_metrics['gpu_usage'] = 0
            
            # Update session state
            st.session_state.performance_metrics.update(current_metrics)
            
            # Update history
            now = datetime.now()
            history = st.session_state.performance_history
            
            history['timestamps'].append(now)
            history['fps_history'].append(current_metrics['fps'])
            history['memory_history'].append(current_metrics['memory_usage'])
            history['cpu_history'].append(current_metrics['cpu_usage'])
            history['processing_time_history'].append(current_metrics['processing_time'])
            
            # Update system health
            self._update_system_health(current_metrics)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _render_performance_overview(self):
        """Render performance metrics overview."""
        st.subheader("⚡ Current Performance")
        
        metrics = st.session_state.performance_metrics
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "🎬 FPS",
                f"{metrics['fps']:.1f}",
                help="Frames per second processing rate"
            )
        
        with col2:
            st.metric(
                "🖥️ CPU Usage",
                f"{metrics['cpu_usage']:.1f}%",
                help="Current CPU utilization"
            )
        
        with col3:
            st.metric(
                "💾 Memory",
                f"{metrics['memory_usage']:.1f} GB",
                help="Current memory usage"
            )
        
        with col4:
            st.metric(
                "🎮 GPU Usage",
                f"{metrics.get('gpu_usage', 0):.1f}%",
                help="GPU utilization (if available)"
            )
        
        with col5:
            st.metric(
                "⏱️ Proc. Time",
                f"{metrics['processing_time']:.1f}ms",
                help="Frame processing time"
            )
    
    def _render_fps_chart(self):
        """Render FPS performance chart."""
        st.subheader("📊 FPS Performance")
        
        history = st.session_state.performance_history
        
        if len(history['timestamps']) > 1:
            df = pd.DataFrame({
                'timestamp': list(history['timestamps']),
                'fps': list(history['fps_history'])
            })
            
            fig = px.line(
                df,
                x='timestamp',
                y='fps',
                title='FPS Over Time',
                labels={'fps': 'Frames Per Second', 'timestamp': 'Time'}
            )
            
            # Add target FPS line
            fig.add_hline(y=10, line_dash="dash", line_color="green", 
                         annotation_text="Target FPS (10)")
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting FPS data...")
    
    def _render_memory_chart(self):
        """Render memory usage chart."""
        st.subheader("💾 Memory Usage")
        
        history = st.session_state.performance_history
        
        if len(history['timestamps']) > 1:
            df = pd.DataFrame({
                'timestamp': list(history['timestamps']),
                'memory': list(history['memory_history'])
            })
            
            fig = px.area(
                df,
                x='timestamp',
                y='memory',
                title='Memory Usage Over Time',
                labels={'memory': 'Memory (GB)', 'timestamp': 'Time'}
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting memory data...")
    
    def _render_cpu_chart(self):
        """Render CPU usage chart."""
        st.subheader("🖥️ CPU Usage")
        
        history = st.session_state.performance_history
        
        if len(history['timestamps']) > 1:
            df = pd.DataFrame({
                'timestamp': list(history['timestamps']),
                'cpu': list(history['cpu_history'])
            })
            
            fig = px.area(
                df,
                x='timestamp',
                y='cpu',
                title='CPU Usage Over Time',
                labels={'cpu': 'CPU Usage (%)', 'timestamp': 'Time'},
                color_discrete_sequence=['#FF6B6B']
            )
            
            # Add warning line at 80%
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="High Usage (80%)")
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting CPU data...")
    
    def _render_processing_time_chart(self):
        """Render processing time chart."""
        st.subheader("⏱️ Processing Time")
        
        history = st.session_state.performance_history
        
        if len(history['timestamps']) > 1:
            df = pd.DataFrame({
                'timestamp': list(history['timestamps']),
                'processing_time': list(history['processing_time_history'])
            })
            
            fig = px.line(
                df,
                x='timestamp',
                y='processing_time',
                title='Frame Processing Time',
                labels={'processing_time': 'Processing Time (ms)', 'timestamp': 'Time'}
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting processing time data...")
    
    def _render_system_health_indicators(self):
        """Render system health status indicators."""
        st.subheader("🏥 System Health")
        
        health = st.session_state.system_health
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Overall status
            status_color = {
                'healthy': '🟢',
                'warning': '🟡',
                'critical': '🔴'
            }.get(health['status'], '⚫')
            
            st.markdown(f"**Overall Status:** {status_color} {health['status'].title()}")
            
            # Uptime
            uptime = datetime.now() - health['uptime_start']
            st.write(f"**Uptime:** {str(uptime).split('.')[0]}")
        
        with col2:
            # Performance stats
            st.write(f"**Total Frames:** {health['total_processed_frames']:,}")
            st.write(f"**Total Proc. Time:** {health['total_processing_time']:.1f}s")
            
            # Average performance
            if health['total_processed_frames'] > 0:
                avg_time = health['total_processing_time'] / health['total_processed_frames']
                st.write(f"**Avg Frame Time:** {avg_time*1000:.1f}ms")
        
        with col3:
            # Error tracking
            st.write(f"**Errors:** {health['error_count']}")
            
            last_check = health['last_health_check']
            seconds_ago = (datetime.now() - last_check).total_seconds()
            st.write(f"**Last Check:** {int(seconds_ago)}s ago")
        
        # Health alerts
        if health['alerts']:
            st.subheader("⚠️ Active Alerts")
            for alert in health['alerts'][-5:]:  # Show last 5 alerts
                alert_time = alert.get('timestamp', datetime.now())
                alert_msg = alert.get('message', 'Unknown alert')
                alert_level = alert.get('level', 'info')
                
                alert_emoji = {
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'error': '❌',
                    'critical': '🚨'
                }.get(alert_level, 'ℹ️')
                
                st.write(f"{alert_emoji} **{alert_time.strftime('%H:%M:%S')}**: {alert_msg}")
    
    def _render_performance_statistics(self):
        """Render detailed performance statistics table."""
        st.subheader("📋 Performance Statistics")
        
        metrics = st.session_state.performance_metrics
        health = st.session_state.system_health
        
        # Create statistics data
        stats_data = {
            'Metric': [
                'Current FPS',
                'Average FPS',
                'Peak Memory Usage',
                'Current CPU Usage',
                'Peak CPU Usage',
                'Total Frames Processed',
                'Total Processing Time',
                'Average Frame Time',
                'System Uptime',
                'Error Rate'
            ],
            'Value': [
                f"{metrics['fps']:.2f}",
                f"{np.mean(list(st.session_state.performance_history['fps_history'])) if st.session_state.performance_history['fps_history'] else 0:.2f}",
                f"{max(st.session_state.performance_history['memory_history']) if st.session_state.performance_history['memory_history'] else 0:.2f} GB",
                f"{metrics['cpu_usage']:.1f}%",
                f"{max(st.session_state.performance_history['cpu_history']) if st.session_state.performance_history['cpu_history'] else 0:.1f}%",
                f"{health['total_processed_frames']:,}",
                f"{health['total_processing_time']:.1f}s",
                f"{(health['total_processing_time'] / health['total_processed_frames'] * 1000) if health['total_processed_frames'] > 0 else 0:.1f}ms",
                f"{str(datetime.now() - health['uptime_start']).split('.')[0]}",
                f"{(health['error_count'] / health['total_processed_frames'] * 100) if health['total_processed_frames'] > 0 else 0:.2f}%"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Performance controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_refresh = st.checkbox(
                "🔄 Auto Refresh",
                value=st.session_state.get("performance_auto_refresh", True),
                help="Automatically refresh performance metrics"
            )
            st.session_state.performance_auto_refresh = auto_refresh
        
        with col2:
            if st.button("🧹 Clear History", help="Clear performance history"):
                self._clear_performance_history()
                st.success("Performance history cleared")
        
        with col3:
            if st.button("📥 Export Metrics", help="Export performance data"):
                self._export_performance_data()
    
    def _update_system_health(self, metrics: Dict[str, Any]):
        """Update system health status based on current metrics."""
        health = st.session_state.system_health
        alerts = []
        
        # Check various health indicators
        if metrics['cpu_usage'] > 90:
            alerts.append({
                'level': 'critical',
                'message': f"CPU usage critical: {metrics['cpu_usage']:.1f}%",
                'timestamp': datetime.now()
            })
            health['status'] = 'critical'
        elif metrics['cpu_usage'] > 75:
            alerts.append({
                'level': 'warning',
                'message': f"CPU usage high: {metrics['cpu_usage']:.1f}%",
                'timestamp': datetime.now()
            })
            if health['status'] == 'healthy':
                health['status'] = 'warning'
        
        if metrics['memory_usage'] > 12:  # > 12GB
            alerts.append({
                'level': 'critical',
                'message': f"Memory usage critical: {metrics['memory_usage']:.1f}GB",
                'timestamp': datetime.now()
            })
            health['status'] = 'critical'
        elif metrics['memory_usage'] > 8:  # > 8GB
            alerts.append({
                'level': 'warning',
                'message': f"Memory usage high: {metrics['memory_usage']:.1f}GB",
                'timestamp': datetime.now()
            })
            if health['status'] == 'healthy':
                health['status'] = 'warning'
        
        if metrics['fps'] < 2 and st.session_state.video_stream_active:
            alerts.append({
                'level': 'warning',
                'message': f"Low FPS detected: {metrics['fps']:.1f}",
                'timestamp': datetime.now()
            })
            if health['status'] == 'healthy':
                health['status'] = 'warning'
        
        # Add new alerts
        health['alerts'].extend(alerts)
        
        # Keep only recent alerts (last 50)
        health['alerts'] = health['alerts'][-50:]
        
        # Reset to healthy if no recent alerts
        if not alerts and health['status'] != 'healthy':
            recent_alerts = [a for a in health['alerts'] 
                           if (datetime.now() - a['timestamp']).total_seconds() < 300]  # 5 minutes
            if not recent_alerts:
                health['status'] = 'healthy'
        
        health['last_health_check'] = datetime.now()
    
    def _clear_performance_history(self):
        """Clear performance history data."""
        history = st.session_state.performance_history
        for key in history:
            history[key].clear()
    
    def _export_performance_data(self):
        """Export performance data to CSV."""
        try:
            history = st.session_state.performance_history
            
            if not history['timestamps']:
                st.warning("No performance data to export")
                return
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': list(history['timestamps']),
                'fps': list(history['fps_history']),
                'memory_gb': list(history['memory_history']),
                'cpu_percent': list(history['cpu_history']),
                'processing_time_ms': list(history['processing_time_history'])
            })
            
            # Convert to CSV
            csv_data = df.to_csv(index=False)
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="📥 Download Performance Data",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
            
            st.success(f"Performance data ready for download ({len(df)} records)")
            
        except Exception as e:
            st.error(f"Error exporting performance data: {e}")

    def run(self):
        """Main dashboard application."""
        st.title("🎬 PREMISE - Real-time CV Dashboard")
        
        # Render navigation
        tabs = self.render_navigation()
        
        with tabs[0]:  # Live Video Stream
            self.render_live_video_tab()
        
        with tabs[1]:  # Video Upload
            self.render_video_upload_tab()
        
        with tabs[2]:  # Analytics
            self.render_analytics_tab()
        
        with tabs[3]:  # Zone Config
            self.render_zone_config_tab()
        
        with tabs[4]:  # Event Monitor
            self.render_event_monitor_tab()
        
        with tabs[5]:  # Performance Monitor
            self.render_performance_monitor_tab()
        
        with tabs[6]:  # System Status
            self.render_system_status_tab()


# Run the dashboard
if __name__ == "__main__":
    dashboard = EnhancedPremiseDashboard()
    dashboard.run()