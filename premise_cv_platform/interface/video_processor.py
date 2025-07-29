"""
Web-Optimized Video Processing Module

This module adapts the visualize_IDEAL.py logic for Streamlit video display,
handling uploaded video files, frame processing, and real-time analysis
with web-specific optimizations.
"""

import cv2
import numpy as np
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Callable
from collections import deque
import tempfile
import os
from io import BytesIO
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from premise_cv_platform.config.settings import settings
from premise_cv_platform.config.zone_config import ZoneManager, Zone, ZoneType, create_default_zone_manager
from premise_cv_platform.storage.data_schemas import Detection, LineEvent, TellerInteractionEvent, AbandonmentEvent
from premise_cv_platform.utils.logging_config import get_zone_logger
from premise_cv_platform.inference.track_people import PersonTracker
from premise_cv_platform.data_ingestion.process_video import VideoProcessor as BaseVideoProcessor

# Import visualization components
from visualize_IDEAL import UpdatedZoneEventDetector


class StreamlitVideoProcessor:
    """Web-optimized video processor for Streamlit dashboard integration."""
    
    def __init__(self, max_fps: int = 10, max_resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize video processor with web optimizations.
        
        Args:
            max_fps: Maximum frames per second for web display
            max_resolution: Maximum resolution for web display (width, height)
        """
        self.logger = get_zone_logger()
        self.max_fps = max_fps
        self.max_resolution = max_resolution
        self.frame_interval = 1.0 / max_fps
        
        # Initialize CV components
        self.tracker = PersonTracker()
        self.detector = UpdatedZoneEventDetector()
        
        # Processing state
        self.is_processing = False
        self.current_frame = None
        self.frame_queue = deque(maxlen=30)  # Buffer for smooth playback
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'fps': 0,
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Event tracking
        self.all_events = []
        self.recent_events = deque(maxlen=50)
        
        # Progress callback
        self.progress_callback: Optional[Callable] = None
        
        self.logger.info("StreamlitVideoProcessor initialized")
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def _update_progress(self, progress: float, message: str, **kwargs):
        """Update progress through callback if available."""
        if self.progress_callback:
            self.progress_callback({
                'progress': progress,
                'message': message,
                'stats': self.processing_stats.copy(),
                **kwargs
            })
    
    def _optimize_frame_for_web(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimize frame for web display.
        
        Args:
            frame: Input frame
            
        Returns:
            Optimized frame for web display
        """
        # Resize if too large
        height, width = frame.shape[:2]
        max_width, max_height = self.max_resolution
        
        if width > max_width or height > max_height:
            # Calculate scaling factor maintaining aspect ratio
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Apply quality optimizations for web display
        # Slight denoising for better compression
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        return frame
    
    def _create_web_visualization(self, frame: np.ndarray, detections: List[Detection], 
                                 zone_events: List[Dict[str, Any]], frame_number: int,
                                 processing_time: float) -> np.ndarray:
        """
        Create visualization optimized for web display.
        
        Args:
            frame: Input frame
            detections: List of person detections
            zone_events: List of zone events for this frame
            frame_number: Current frame number
            processing_time: Time taken to process this frame
            
        Returns:
            Annotated frame for web display
        """
        vis_frame = frame.copy()
        
        # Draw zones with web-optimized styling
        vis_frame = self._draw_web_zones(vis_frame)
        
        # Draw detections and events
        vis_frame = self._draw_web_detections(vis_frame, detections, zone_events)
        
        # Add web-optimized overlay
        vis_frame = self._add_web_overlay(vis_frame, frame_number, processing_time)
        
        return vis_frame
    
    def _draw_web_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone overlays optimized for web display."""
        height, width = frame.shape[:2]
        
        # Scale zone coordinates based on frame size
        scale_x = width / 1920  # Assuming original coordinates for 1920px width
        scale_y = height / 1080  # Assuming original coordinates for 1080px height
        
        # Draw Teller Access Line (bright cyan for visibility)
        line_y = int(600 * scale_y)
        start_x, end_x = int(300 * scale_x), int(800 * scale_x)
        
        cv2.line(frame, (start_x, line_y), (end_x, line_y), (0, 255, 255), 3)
        cv2.putText(frame, "Teller Access Line", (start_x, line_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6 * min(scale_x, scale_y), (0, 255, 255), 2)
        
        # Draw Teller Interaction Zone (bright magenta)
        zone_x1, zone_y1 = int(400 * scale_x), int(100 * scale_y)
        zone_x2, zone_y2 = int(700 * scale_x), int(300 * scale_y)
        
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 255), 2)
        cv2.putText(frame, "Teller Zone", (zone_x1, zone_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6 * min(scale_x, scale_y), (255, 0, 255), 2)
        
        return frame
    
    def _draw_web_detections(self, frame: np.ndarray, detections: List[Detection], 
                           zone_events: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections and events optimized for web display."""
        height, width = frame.shape[:2]
        
        # Track which people have events this frame
        event_people = set()
        for event in zone_events:
            person_id = event.get('person_id', '')
            event_people.add(person_id)
        
        for detection in detections:
            person_id = f"person_{detection.person_id:03d}"
            
            # Scale bounding box coordinates
            bbox_x1 = int(detection.bbox_x1)
            bbox_y1 = int(detection.bbox_y1)
            bbox_x2 = int(detection.bbox_x2)
            bbox_y2 = int(detection.bbox_y2)
            
            center_x = int(detection.center_x)
            center_y = int(detection.center_y)
            
            # Choose colors for web visibility
            if person_id in event_people:
                color = (0, 69, 255)  # Bright orange for events
                thickness = 3
                cv2.circle(frame, (center_x, center_y), 12, color, -1)  # Larger circle for visibility
            else:
                color = (0, 255, 0)  # Bright green for normal tracking
                thickness = 2
            
            # Draw bounding box with better visibility
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, thickness)
            
            # Draw person ID with background for better readability
            text = f"ID:{detection.person_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(frame, (bbox_x1, bbox_y1 - text_size[1] - 10), 
                         (bbox_x1 + text_size[0] + 10, bbox_y1), color, -1)
            cv2.putText(frame, text, (bbox_x1 + 5, bbox_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Event indicator
            if person_id in event_people:
                cv2.putText(frame, "EVENT!", (bbox_x1, bbox_y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 69, 255), 2)
        
        return frame
    
    def _add_web_overlay(self, frame: np.ndarray, frame_number: int, 
                        processing_time: float) -> np.ndarray:
        """Add information overlay optimized for web display."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay area
        overlay = frame.copy()
        
        # Top-left info panel
        panel_height = 120
        cv2.rectangle(overlay, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add frame information
        y_offset = 35
        cv2.putText(frame, f"Frame: {frame_number}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        fps = self.processing_stats.get('fps', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Process: {processing_time*1000:.1f}ms", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        total_events = len(self.all_events)
        cv2.putText(frame, f"Events: {total_events}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Web indicator
        cv2.putText(frame, "WEB STREAM", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def process_uploaded_video_file(self, uploaded_file, output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process uploaded video file for web display.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            output_callback: Callback for processed frames
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting web processing of uploaded video: {uploaded_file.name}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_video_path = temp_file.name
        
        try:
            # Process the temporary video file
            result = self.process_video_file(temp_video_path, output_callback)
            
            # Clean up temporary file
            os.unlink(temp_video_path)
            
            return result
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise e
    
    def process_video_file(self, video_path: str, output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process video file with web optimizations.
        
        Args:
            video_path: Path to video file
            output_callback: Callback function for processed frames
            
        Returns:
            Processing results dictionary
        """
        self.is_processing = True
        self.processing_stats['start_time'] = datetime.now()
        self.all_events.clear()
        self.recent_events.clear()
        
        self._update_progress(0, "Initializing video processing...")
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.processing_stats['total_frames'] = total_frames
            
            self.logger.info(f"Processing video: {total_frames} frames at {original_fps} FPS")
            self._update_progress(5, f"Video loaded: {total_frames} frames")
            
            # Load YOLO model if not already loaded
            if not hasattr(self.tracker, 'model') or self.tracker.model is None:
                self._update_progress(10, "Loading YOLO model...")
                self.tracker.load_model()
            
            frame_number = 0
            last_process_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Control processing rate for web optimization
                current_time = time.time()
                if current_time - last_process_time < self.frame_interval:
                    frame_number += 1
                    continue
                
                process_start = time.time()
                
                # Optimize frame for web display
                optimized_frame = self._optimize_frame_for_web(frame)
                
                # Run detection and tracking
                detections = self.tracker.detect_and_track(
                    optimized_frame, frame_number, datetime.now()
                )
                
                # Process zone events
                zone_events = self.detector.detect_zone_events(detections)
                
                # Store events
                for event in zone_events:
                    event_data = {
                        'frame': frame_number,
                        'timestamp': datetime.now(),
                        'event': event
                    }
                    self.all_events.append(event_data)
                    self.recent_events.append(event_data)
                
                # Create web-optimized visualization
                processing_time = time.time() - process_start
                vis_frame = self._create_web_visualization(
                    optimized_frame, detections, zone_events, frame_number, processing_time
                )
                
                # Update processing stats
                self.processing_stats['processed_frames'] += 1
                self.processing_stats['fps'] = 1.0 / (time.time() - last_process_time)
                self.processing_stats['processing_time'] = processing_time
                
                # Add to frame queue for smooth playback
                self.frame_queue.append({
                    'frame': vis_frame,
                    'frame_number': frame_number,
                    'timestamp': datetime.now(),
                    'events': zone_events.copy()
                })
                
                # Update current frame
                self.current_frame = vis_frame
                
                # Call output callback if provided
                if output_callback:
                    output_callback(vis_frame, frame_number, zone_events, self.processing_stats)
                
                # Update progress
                progress = (frame_number / total_frames) * 100
                self._update_progress(
                    progress, 
                    f"Processing frame {frame_number}/{total_frames}",
                    current_frame=vis_frame,
                    events=zone_events
                )
                
                frame_number += 1
                last_process_time = current_time
                
                # Log progress every 100 frames
                if frame_number % 100 == 0:
                    self.logger.info(f"Processed {frame_number}/{total_frames} frames")
            
            cap.release()
            
            # Finalize processing stats
            self.processing_stats['end_time'] = datetime.now()
            total_time = (self.processing_stats['end_time'] - self.processing_stats['start_time']).total_seconds()
            
            processing_results = {
                'success': True,
                'total_frames': total_frames,
                'processed_frames': self.processing_stats['processed_frames'],
                'total_events': len(self.all_events),
                'processing_time': total_time,
                'average_fps': self.processing_stats['processed_frames'] / total_time if total_time > 0 else 0,
                'events_by_type': self._categorize_events(),
                'video_path': video_path
            }
            
            self._update_progress(100, "Video processing completed!", results=processing_results)
            
            self.logger.info(f"Web processing completed: {processing_results}")
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Error during web video processing: {e}")
            self._update_progress(0, f"Processing error: {str(e)}", error=str(e))
            return {
                'success': False,
                'error': str(e),
                'total_frames': 0,
                'processed_frames': self.processing_stats.get('processed_frames', 0)
            }
        
        finally:
            self.is_processing = False
    
    def _categorize_events(self) -> Dict[str, int]:
        """Categorize events by type for reporting."""
        event_counts = {
            'line_events': 0,
            'teller_interactions': 0,
            'abandonment_events': 0,
            'other_events': 0
        }
        
        for event_data in self.all_events:
            event = event_data['event']
            event_type = str(event.get('event_type', '')).lower()
            
            if 'line' in event_type:
                event_counts['line_events'] += 1
            elif 'teller' in event_type:
                event_counts['teller_interactions'] += 1
            elif 'abandon' in event_type:
                event_counts['abandonment_events'] += 1
            else:
                event_counts['other_events'] += 1
        
        return event_counts
    
    def get_recent_frames(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent processed frames."""
        return list(self.frame_queue)[-count:] if self.frame_queue else []
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent events."""
        return list(self.recent_events)[-count:] if self.recent_events else []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string for web display."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    
    def stop_processing(self):
        """Stop current processing."""
        self.is_processing = False
        self.logger.info("Video processing stopped by user request")


# Factory function for easy instantiation
def create_streamlit_video_processor(max_fps: int = 10, max_resolution: Tuple[int, int] = (1280, 720)) -> StreamlitVideoProcessor:
    """
    Create a StreamlitVideoProcessor instance with specified parameters.
    
    Args:
        max_fps: Maximum frames per second for processing
        max_resolution: Maximum resolution for web display
        
    Returns:
        Configured StreamlitVideoProcessor instance
    """
    return StreamlitVideoProcessor(max_fps=max_fps, max_resolution=max_resolution)