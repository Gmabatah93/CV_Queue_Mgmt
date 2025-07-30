"""
Simple PREMISE CV Platform Streamlit Dashboard

This module provides a simple web-based dashboard that displays video processing
from visualize_IDEAL.py - just video and zone detection, nothing more.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from premise_cv_platform.inference.track_people import PersonTracker
from premise_cv_platform.data_ingestion.process_video import VideoProcessor
from premise_cv_platform.utils.logging_config import get_zone_logger

# Import the detector from visualize_IDEAL.py
from visualize_IDEAL import UpdatedZoneEventDetector, VisualUpdatedDetector


class SimplePremiseDashboard:
    """Simple dashboard that shows video processing from visualize_IDEAL.py."""
    
    def __init__(self):
        self.logger = get_zone_logger()
        self.tracker = None
        self.detector = None
        self.logger.info("SimplePremiseDashboard initialized")
    
    def run(self):
        """Run the simple dashboard."""
        st.set_page_config(
            page_title="PREMISE CV - Simple Dashboard",
            page_icon="🎥",
            layout="wide"
        )
        
        st.title("🎥 PREMISE CV - Simple Dashboard")
        st.markdown("*Simple video processing dashboard - displays video with zone detection*")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            # Process and display video
            if st.button("🎬 Process Video"):
                self._process_and_display_video(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Show instructions
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload a video file using the file uploader above
        2. Click "Process Video" to start processing
        3. The video will display with zone overlays and person detection
        4. Events will be logged as they occur
        
        **Zone Colors:**
        - 🟡 **Cyan Line**: Teller Access Line
        - 🟣 **Magenta Rectangle**: Teller Interaction Zone  
        - 🟢 **Green Boxes**: People with no zone activity
        - 🔴 **Red Boxes**: People triggering events
        """)
    
    def _process_and_display_video(self, video_path: str):
        """Process and display video with zone detection."""
        st.subheader("🎬 Video Processing")
        
        # Initialize components
        with st.spinner("Loading YOLO model..."):
            self.tracker = PersonTracker()
            self.tracker.load_model()
            self.detector = UpdatedZoneEventDetector()
            
            # Override the detector's simple zones for horizontal orientation
            self._setup_horizontal_zones()
        
        st.success("✅ YOLO model loaded successfully")
        
        # Create placeholders for display
        video_placeholder = st.empty()
        event_placeholder = st.empty()
        
        # Process video
        try:
            with VideoProcessor(video_path) as processor:
                cap = cv2.VideoCapture(video_path)
                
                frame_number = 0
                all_events = []
                
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run person detection and tracking
                    detections = self.tracker.detect_and_track(frame, frame_number, datetime.now())
                    
                    # Process zone events using custom horizontal detection
                    zone_events = self._detect_horizontal_zone_events(detections, frame)
                    
                    # Create visualization (same as visualize_IDEAL.py)
                    vis_frame = self._create_visualization(frame, detections, zone_events, frame_number)
                    
                    # Display frame
                    with video_placeholder.container():
                        # Convert BGR to RGB for Streamlit
                        rgb_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame, caption=f"Frame {frame_number}")
                    
                    # Log events
                    for event in zone_events:
                        all_events.append({
                            'frame': frame_number,
                            'event': event
                        })
                    
                    # Display events
                    with event_placeholder.container():
                        st.subheader(f"📝 Events (Total: {len(all_events)})")
                        if zone_events:
                            for event in zone_events:
                                event_type = event.get('event_type', 'unknown')
                                person_id = event.get('person_id', 'unknown')
                                st.write(f"🔔 Frame {frame_number}: **{event_type}** - Person {person_id}")
                        elif all_events:
                            # Show last few events
                            for event_data in all_events[-5:]:
                                frame = event_data['frame']
                                event = event_data['event']
                                event_type = event.get('event_type', 'unknown')
                                person_id = event.get('person_id', 'unknown')
                                st.write(f"Frame {frame}: {event_type} - Person {person_id}")
                    
                    # Update progress
                    progress = min(frame_number / total_frames, 1.0)
                    progress_bar.progress(progress)
                    
                    frame_number += 1
                    
                    # Small delay to make it watchable
                    time.sleep(0.1)
                
                cap.release()
                
                # Show final results
                st.success("✅ Video processing complete!")
                st.subheader("📊 Final Results")
                st.write(f"**Total Events:** {len(all_events)}")
                
                if all_events:
                    st.subheader("📝 Complete Event Log:")
                    for event_data in all_events:
                        frame = event_data['frame']
                        event = event_data['event']
                        event_type = event.get('event_type', 'unknown')
                        person_id = event.get('person_id', 'unknown')
                        st.write(f"Frame {frame}: {event_type} - Person {person_id}")
                
        except Exception as e:
            st.error(f"❌ Error during video processing: {e}")
            self.logger.error(f"Video processing error: {e}")
    
    def _create_visualization(self, frame, detections, zone_events, frame_number):
        """Create visualization exactly like visualize_IDEAL.py."""
        vis_frame = frame.copy()
        
        # Draw zones (same as visualize_IDEAL.py)
        vis_frame = self._draw_zones(vis_frame)
        
        # Draw detections and events (same as visualize_IDEAL.py)
        vis_frame = self._draw_detections_and_events(vis_frame, detections, zone_events, frame_number)
        
        # Add overlay (same as visualize_IDEAL.py)
        vis_frame = self._add_overlay(vis_frame, frame_number)
        
        return vis_frame
    
    def _draw_zones(self, frame):
        """Draw the zones on the frame adjusted for horizontal video orientation."""
        # Get frame dimensions to adjust coordinates for horizontal orientation
        height, width = frame.shape[:2]
        
        # Adjust coordinates for horizontal video (assuming 90-degree rotation difference)
        # Original: line_y=600, x_range=(300, 800) - vertical orientation
        # Adjusted for horizontal: scale coordinates to frame size
        
        # Draw Teller Access Line (Cyan) - adjusted a little to the right
        line_x = int(width * 0.4)  # Adjusted slightly to the right
        cv2.line(frame, (line_x, int(height * 0.2)), (line_x, int(height * 0.8)), (255, 255, 0), 3)
        cv2.putText(frame, "Teller Access Line", (line_x + 10, int(height * 0.3)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw Teller Interaction Zone (Magenta rectangle) - shifted a bit more left
        # Position at the very left edge of the frame
        zone_x1, zone_y1 = int(width * 0.05), int(height * 0.2)  # Shifted to very left edge
        zone_x2, zone_y2 = int(width * 0.2), int(height * 0.8)   # Same narrow width, at left edge
        zone_points = np.array([[zone_x1, zone_y1], [zone_x2, zone_y1], 
                               [zone_x2, zone_y2], [zone_x1, zone_y2]], np.int32)
        cv2.polylines(frame, [zone_points], True, (255, 0, 255), 2)
        cv2.putText(frame, "Teller Interaction Zone", (zone_x1, zone_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return frame
    
    def _draw_detections_and_events(self, frame, detections, zone_events, frame_number):
        """Draw detections and highlight events (same as visualize_IDEAL.py)."""
        # Track which people have events this frame
        event_people = set()
        for event in zone_events:
            person_id = event.get('person_id', '')
            event_people.add(person_id)
        
        for detection in detections:
            person_id = f"person_{detection.person_id:03d}"
            center = (int(detection.center_x), int(detection.center_y))
            
            # Create bounding box
            bbox = (int(detection.bbox_x1), int(detection.bbox_y1), 
                   int(detection.bbox_x2), int(detection.bbox_y2))
            
            # Check zone status with adjusted coordinates for horizontal orientation
            frame_height, frame_width = frame.shape[:2]
            
            # Adjusted line crossing check (vertical line instead of horizontal)
            line_x = int(frame_width * 0.4)  # Adjusted slightly right
            in_line = self._check_vertical_line_crossing(bbox, line_x)
            
            # Adjusted zone intersection check
            zone_config = {
                'x_range': (int(frame_width * 0.05), int(frame_width * 0.2)),   # At very left edge
                'y_range': (int(frame_height * 0.2), int(frame_height * 0.8))   # Taller, more vertical
            }
            in_zone = self._check_zone_intersection_adjusted(bbox, zone_config)
            
            # Choose color based on zone status and events
            if person_id in event_people:
                color = (0, 0, 255)  # Red for people with events
                thickness = 3
            elif in_line:
                color = (0, 255, 255)  # Cyan for line crossing
                thickness = 2
            elif in_zone:
                color = (255, 0, 255)  # Magenta for zone intersection
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for no zone
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Draw person ID
            cv2.putText(frame, f"ID:{detection.person_id}", 
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Highlight people with events
            if person_id in event_people:
                cv2.circle(frame, center, 8, (0, 0, 255), -1)  # Red dot
                cv2.putText(frame, "EVENT!", (bbox[0], bbox[3]+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def _add_overlay(self, frame, frame_number):
        """Add information overlay (same as visualize_IDEAL.py)."""
        # Create overlay
        overlay = frame.copy()
        
        # Add frame info
        cv2.putText(overlay, f"Frame: {frame_number}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add zone info
        y_pos = 60
        cv2.putText(overlay, "Zone Colors:", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
        cv2.putText(overlay, "- Cyan: Teller Access Line", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 20
        cv2.putText(overlay, "- Magenta: Teller Interaction Zone", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y_pos += 20
        cv2.putText(overlay, "- Green: No Zone", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += 20
        cv2.putText(overlay, "- Red: Event Triggered", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return overlay
    
    def _setup_horizontal_zones(self):
        """Set up zones adjusted for horizontal video orientation."""
        # This will be applied dynamically based on frame dimensions during processing
        # The detector will use the frame-relative coordinates
        pass
    
    def _detect_horizontal_zone_events(self, detections, frame):
        """Detect zone events adjusted for horizontal video orientation."""
        events = []
        frame_height, frame_width = frame.shape[:2]
        
        # Define zones for horizontal orientation
        line_x = int(frame_width * 0.4)  # Vertical line position - adjusted slightly right
        teller_zone = {
            'x_range': (int(frame_width * 0.05), int(frame_width * 0.2)),   # At very left edge
            'y_range': (int(frame_height * 0.2), int(frame_height * 0.8))   # Taller, more vertical
        }
        
        for detection in detections:
            person_id = f"person_{detection.person_id:03d}"
            timestamp = detection.timestamp
            
            # Create bounding box
            bbox = (int(detection.bbox_x1), int(detection.bbox_y1), 
                   int(detection.bbox_x2), int(detection.bbox_y2))
            
            # Check line crossing (vertical line for horizontal video)
            if self._check_vertical_line_crossing(bbox, line_x):
                events.append({
                    'event_type': 'line_entered',  # Simplified for horizontal view
                    'person_id': person_id,
                    'timestamp': timestamp,
                    'zone_id': 'teller_access_line'
                })
            
            # Check teller zone intersection
            if self._check_zone_intersection_adjusted(bbox, teller_zone):
                events.append({
                    'event_type': 'teller_zone_entered',
                    'person_id': person_id,
                    'timestamp': timestamp,
                    'zone_id': 'teller_interaction'
                })
        
        return events
    
    def _check_vertical_line_crossing(self, bbox: Tuple[int, int, int, int], line_x: int) -> bool:
        """Check if bounding box crosses a vertical line (adjusted for horizontal video)."""
        x1, y1, x2, y2 = bbox
        return (x1 < line_x < x2) or (x2 < line_x < x1)
    
    def _check_zone_intersection_adjusted(self, bbox: Tuple[int, int, int, int], zone: Dict) -> bool:
        """Check if bounding box intersects with zone (adjusted for horizontal video)."""
        x1, y1, x2, y2 = bbox
        zone_x_min, zone_x_max = zone['x_range']
        zone_y_min, zone_y_max = zone['y_range']
        return not (x2 < zone_x_min or x1 > zone_x_max or 
                    y2 < zone_y_min or y1 > zone_y_max)


def main():
    """Main function to run the simple dashboard."""
    dashboard = SimplePremiseDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()