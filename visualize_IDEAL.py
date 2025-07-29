#!/usr/bin/env python3
"""
Visualize Updated Zone Event Detector

This script provides real-time visualization of the updated detector
showing zone events, person tracking, and event logging.
"""

import cv2
import numpy as np
import sys
import math
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from loguru import logger

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from premise_cv_platform.config.settings import settings
from premise_cv_platform.config.zone_config import ZoneManager, Zone, ZoneType, create_default_zone_manager
from premise_cv_platform.storage.data_schemas import (
    Detection, LineEvent, TellerInteractionEvent, AbandonmentEvent, EventType
)
from premise_cv_platform.utils.logging_config import get_zone_logger
from premise_cv_platform.inference.track_people import PersonTracker
from premise_cv_platform.data_ingestion.process_video import VideoProcessor


class UpdatedZoneEventDetector:
    """Updated zone-based event detection using simple bounding box logic."""
    
    def __init__(self, zone_manager: Optional[ZoneManager] = None):
        self.zone_manager = zone_manager or self._create_default_zones()
        self.tracking_states: Dict[str, Dict[str, Any]] = {}  # person_id -> state
        self.zone_states: Dict[str, Dict[str, bool]] = {}  # person_id -> zone_id -> in_zone
        self.zone_logger = get_zone_logger()
        
        # Event storage
        self.line_events: List[LineEvent] = []
        self.teller_events: List[TellerInteractionEvent] = []
        self.abandonment_events: List[AbandonmentEvent] = []
        
        # Dwell time tracking
        self.zone_entry_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)  # person_id -> zone_id -> entry_time
        
        # Simple zone definitions for bounding box logic
        self.simple_zones = self._create_simple_zones()
        
        self.zone_logger.info("UpdatedZoneEventDetector initialized")
    
    def _create_default_zones(self) -> ZoneManager:
        """Create default zone manager from settings."""
        line_points = settings.get_line_zone_coordinates()
        teller_points = settings.get_teller_zone_coordinates()
        
        return create_default_zone_manager(line_points, teller_points)
    
    def _create_simple_zones(self) -> Dict[str, Dict]:
        """Create simple zone definitions for bounding box logic."""
        return {
            'teller_access_line': {
                'y_position': 600,
                'x_range': (300, 800),
                'name': 'Teller Access Line',
                'zone_type': ZoneType.LINE
            },
            'teller_interaction': {
                'x_range': (400, 700),
                'y_range': (100, 300),
                'name': 'Teller Interaction Zone',
                'zone_type': ZoneType.TELLER
            }
        }
    
    def _check_line_crossing(self, bbox: Tuple[float, float, float, float], line_y: float) -> bool:
        """Check if bounding box crosses the horizontal line."""
        x1, y1, x2, y2 = bbox
        return (y1 < line_y < y2) or (y2 < line_y < y1)
    
    def _check_zone_intersection(self, bbox: Tuple[float, float, float, float], zone: Dict) -> bool:
        """Check if bounding box intersects with the zone."""
        x1, y1, x2, y2 = bbox
        zone_x_min, zone_x_max = zone['x_range']
        zone_y_min, zone_y_max = zone['y_range']
        return not (x2 < zone_x_min or x1 > zone_x_max or 
                    y2 < zone_y_min or y1 > zone_y_max)
    
    def detect_zone_events(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        """Detect zone-based events using simple bounding box logic."""
        events = []
        
        for detection in detections:
            person_id = f"person_{detection.person_id:03d}"
            timestamp = detection.timestamp
            
            # Create bounding box
            bbox = (detection.bbox_x1, detection.bbox_y1, 
                   detection.bbox_x2, detection.bbox_y2)
            
            # Initialize tracking states if needed
            if person_id not in self.tracking_states:
                self.tracking_states[person_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'line_entries': [],
                    'line_exits': [],
                    'teller_interactions': []
                }
                self.zone_states[person_id] = {}
            
            # Update last seen
            self.tracking_states[person_id]['last_seen'] = timestamp
            
            # Check each simple zone
            for zone_id, zone_config in self.simple_zones.items():
                # Check if person is in zone using bounding box logic
                if zone_config['zone_type'] == ZoneType.LINE:
                    in_zone = self._check_line_crossing(bbox, zone_config['y_position'])
                else:
                    in_zone = self._check_zone_intersection(bbox, zone_config)
                
                # Get previous state
                was_in_zone = self.zone_states[person_id].get(zone_id, False)
                
                # Detect zone entry
                if in_zone and not was_in_zone:
                    event = self._handle_zone_entry(person_id, zone_id, zone_config, timestamp)
                    if event:
                        events.append(event)
                    
                    # Record entry time for dwell time calculation
                    self.zone_entry_times[person_id][zone_id] = timestamp
                
                # Detect zone exit
                elif not in_zone and was_in_zone:
                    event = self._handle_zone_exit(person_id, zone_id, zone_config, timestamp)
                    if event:
                        events.append(event)
                    
                    # Remove entry time
                    if zone_id in self.zone_entry_times[person_id]:
                        del self.zone_entry_times[person_id][zone_id]
                
                # Update zone state
                self.zone_states[person_id][zone_id] = in_zone
                
                # Check for teller interaction (dwell time)
                if (zone_config['zone_type'] == ZoneType.TELLER and 
                    in_zone and self._check_teller_dwell_time(person_id, zone_id, timestamp)):
                    interaction_event = self._create_teller_interaction_event(person_id, zone_id, timestamp)
                    if interaction_event:
                        events.append(interaction_event)
        
        return events
    
    def _handle_zone_entry(self, person_id: str, zone_id: str, zone_config: Dict, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Handle zone entry event."""
        if zone_config['zone_type'] == ZoneType.LINE:
            event_type = EventType.LINE_ENTERED
            event_data = {
                'event_type': event_type,
                'person_id': person_id,
                'line_zone_id': zone_id,
                'timestamp': timestamp
            }
        else:
            event_data = {
                'event_type': 'teller_zone_entered',
                'person_id': person_id,
                'zone_id': zone_id,
                'timestamp': timestamp
            }
        
        self.zone_logger.info(f"{zone_config['name']} entry: {person_id} entered {zone_id} at {timestamp}")
        return event_data
    
    def _handle_zone_exit(self, person_id: str, zone_id: str, zone_config: Dict, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Handle zone exit event."""
        if zone_config['zone_type'] == ZoneType.LINE:
            event_type = EventType.LINE_EXITED
            event_data = {
                'event_type': event_type,
                'person_id': person_id,
                'line_zone_id': zone_id,
                'timestamp': timestamp
            }
        else:
            event_data = {
                'event_type': 'teller_zone_exited',
                'person_id': person_id,
                'zone_id': zone_id,
                'timestamp': timestamp
            }
        
        self.zone_logger.info(f"{zone_config['name']} exit: {person_id} exited {zone_id} at {timestamp}")
        return event_data
    
    def _check_teller_dwell_time(self, person_id: str, zone_id: str, current_time: datetime) -> bool:
        """Check if person has spent sufficient time in teller zone for interaction."""
        if zone_id not in self.zone_entry_times[person_id]:
            return False
        
        entry_time = self.zone_entry_times[person_id][zone_id]
        dwell_time = current_time - entry_time
        return dwell_time >= timedelta(seconds=2)
    
    def _create_teller_interaction_event(self, person_id: str, zone_id: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Create a teller interaction event."""
        event_data = {
            'event_type': EventType.TELLER_INTERACTED,
            'person_id': person_id,
            'teller_zone_id': zone_id,
            'timestamp': timestamp
        }
        
        self.zone_logger.info(f"Teller interaction: {person_id} interacted with {zone_id} at {timestamp}")
        return event_data


class VisualUpdatedDetector:
    """Visualize the updated zone event detector in real-time."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.tracker = PersonTracker()
        self.detector = UpdatedZoneEventDetector()
        
        # Visualization tracking
        self.all_events = []
        self.event_log = []
        
        # Load YOLO model
        print(" Loading YOLO model for visualization...")
        self.tracker.load_model()
        print("✅ YOLO model loaded successfully")
    
    def run_visualization(self):
        """Run the real-time visualization."""
        print("\n🎬 Visualizing Updated Zone Event Detector")
        print("=" * 50)
        print(f" Input Video: {self.video_path}")
        print(f" Purpose: Real-time visualization of zone events")
        print(f" Controls:")
        print(f"   - Press 'q' to quit")
        print(f"   - Press 'p' to pause/resume")
        print(f"   - Press 's' to save current frame")
        print("=" * 50)
        
        try:
            with VideoProcessor(self.video_path) as processor:
                cap = cv2.VideoCapture(self.video_path)
                
                frame_number = 0
                paused = False
                
                while True:
                    if not paused:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Run person detection and tracking
                        detections = self.tracker.detect_and_track(frame, frame_number, datetime.now())
                        
                        # Process zone events
                        zone_events = self.detector.detect_zone_events(detections)
                        
                        # Create visualization
                        vis_frame = self.create_visualization(frame, detections, zone_events, frame_number)
                        
                        # Display the frame
                        cv2.imshow('Updated Zone Event Detector', vis_frame)
                        
                        # Log events
                        for event in zone_events:
                            self.all_events.append({
                                'frame': frame_number,
                                'event': event
                            })
                        
                        frame_number += 1
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f" {'⏸️  Paused' if paused else '▶️  Resumed'}")
                    elif key == ord('s'):
                        filename = f"updated_detector_frame_{frame_number:06d}.jpg"
                        cv2.imwrite(filename, vis_frame)
                        print(f"💾 Saved frame: {filename}")
                
                cap.release()
                cv2.destroyAllWindows()
                
                # Show final results
                self.show_final_results()
                
        except Exception as e:
            print(f"❌ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_visualization(self, frame, detections, zone_events, frame_number):
        """Create visualization of the current frame."""
        vis_frame = frame.copy()
        
        # Draw zones
        vis_frame = self.draw_zones(vis_frame)
        
        # Draw detections and events
        vis_frame = self.draw_detections_and_events(vis_frame, detections, zone_events, frame_number)
        
        # Add overlay
        vis_frame = self.add_overlay(vis_frame, frame_number)
        
        return vis_frame
    
    def draw_zones(self, frame):
        """Draw the zones on the frame."""
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
    
    def draw_detections_and_events(self, frame, detections, zone_events, frame_number):
        """Draw detections and highlight events."""
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
            
            # Check zone status
            in_line = self.detector._check_line_crossing(bbox, 600)
            in_zone = self.detector._check_zone_intersection(bbox, {
                'x_range': (400, 700), 'y_range': (100, 300)
            })
            
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
    
    def add_overlay(self, frame, frame_number):
        """Add information overlay."""
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
        
        # Add event count
        y_pos += 40
        cv2.putText(overlay, f"Total Events: {len(self.all_events)}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return overlay
    
    def show_final_results(self):
        """Show final results after visualization."""
        print(f"\n📊 Visualization Results")
        print("=" * 50)
        
        print(f" Total Events: {len(self.all_events)}")
        
        if self.all_events:
            print(f"\n📝 Event Log:")
            for event_data in self.all_events:
                frame = event_data['frame']
                event = event_data['event']
                print(f"   Frame {frame}: {event.get('event_type')} - "
                      f"Person {event.get('person_id')}")
        
        print(f"\n✅ Visualization Complete!")


def main():
    """Main function to run the visualization."""
    video_path = "videos/bank_sample.MOV"
    
    print("🎬 Visualizing Updated Zone Event Detector")
    print("=" * 50)
    print(f"📹 Input Video: {video_path}")
    print(f" Purpose: Real-time visualization of the working detector")
    print("=" * 50)
    
    # Create visualizer
    visualizer = VisualUpdatedDetector(video_path)
    
    # Run visualization
    visualizer.run_visualization()
    
    print(f"\n✅ Visualization Complete!")


if __name__ == "__main__":
    main()