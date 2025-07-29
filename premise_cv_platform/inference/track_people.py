"""
YOLO11 person detection and tracking with persistence for PREMISE CV Platform.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import torch
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics not installed. Run: pip install ultralytics")
    raise

from premise_cv_platform.config.settings import settings
from premise_cv_platform.storage.data_schemas import Detection, PersonTrackingState
from premise_cv_platform.utils.logging_config import get_detection_logger, PerformanceTimer


class PersonTracker:
    """YOLO11 person detection and tracking with ID persistence."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.model_name
        self.model: Optional[YOLO] = None
        self.tracking_states: Dict[int, PersonTrackingState] = {}
        self.detection_logger = get_detection_logger()
        
        # Detection parameters
        self.confidence_threshold = settings.model_confidence_threshold
        self.iou_threshold = settings.model_iou_threshold
        self.person_class_id = 0  # COCO class ID for person
        
        # Performance tracking
        self.total_detections = 0
        self.successful_tracks = 0
        
        self.detection_logger.info(f"PersonTracker initialized with model: {self.model_name}")
    
    def load_model(self) -> None:
        """Load YOLO model with proper error handling."""
        try:
            model_path = Path("premise_cv_platform/inference/models") / self.model_name
            
            # If model doesn't exist locally, let ultralytics download it
            if not model_path.exists():
                self.detection_logger.info(f"Model not found locally, downloading: {self.model_name}")
                self.model = YOLO(self.model_name)
            else:
                self.model = YOLO(str(model_path))
            
            # Set device (GPU if available and enabled)
            if settings.gpu_enabled and torch.cuda.is_available():
                device = 'cuda'
                self.detection_logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                self.detection_logger.info("Using CPU for inference")
            
            self.model.to(device)
            self.detection_logger.info(f"YOLO model loaded successfully: {self.model_name}")
            
        except Exception as e:
            self.detection_logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def detect_and_track(self, frame: np.ndarray, frame_number: int, 
                        timestamp: datetime) -> List[Detection]:
        """
        Run YOLO detection and tracking on frame.
        
        Args:
            frame: Input video frame  
            frame_number: Frame number for tracking
            timestamp: Frame timestamp
            
        Returns:
            List of Detection objects with tracking IDs
        """
        if self.model is None:
            self.load_model()
        
        detections = []
        
        with PerformanceTimer(f"YOLO inference frame {frame_number}"):
            try:
                # Run YOLO with tracking persistence
                results = self.model.track(
                    frame,
                    persist=settings.tracking_persistence,
                    classes=[self.person_class_id],  # Only detect persons
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Process results
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Check if detections exist
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        
                        # Get tracking IDs if available
                        track_ids = boxes.id
                        if track_ids is not None:
                            track_ids = track_ids.cpu().numpy().astype(int)
                        else:
                            # Fallback: use index as ID if tracking fails
                            track_ids = list(range(len(boxes)))
                            self.detection_logger.warning(f"No tracking IDs available for frame {frame_number}")
                        
                        # Extract detection data
                        xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                        conf = boxes.conf.cpu().numpy()  # Confidence scores
                        
                        for i, (box, confidence) in enumerate(zip(xyxy, conf)):
                            if i < len(track_ids):
                                person_id = int(track_ids[i])
                                
                                x1, y1, x2, y2 = box
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # Create Detection object
                                detection = Detection(
                                    timestamp=timestamp,
                                    frame_number=frame_number,
                                    person_id=person_id,
                                    confidence=float(confidence),
                                    bbox_x1=float(x1),
                                    bbox_y1=float(y1),
                                    bbox_x2=float(x2),
                                    bbox_y2=float(y2),
                                    center_x=float(center_x),
                                    center_y=float(center_y)
                                )
                                
                                detections.append(detection)
                                
                                # Update tracking state
                                self._update_tracking_state(person_id, timestamp, confidence)
                                
                                self.total_detections += 1
                
                if detections:
                    self.detection_logger.debug(f"Frame {frame_number}: Detected {len(detections)} persons")
                
            except Exception as e:
                self.detection_logger.error(f"Detection failed for frame {frame_number}: {e}")
        
        return detections
    
    def _update_tracking_state(self, person_id: int, timestamp: datetime, 
                              confidence: float) -> None:
        """Update tracking state for a person."""
        if person_id not in self.tracking_states:
            # Create new tracking state
            self.tracking_states[person_id] = PersonTrackingState(
                person_id=f"person_{person_id:03d}",
                first_detected=timestamp,
                last_detected=timestamp
            )
            self.successful_tracks += 1
            self.detection_logger.debug(f"New person tracked: person_{person_id:03d}")
        else:
            # Update existing state
            state = self.tracking_states[person_id]
            state.last_detected = timestamp
        
        # Update detection statistics
        self.tracking_states[person_id].update_detection_stats(confidence)
    
    def get_person_state(self, person_id: int) -> Optional[PersonTrackingState]:
        """Get tracking state for a specific person."""
        return self.tracking_states.get(person_id)
    
    def get_active_persons(self, time_threshold: float = 5.0) -> List[PersonTrackingState]:
        """Get persons detected within the last N seconds."""
        current_time = datetime.now()
        active_persons = []
        
        for state in self.tracking_states.values():
            time_diff = (current_time - state.last_detected).total_seconds()
            if time_diff <= time_threshold:
                active_persons.append(state)
        
        return active_persons
    
    def cleanup_old_tracks(self, max_age_seconds: float = 30.0) -> int:
        """Remove tracking states for persons not seen recently."""
        current_time = datetime.now()
        removed_count = 0
        
        person_ids_to_remove = []
        for person_id, state in self.tracking_states.items():
            time_diff = (current_time - state.last_detected).total_seconds()
            if time_diff > max_age_seconds:
                person_ids_to_remove.append(person_id)
        
        for person_id in person_ids_to_remove:
            del self.tracking_states[person_id]
            removed_count += 1
        
        if removed_count > 0:
            self.detection_logger.info(f"Cleaned up {removed_count} old tracking states")
        
        return removed_count
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        stats = {
            "total_detections": self.total_detections,
            "unique_persons_tracked": len(self.tracking_states),
            "successful_tracks": self.successful_tracks,
            "active_persons": len(self.get_active_persons()),
            "average_detections_per_person": 0,
            "tracking_quality": 0
        }
        
        if self.tracking_states:
            total_person_detections = sum(
                state.total_detections for state in self.tracking_states.values()
            )
            stats["average_detections_per_person"] = (
                total_person_detections / len(self.tracking_states)
            )
            
            # Calculate tracking quality based on detection consistency
            avg_confidence = np.mean([
                state.average_confidence for state in self.tracking_states.values()
            ])
            stats["tracking_quality"] = float(avg_confidence)
        
        return stats
    
    def visualize_detections(self, frame: np.ndarray, 
                           detections: List[Detection]) -> np.ndarray:
        """Draw detection results on frame for visualization."""
        vis_frame = frame.copy()
        
        for detection in detections:
            # Draw bounding box
            x1, y1, x2, y2 = (
                int(detection.bbox_x1), int(detection.bbox_y1),
                int(detection.bbox_x2), int(detection.bbox_y2)
            )
            
            # Color based on confidence
            confidence = detection.confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw person ID and confidence
            label = f"ID: {detection.person_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(vis_frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw center point
            center_x, center_y = int(detection.center_x), int(detection.center_y)
            cv2.circle(vis_frame, (center_x, center_y), 4, color, -1)
        
        # Add statistics overlay
        stats_text = f"Detections: {len(detections)} | Total Tracked: {len(self.tracking_states)}"
        cv2.putText(vis_frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def save_model_locally(self, save_path: Optional[str] = None) -> Path:
        """Save the loaded model to local path."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        if save_path is None:
            models_dir = Path("premise_cv_platform/inference/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            save_path = models_dir / self.model_name
        
        # Save model
        torch.save(self.model.model.state_dict(), save_path)
        self.detection_logger.info(f"Model saved to: {save_path}")
        
        return Path(save_path)
    
    def export_tracking_data(self) -> Dict[str, Any]:
        """Export all tracking data for analysis."""
        export_data = {
            "statistics": self.get_tracking_statistics(),
            "tracking_states": {},
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Export tracking states
        for person_id, state in self.tracking_states.items():
            export_data["tracking_states"][person_id] = {
                "person_id": state.person_id,
                "first_detected": state.first_detected.isoformat(),
                "last_detected": state.last_detected.isoformat(),
                "total_detections": state.total_detections,
                "average_confidence": state.average_confidence,
                "current_zones": state.current_zones,
                "zone_history": state.zone_history
            }
        
        return export_data


# Import cv2 at the end to avoid circular import issues
import cv2