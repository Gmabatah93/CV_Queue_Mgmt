"""
Video processing module for PREMISE CV Platform with OpenCV VideoCapture.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger

from premise_cv_platform.config.settings import settings
from premise_cv_platform.utils.logging_config import get_video_logger, PerformanceTimer


class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass


class VideoProcessor:
    """Core video processing with OpenCV VideoCapture and proper resource management."""
    
    def __init__(self, video_path: Optional[str] = None):
        self.video_path = Path(video_path or settings.video_path)
        self.video_logger = get_video_logger()
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_properties: Dict[str, Any] = {}
        
        self.video_logger.info(f"VideoProcessor initialized for: {self.video_path}")
    
    def _validate_video_path(self) -> None:
        """Validate that video file exists and is readable."""
        if not self.video_path.exists():
            raise VideoProcessingError(f"Video file not found: {self.video_path}")
        
        if not self.video_path.is_file():
            raise VideoProcessingError(f"Path is not a file: {self.video_path}")
        
        # Check file extension
        valid_extensions = {'.mov', '.mp4', '.avi', '.mkv', '.wmv', '.flv'}
        if self.video_path.suffix.lower() not in valid_extensions:
            self.video_logger.warning(
                f"Unusual video extension: {self.video_path.suffix}. "
                f"Supported: {valid_extensions}"
            )
    
    def open_video(self) -> None:
        """Open video file and extract properties."""
        self._validate_video_path()
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise VideoProcessingError(f"Cannot open video: {self.video_path}")
        
        # Extract video properties for logging and validation
        self.video_properties = {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": self.cap.get(cv2.CAP_PROP_FOURCC),
            "duration_seconds": None
        }
        
        # Calculate duration
        if self.video_properties["fps"] > 0:
            self.video_properties["duration_seconds"] = (
                self.video_properties["total_frames"] / self.video_properties["fps"]
            )
        
        # Log video properties following examples/ingestion_log.txt format
        self.video_logger.info(f"INFO: Video ingestion started for: {self.video_path.name}")
        self.video_logger.info(f"INFO: Video resolution: {self.video_properties['width']}x{self.video_properties['height']}")
        self.video_logger.info(f"INFO: Total frames: {self.video_properties['total_frames']}")
        self.video_logger.info(f"INFO: Frame rate (FPS): {self.video_properties['fps']}")
        
        if self.video_properties["duration_seconds"]:
            duration_str = str(timedelta(seconds=int(self.video_properties["duration_seconds"])))
            self.video_logger.info(f"INFO: Video duration: {duration_str} (HH:MM:SS)")
        
        self.video_logger.info(f"INFO: Video ingestion complete for: {self.video_path.name}")
    
    def close_video(self) -> None:
        """Properly release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.video_logger.info("Video resources released")
    
    def __enter__(self):
        """Context manager entry."""
        self.open_video()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self.close_video()
        
        if exc_type is not None:
            self.video_logger.error(f"Video processing failed: {exc_val}")
        
        return False  # Don't suppress exceptions
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from video with error handling."""
        if self.cap is None:
            raise VideoProcessingError("Video not opened. Call open_video() first.")
        
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        if frame is None:
            self.video_logger.warning("Received None frame from video")
            return False, None
        
        # Validate frame dimensions
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            self.video_logger.warning("Received empty frame")
            return False, None
        
        return True, frame
    
    def get_frame_generator(self) -> Generator[Tuple[int, np.ndarray, datetime], None, None]:
        """Generator that yields frames with frame number and timestamp."""
        if self.cap is None:
            raise VideoProcessingError("Video not opened. Call open_video() first.")
        
        frame_number = 0
        start_time = datetime.now()
        
        with PerformanceTimer("Video frame generation"):
            while True:
                ret, frame = self.read_frame()
                
                if not ret or frame is None:
                    break
                
                # Calculate timestamp based on frame number and FPS
                if self.video_properties["fps"] > 0:
                    frame_time_offset = timedelta(
                        seconds=frame_number / self.video_properties["fps"]
                    )
                    frame_timestamp = start_time + frame_time_offset
                else:
                    frame_timestamp = start_time
                
                yield frame_number, frame, frame_timestamp
                frame_number += 1
                
                # Log progress periodically
                if frame_number % 300 == 0:  # Every 10 seconds at 30fps
                    progress = (frame_number / self.video_properties["total_frames"]) * 100
                    self.video_logger.info(f"Processing progress: {progress:.1f}% ({frame_number}/{self.video_properties['total_frames']} frames)")
    
    def get_current_frame_number(self) -> int:
        """Get current frame position."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """Seek to specific frame number."""
        if self.cap is None:
            raise VideoProcessingError("Video not opened")
        
        if frame_number < 0 or frame_number >= self.video_properties["total_frames"]:
            self.video_logger.warning(f"Frame number {frame_number} out of range")
            return False
        
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if success:
            self.video_logger.debug(f"Seeked to frame {frame_number}")
        else:
            self.video_logger.error(f"Failed to seek to frame {frame_number}")
        
        return success
    
    def seek_to_timestamp(self, timestamp_seconds: float) -> bool:
        """Seek to specific timestamp in seconds."""
        if self.video_properties["fps"] <= 0:
            self.video_logger.error("Cannot seek by timestamp: invalid FPS")
            return False
        
        frame_number = int(timestamp_seconds * self.video_properties["fps"])
        return self.seek_to_frame(frame_number)
    
    def extract_frame_at_timestamp(self, timestamp_seconds: float) -> Optional[np.ndarray]:
        """Extract a single frame at specified timestamp."""
        if not self.seek_to_timestamp(timestamp_seconds):
            return None
        
        ret, frame = self.read_frame()
        return frame if ret else None
    
    def save_frame(self, frame: np.ndarray, output_path: str, 
                   timestamp: Optional[datetime] = None) -> bool:
        """Save a frame to disk with optional timestamp overlay."""
        try:
            frame_to_save = frame.copy()
            
            # Add timestamp overlay if provided
            if timestamp and settings.save_debug_frames:
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                cv2.putText(
                    frame_to_save, timestamp_str,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            success = cv2.imwrite(output_path, frame_to_save)
            
            if success:
                self.video_logger.debug(f"Frame saved to {output_path}")
            else:
                self.video_logger.error(f"Failed to save frame to {output_path}")
            
            return success
            
        except Exception as e:
            self.video_logger.error(f"Error saving frame: {e}")
            return False
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get comprehensive video information."""
        info = {
            "file_path": str(self.video_path),
            "file_size_mb": self.video_path.stat().st_size / (1024 * 1024) if self.video_path.exists() else 0,
            "properties": self.video_properties.copy(),
            "is_opened": self.cap is not None and self.cap.isOpened() if self.cap else False
        }
        
        # Add calculated metrics
        if self.video_properties.get("fps", 0) > 0 and self.video_properties.get("total_frames", 0) > 0:
            info["estimated_processing_time"] = (
                self.video_properties["total_frames"] / settings.video_processing_fps
            )
        
        return info
    
    def validate_video_quality(self) -> Dict[str, Any]:
        """Validate video quality and detect potential issues."""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        if not self.video_properties:
            validation_results["errors"].append("No video properties available")
            validation_results["is_valid"] = False
            return validation_results
        
        # Check resolution
        width, height = self.video_properties["width"], self.video_properties["height"]
        if width < 640 or height < 480:
            validation_results["warnings"].append(
                f"Low resolution: {width}x{height}. Recommend >= 720p for better detection accuracy"
            )
        
        # Check FPS
        fps = self.video_properties["fps"]
        if fps < 15:
            validation_results["warnings"].append(
                f"Low frame rate: {fps} FPS. May affect tracking accuracy"
            )
        elif fps > 60:
            validation_results["recommendations"].append(
                f"High frame rate: {fps} FPS. Consider downsampling for performance"
            )
        
        # Check duration
        duration = self.video_properties.get("duration_seconds", 0)
        if duration > 3600:  # 1 hour
            validation_results["recommendations"].append(
                "Long video duration. Consider processing in segments for memory efficiency"
            )
        
        # Check aspect ratio for overhead camera assumption
        aspect_ratio = width / height if height > 0 else 0
        if not (1.2 <= aspect_ratio <= 2.0):
            validation_results["warnings"].append(
                f"Unusual aspect ratio: {aspect_ratio:.2f}. Banking overhead cameras typically have 16:9 or 4:3 ratio"
            )
        
        return validation_results


def process_video_file(video_path: str, frame_callback=None) -> Dict[str, Any]:
    """Process a video file with optional frame callback function."""
    results = {
        "success": False,
        "total_frames_processed": 0,
        "processing_time": 0,
        "video_info": {},
        "errors": []
    }
    
    start_time = datetime.now()
    
    try:
        with VideoProcessor(video_path) as processor:
            results["video_info"] = processor.get_video_info()
            
            # Validate video quality
            validation = processor.validate_video_quality()
            if not validation["is_valid"]:
                results["errors"].extend(validation["errors"])
                return results
            
            # Process frames
            for frame_number, frame, timestamp in processor.get_frame_generator():
                if frame_callback:
                    try:
                        frame_callback(frame_number, frame, timestamp)
                    except Exception as e:
                        logger.error(f"Frame callback error at frame {frame_number}: {e}")
                        results["errors"].append(f"Frame {frame_number}: {str(e)}")
                
                results["total_frames_processed"] = frame_number + 1
            
            results["success"] = True
            
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        results["errors"].append(str(e))
    
    finally:
        end_time = datetime.now()
        results["processing_time"] = (end_time - start_time).total_seconds()
    
    return results