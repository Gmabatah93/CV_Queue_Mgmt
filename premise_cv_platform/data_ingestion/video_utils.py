"""
Video utility functions for PREMISE CV Platform.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from premise_cv_platform.config.settings import settings


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """Resize frame to target size with optional aspect ratio maintenance."""
    height, width = frame.shape[:2]
    target_width, target_height = target_size
    
    if maintain_aspect:
        # Calculate scaling factor
        scale_x = target_width / width
        scale_y = target_height / height
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded frame if needed
        if new_width != target_width or new_height != target_height:
            # Create black background
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate padding offsets
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place resized frame in center
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            
            return padded
        else:
            return resized
    else:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


def crop_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop frame to bounding box coordinates (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    
    # Clamp coordinates to frame boundaries
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(x1, min(x2, width))
    y2 = max(y1, min(y2, height))
    
    return frame[y1:y2, x1:x2]


def enhance_frame(frame: np.ndarray, brightness: float = 0, 
                 contrast: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Enhance frame with brightness, contrast, and gamma correction."""
    enhanced = frame.astype(np.float32)
    
    # Apply brightness
    enhanced += brightness
    
    # Apply contrast
    enhanced *= contrast
    
    # Apply gamma correction
    if gamma != 1.0:
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
    
    # Clip values to valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def detect_motion(frame1: np.ndarray, frame2: np.ndarray, 
                 threshold: int = 25) -> Tuple[np.ndarray, float]:
    """Detect motion between two frames."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate motion percentage
    motion_pixels = np.count_nonzero(motion_mask)
    total_pixels = motion_mask.size
    motion_percentage = (motion_pixels / total_pixels) * 100
    
    return motion_mask, motion_percentage


def draw_detection_info(frame: np.ndarray, detections: List[Dict], 
                       zone_polygons: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Draw detection information on frame for visualization."""
    frame_with_info = frame.copy()
    
    # Draw zone polygons if provided
    if zone_polygons:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        for i, polygon in enumerate(zone_polygons):
            color = colors[i % len(colors)]
            cv2.polylines(frame_with_info, [polygon.astype(np.int32)], True, color, 2)
    
    # Draw detections
    for detection in detections:
        # Extract detection info
        bbox = detection.get('bbox', [0, 0, 0, 0])
        person_id = detection.get('person_id', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw bounding box
        cv2.rectangle(frame_with_info, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw person ID and confidence
        label = f"ID: {person_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for text
        cv2.rectangle(frame_with_info, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     (0, 255, 0), -1)
        
        # Text
        cv2.putText(frame_with_info, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame_with_info, (center_x, center_y), 3, (255, 0, 0), -1)
    
    return frame_with_info


def calculate_frame_metrics(frame: np.ndarray) -> Dict[str, float]:
    """Calculate various metrics for frame quality assessment."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    metrics = {
        'brightness': np.mean(gray),
        'std_dev': np.std(gray),
        'contrast': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0,
    }
    
    # Calculate Laplacian variance (measure of blur)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['sharpness'] = np.var(laplacian)
    
    return metrics


def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int],
                       codec: str = 'mp4v') -> cv2.VideoWriter:
    """Create OpenCV VideoWriter with proper codec settings."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        frame_size
    )
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    
    logger.info(f"Video writer created: {output_path} ({frame_size[0]}x{frame_size[1]} @ {fps} fps)")
    return writer


def extract_video_frames(video_path: str, output_dir: str, 
                        interval: int = 1, max_frames: Optional[int] = None) -> int:
    """Extract frames from video at specified interval."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at interval
            if frame_count % interval == 0:
                frame_filename = output_path / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
    
    finally:
        cap.release()
    
    logger.info(f"Extracted {extracted_count} frames from {video_path}")
    return extracted_count


def validate_video_file(video_path: str) -> Dict[str, Any]:
    """Validate video file and return detailed information."""
    path = Path(video_path)
    validation = {
        'valid': False,
        'exists': path.exists(),
        'readable': False,
        'properties': {},
        'errors': []
    }
    
    if not validation['exists']:
        validation['errors'].append(f"File does not exist: {video_path}")
        return validation
    
    try:
        cap = cv2.VideoCapture(video_path)
        validation['readable'] = cap.isOpened()
        
        if validation['readable']:
            validation['properties'] = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': None
            }
            
            if validation['properties']['fps'] > 0:
                validation['properties']['duration'] = (
                    validation['properties']['frame_count'] / validation['properties']['fps']
                )
            
            # Check if we can read at least one frame
            ret, frame = cap.read()
            if ret and frame is not None:
                validation['valid'] = True
                validation['properties']['frame_shape'] = frame.shape
            else:
                validation['errors'].append("Cannot read frames from video")
        else:
            validation['errors'].append("Cannot open video file")
        
        cap.release()
        
    except Exception as e:
        validation['errors'].append(f"Error validating video: {str(e)}")
    
    return validation


def create_video_thumbnail(video_path: str, output_path: str, 
                          timestamp: float = 1.0) -> bool:
    """Create thumbnail image from video at specified timestamp."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Seek to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Resize to thumbnail size
            thumbnail = resize_frame(frame, (320, 240), maintain_aspect=True)
            success = cv2.imwrite(output_path, thumbnail)
            
            if success:
                logger.info(f"Thumbnail created: {output_path}")
            return success
        
        return False
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")
        return False


def get_video_codec_info(video_path: str) -> Dict[str, Any]:
    """Get detailed codec information from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    codec_info = {
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'fourcc_str': '',
        'backend': cap.getBackendName() if hasattr(cap, 'getBackendName') else 'unknown'
    }
    
    # Convert fourcc to string
    fourcc_bytes = [
        codec_info['fourcc'] & 0xff,
        (codec_info['fourcc'] >> 8) & 0xff,
        (codec_info['fourcc'] >> 16) & 0xff,
        (codec_info['fourcc'] >> 24) & 0xff
    ]
    codec_info['fourcc_str'] = ''.join([chr(b) for b in fourcc_bytes])
    
    cap.release()
    return codec_info