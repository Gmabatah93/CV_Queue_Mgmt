"""
Test Module 01: Video Processing

Tests the core video processing functionality including:
- VideoProcessor class (process_video.py)
- Video utility functions (video_utils.py)
- Frame manipulation and enhancement
- Video file validation and quality checks
"""

import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from premise_cv_platform.data_ingestion.process_video import (
    VideoProcessor, 
    VideoProcessingError,
    process_video_file
)
from premise_cv_platform.data_ingestion.video_utils import (
    resize_frame,
    crop_frame,
    enhance_frame,
    detect_motion,
    draw_detection_info,
    calculate_frame_metrics,
    create_video_writer,
    extract_video_frames,
    validate_video_file,
    create_video_thumbnail,
    get_video_codec_info
)


class TestVideoProcessor:
    """Test suite for VideoProcessor class."""
    
    @pytest.fixture
    def sample_video_path(self, tmp_path):
        """Create a mock video path for testing."""
        return str(tmp_path / "sample_video.mp4")
    
    @pytest.fixture
    def mock_video_capture(self):
        """Mock OpenCV VideoCapture for testing."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        return mock_cap
    
    def test_video_processor_initialization(self, sample_video_path):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor(sample_video_path)
        assert processor.video_path == Path(sample_video_path)
        assert processor.cap is None
        assert processor.video_properties == {}
    
    def test_validate_video_path_file_not_found(self, tmp_path):
        """Test validation when video file doesn't exist."""
        non_existent_path = tmp_path / "non_existent.mp4"
        processor = VideoProcessor(str(non_existent_path))
        
        with pytest.raises(VideoProcessingError, match="Video file not found"):
            processor._validate_video_path()
    
    def test_validate_video_path_invalid_extension(self, tmp_path):
        """Test validation with invalid file extension."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.touch()
        processor = VideoProcessor(str(invalid_file))
        
        # Should not raise error but log warning
        processor._validate_video_path()
    
    @patch('cv2.VideoCapture')
    def test_open_video_success(self, mock_cv2_videocapture, sample_video_path, mock_video_capture):
        """Test successful video opening."""
        # Create a temporary file
        video_file = Path(sample_video_path)
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.touch()
        
        mock_cv2_videocapture.return_value = mock_video_capture
        
        processor = VideoProcessor(sample_video_path)
        processor.open_video()
        
        assert processor.cap == mock_video_capture
        assert processor.video_properties["width"] == 1920
        assert processor.video_properties["height"] == 1080
        assert processor.video_properties["fps"] == 30.0
        assert processor.video_properties["total_frames"] == 300
        assert processor.video_properties["duration_seconds"] == 10.0  # 300/30
    
    @patch('cv2.VideoCapture')
    def test_open_video_failure(self, mock_cv2_videocapture, sample_video_path):
        """Test video opening failure."""
        # Create a temporary file
        video_file = Path(sample_video_path)
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.touch()
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_videocapture.return_value = mock_cap
        
        processor = VideoProcessor(sample_video_path)
        
        with pytest.raises(VideoProcessingError, match="Cannot open video"):
            processor.open_video()
    
    def test_close_video(self, sample_video_path, mock_video_capture):
        """Test video resource cleanup."""
        processor = VideoProcessor(sample_video_path)
        processor.cap = mock_video_capture
        
        processor.close_video()
        
        assert processor.cap is None
        mock_video_capture.release.assert_called_once()
    
    def test_context_manager(self, sample_video_path, mock_video_capture):
        """Test VideoProcessor as context manager."""
        # Create a temporary file
        video_file = Path(sample_video_path)
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.touch()
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture):
            with VideoProcessor(sample_video_path) as processor:
                assert processor.cap == mock_video_capture
            
            # Should be closed after context exit
            mock_video_capture.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_read_frame(self, mock_cv2_videocapture, sample_video_path, mock_video_capture):
        """Test frame reading functionality."""
        # Create a temporary file
        video_file = Path(sample_video_path)
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.touch()
        
        mock_cv2_videocapture.return_value = mock_video_capture
        
        # Mock frame data
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_video_capture.read.return_value = (True, mock_frame)
        
        processor = VideoProcessor(sample_video_path)
        processor.open_video()
        
        success, frame = processor.read_frame()
        
        assert success is True
        assert frame is not None
        assert frame.shape == (1080, 1920, 3)
        mock_video_capture.read.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_get_frame_generator(self, mock_cv2_videocapture, sample_video_path, mock_video_capture):
        """Test frame generator functionality."""
        # Create a temporary file
        video_file = Path(sample_video_path)
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.touch()
        
        mock_cv2_videocapture.return_value = mock_video_capture
        
        # Mock frame data
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_video_capture.read.return_value = (True, mock_frame)
        
        processor = VideoProcessor(sample_video_path)
        processor.open_video()
        
        generator = processor.get_frame_generator()
        frames = list(generator)
        
        assert len(frames) > 0
        for frame_number, frame, timestamp in frames:
            assert isinstance(frame_number, int)
            assert isinstance(frame, np.ndarray)
            assert isinstance(timestamp, datetime)
    
    def test_get_video_info(self, sample_video_path, mock_video_capture):
        """Test video information retrieval."""
        processor = VideoProcessor(sample_video_path)
        processor.cap = mock_video_capture
        processor.video_properties = {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "total_frames": 300,
            "duration_seconds": 10.0
        }
        
        info = processor.get_video_info()
        
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["fps"] == 30.0
        assert info["total_frames"] == 300
        assert info["duration_seconds"] == 10.0


class TestVideoUtils:
    """Test suite for video utility functions."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_resize_frame_maintain_aspect(self, sample_frame):
        """Test frame resizing with aspect ratio maintenance."""
        target_size = (320, 240)
        resized = resize_frame(sample_frame, target_size, maintain_aspect=True)
        
        assert resized.shape[:2] == target_size[::-1]  # OpenCV uses (height, width)
        assert resized.shape[2] == 3  # 3 channels
    
    def test_resize_frame_no_aspect(self, sample_frame):
        """Test frame resizing without aspect ratio maintenance."""
        target_size = (320, 240)
        resized = resize_frame(sample_frame, target_size, maintain_aspect=False)
        
        assert resized.shape[:2] == target_size[::-1]
        assert resized.shape[2] == 3
    
    def test_crop_frame(self, sample_frame):
        """Test frame cropping functionality."""
        bbox = (100, 100, 300, 300)  # (x1, y1, x2, y2)
        cropped = crop_frame(sample_frame, bbox)
        
        expected_height = bbox[3] - bbox[1]  # y2 - y1
        expected_width = bbox[2] - bbox[0]   # x2 - x1
        
        assert cropped.shape[:2] == (expected_height, expected_width)
        assert cropped.shape[2] == 3
    
    def test_crop_frame_boundary_clamping(self, sample_frame):
        """Test frame cropping with boundary clamping."""
        # Test coordinates outside frame boundaries
        bbox = (-50, -50, 1000, 1000)
        cropped = crop_frame(sample_frame, bbox)
        
        # Should clamp to frame boundaries
        assert cropped.shape[:2] == (480, 640)
        assert cropped.shape[2] == 3
    
    def test_enhance_frame(self, sample_frame):
        """Test frame enhancement functionality."""
        enhanced = enhance_frame(sample_frame, brightness=10, contrast=1.2, gamma=1.1)
        
        assert enhanced.shape == sample_frame.shape
        assert enhanced.dtype == np.uint8
    
    def test_detect_motion(self, sample_frame):
        """Test motion detection between frames."""
        # Create a slightly different frame
        frame2 = sample_frame.copy()
        frame2[100:200, 100:200] = 255  # Add white rectangle
        
        motion_mask, motion_percentage = detect_motion(sample_frame, frame2)
        
        assert motion_mask.shape[:2] == sample_frame.shape[:2]
        assert 0 <= motion_percentage <= 100
        assert motion_percentage > 0  # Should detect some motion
    
    def test_draw_detection_info(self, sample_frame):
        """Test drawing detection information on frame."""
        detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.95, "class": "person"},
            {"bbox": [300, 300, 400, 400], "confidence": 0.87, "class": "object"}
        ]
        
        annotated_frame = draw_detection_info(sample_frame, detections)
        
        assert annotated_frame.shape == sample_frame.shape
        assert annotated_frame.dtype == np.uint8
    
    def test_calculate_frame_metrics(self, sample_frame):
        """Test frame metrics calculation."""
        metrics = calculate_frame_metrics(sample_frame)
        
        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "sharpness" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_create_video_writer(self, tmp_path):
        """Test video writer creation."""
        output_path = str(tmp_path / "output.mp4")
        fps = 30.0
        frame_size = (640, 480)
        
        writer = create_video_writer(output_path, fps, frame_size)
        
        assert writer is not None
        writer.release()
    
    def test_validate_video_file(self, tmp_path):
        """Test video file validation."""
        # Create a mock video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        # Mock cv2.VideoCapture for validation
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 300
            }.get(prop, 0)
            mock_cv2.return_value = mock_cap
            
            result = validate_video_file(str(video_path))
            
            assert result["is_valid"] is True
            assert result["width"] == 1920
            assert result["height"] == 1080
            assert result["fps"] == 30.0
            assert result["total_frames"] == 300


class TestVideoProcessingIntegration:
    """Integration tests for video processing pipeline."""
    
    @pytest.fixture
    def sample_video_file(self, tmp_path):
        """Create a mock video file for integration testing."""
        video_path = tmp_path / "integration_test.mp4"
        video_path.touch()
        return str(video_path)
    
    @patch('cv2.VideoCapture')
    def test_process_video_file_integration(self, mock_cv2_videocapture, sample_video_file):
        """Test the complete video processing workflow."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90,  # 3 seconds at 30fps
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'mp4v')
        }.get(prop, 0)
        
        # Mock frame reading
        mock_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        
        mock_cv2_videocapture.return_value = mock_cap
        
        # Test frame callback
        processed_frames = []
        def frame_callback(frame_number, frame, timestamp):
            processed_frames.append((frame_number, frame, timestamp))
        
        result = process_video_file(sample_video_file, frame_callback)
        
        assert result["total_frames"] == 90
        assert result["duration_seconds"] == 3.0
        assert result["fps"] == 30.0
        assert len(processed_frames) > 0
        
        # Verify callback was called with correct data
        for frame_number, frame, timestamp in processed_frames:
            assert isinstance(frame_number, int)
            assert isinstance(frame, np.ndarray)
            assert isinstance(timestamp, datetime)
            assert frame.shape == (1080, 1920, 3)


class TestVideoProcessingErrorHandling:
    """Test error handling in video processing."""
    
    def test_video_processing_error_inheritance(self):
        """Test that VideoProcessingError inherits from Exception."""
        error = VideoProcessingError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    @patch('cv2.VideoCapture')
    def test_video_processor_error_handling(self, mock_cv2_videocapture, tmp_path):
        """Test error handling in VideoProcessor."""
        # Test with non-existent file
        non_existent_path = tmp_path / "non_existent.mp4"
        processor = VideoProcessor(str(non_existent_path))
        
        with pytest.raises(VideoProcessingError):
            processor.open_video()
        
        # Test with invalid video file
        invalid_file = tmp_path / "invalid.mp4"
        invalid_file.touch()
        
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2_videocapture.return_value = mock_cap
        
        processor = VideoProcessor(str(invalid_file))
        
        with pytest.raises(VideoProcessingError, match="Cannot open video"):
            processor.open_video()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 