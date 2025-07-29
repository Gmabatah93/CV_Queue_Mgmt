"""
Unit tests for async processing utilities.
"""

import pytest
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from premise_cv_platform.utils.async_processor import (
    ProcessingStatus,
    UploadedVideoProcessor,
    ProcessingManager,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        """Return buffer for file content."""
        return self._content


class TestProcessingStatus:
    """Test cases for ProcessingStatus dataclass."""

    def test_processing_status_init(self):
        """Test ProcessingStatus initialization."""
        status = ProcessingStatus(status="idle", message="Ready")

        assert status.status == "idle"
        assert status.message == "Ready"
        assert status.progress == 0.0
        assert status.video_file is None
        assert status.results is None
        assert status.error is None

    def test_processing_status_to_dict(self):
        """Test ProcessingStatus to_dict conversion."""
        status = ProcessingStatus(
            status="processing", message="Processing...", progress=50.0
        )

        result = status.to_dict()

        assert result["status"] == "processing"
        assert result["message"] == "Processing..."
        assert result["progress"] == 50.0
        assert result["video_file"] is None


class TestUploadedVideoProcessor:
    """Test cases for UploadedVideoProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = UploadedVideoProcessor(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test UploadedVideoProcessor initialization."""
        assert self.processor.temp_dir == self.temp_dir
        assert self.processor.current_status.status == "idle"
        assert self.processor.current_status.message == "Ready"

    @patch("main.PremiseCVPipeline")
    def test_process_uploaded_video_success(self, mock_pipeline_class):
        """Test successful video processing."""
        # Create mock uploaded file
        uploaded_file = MockUploadedFile("test.mp4", b"test video content")

        # Mock file manager
        with patch.object(
            self.processor.file_manager, "save_uploaded_file"
        ) as mock_save:
            mock_save.return_value = "/tmp/test_video.mp4"

            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.process_video.return_value = {
                "success": True,
                "processing_stats": {"total_frames": 100, "unique_persons": 5},
                "events_generated": {"line_events": 3, "teller_interactions": 2},
            }

            # Mock cleanup
            with patch.object(
                self.processor.file_manager, "cleanup_temp_file"
            ) as mock_cleanup:

                # Process video
                result = self.processor.process_uploaded_video(uploaded_file)

                # Verify success
                assert result.status == "completed"
                assert result.results is not None
                mock_save.assert_called_once()
                mock_cleanup.assert_called_once()

    def test_process_uploaded_video_file_save_error(self):
        """Test video processing with file save error."""
        uploaded_file = MockUploadedFile("test.mp4", b"test video content")

        # Mock file manager to raise error
        with patch.object(
            self.processor.file_manager, "save_uploaded_file"
        ) as mock_save:
            mock_save.side_effect = IOError("Failed to save file")

            # Process video should handle error
            result = self.processor.process_uploaded_video(uploaded_file)

            assert result.status == "error"
            assert "Failed to save file" in result.error

    @patch("main.PremiseCVPipeline")
    def test_process_uploaded_video_processing_error(self, mock_pipeline_class):
        """Test video processing with pipeline error."""
        uploaded_file = MockUploadedFile("test.mp4", b"test video content")

        # Mock file save success
        with patch.object(
            self.processor.file_manager, "save_uploaded_file"
        ) as mock_save:
            mock_save.return_value = "/tmp/test_video.mp4"

            # Mock pipeline to raise error
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.process_video.side_effect = Exception("Processing failed")

            # Mock cleanup
            with patch.object(
                self.processor.file_manager, "cleanup_temp_file"
            ) as mock_cleanup:

                # Process video
                result = self.processor.process_uploaded_video(uploaded_file)

                # Verify error handling
                assert result.status == "error"
                assert "Processing failed" in result.error
                mock_cleanup.assert_called_once()

    def test_progress_callback(self):
        """Test progress callback functionality."""
        uploaded_file = MockUploadedFile("test.mp4", b"test video content")
        callback_calls = []

        def progress_callback(status):
            callback_calls.append(status.status)

        # Mock file save
        with patch.object(
            self.processor.file_manager, "save_uploaded_file"
        ) as mock_save:
            mock_save.return_value = "/tmp/test_video.mp4"

            # Mock pipeline success
            with patch("main.PremiseCVPipeline"):
                with patch.object(
                    self.processor.file_manager, "cleanup_temp_file"
                ):

                    # Process with callback
                    self.processor.process_uploaded_video(
                        uploaded_file, progress_callback=progress_callback
                    )

                    # Should have received callbacks
                    assert len(callback_calls) > 0
                    assert "uploading" in callback_calls

    def test_get_current_status(self):
        """Test getting current processing status."""
        status = self.processor.get_current_status()

        assert isinstance(status, ProcessingStatus)
        assert status.status == "idle"

    def test_is_processing(self):
        """Test processing state check."""
        # Initially not processing
        assert not self.processor.is_processing()

        # Set processing state
        self.processor.current_status.status = "processing"
        assert self.processor.is_processing()

    def test_cancel_processing(self):
        """Test processing cancellation."""
        # Set processing state
        self.processor.current_status.status = "processing"

        # Cancel processing
        self.processor.cancel_processing()

        assert self.processor.stop_processing is True

    def test_cleanup_temp_files(self):
        """Test temp file cleanup."""
        with patch.object(
            self.processor.file_manager, "cleanup_old_temp_files"
        ) as mock_cleanup:
            mock_cleanup.return_value = 3

            result = self.processor.cleanup_temp_files()

            assert result == 3
            mock_cleanup.assert_called_once_with(24)

    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        # Mock temp file stats
        with patch.object(
            self.processor.file_manager, "get_temp_file_stats"
        ) as mock_stats:
            mock_stats.return_value = {"count": 2, "total_size_mb": 150.5}

            stats = self.processor.get_processing_stats()

            assert "current_status" in stats
            assert "temp_files" in stats
            assert "is_processing" in stats
            assert stats["temp_files"]["count"] == 2


class TestProcessingManager:
    """Test cases for ProcessingManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ProcessingManager()

    def test_init(self):
        """Test ProcessingManager initialization."""
        assert isinstance(self.manager.processors, dict)
        assert isinstance(self.manager.active_sessions, dict)
        assert len(self.manager.processors) == 0
        assert len(self.manager.active_sessions) == 0

    def test_get_processor_new_session(self):
        """Test getting processor for new session."""
        session_id = "test_session_123"

        processor = self.manager.get_processor(session_id)

        assert isinstance(processor, UploadedVideoProcessor)
        assert session_id in self.manager.active_sessions
        assert len(self.manager.processors) == 1

    def test_get_processor_existing_session(self):
        """Test getting processor for existing session."""
        session_id = "test_session_123"

        # Get processor twice
        processor1 = self.manager.get_processor(session_id)
        processor2 = self.manager.get_processor(session_id)

        # Should return same processor
        assert processor1 is processor2
        assert len(self.manager.processors) == 1

    def test_cleanup_session(self):
        """Test session cleanup."""
        session_id = "test_session_123"

        # Create processor for session
        processor = self.manager.get_processor(session_id)

        # Mock processor methods
        with patch.object(processor, "cancel_processing") as mock_cancel:
            with patch.object(processor, "cleanup_temp_files") as mock_cleanup:

                # Cleanup session
                self.manager.cleanup_session(session_id)

                # Verify cleanup was called
                mock_cancel.assert_called_once()
                mock_cleanup.assert_called_once()

                # Verify session was removed
                assert session_id not in self.manager.active_sessions
                assert len(self.manager.processors) == 0

    def test_cleanup_nonexistent_session(self):
        """Test cleanup of non-existent session."""
        # Should not raise error
        self.manager.cleanup_session("nonexistent_session")

    def test_get_all_stats(self):
        """Test getting statistics for all processors."""
        # Create some processors
        session1 = "session1"
        session2 = "session2"

        processor1 = self.manager.get_processor(session1)
        processor2 = self.manager.get_processor(session2)

        # Mock processor stats
        with patch.object(processor1, "get_processing_stats") as mock_stats1:
            with patch.object(processor2, "get_processing_stats") as mock_stats2:
                mock_stats1.return_value = {"status": "idle"}
                mock_stats2.return_value = {"status": "processing"}

                stats = self.manager.get_all_stats()

                assert stats["total_processors"] == 2
                assert stats["active_sessions"] == 2
                assert len(stats["processors"]) == 2