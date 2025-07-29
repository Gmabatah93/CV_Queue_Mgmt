"""
Unit tests for file handling utilities.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import io

from premise_cv_platform.utils.file_handler import TempFileManager


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content
        self._buffer = io.BytesIO(content)

    def getbuffer(self):
        """Return buffer for file content."""
        return self._content


class TestTempFileManager:
    """Test cases for TempFileManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TempFileManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test TempFileManager initialization."""
        assert self.manager.temp_dir == Path(self.temp_dir)
        assert self.manager.temp_dir.exists()

    def test_save_uploaded_file_success(self):
        """Test successful file upload save."""
        # Create mock uploaded file
        test_content = b"test video content"
        uploaded_file = MockUploadedFile("test_video.mp4", test_content)

        # Mock validation to return valid
        with patch.object(self.manager, "validate_uploaded_file") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}

            # Save file
            result_path = self.manager.save_uploaded_file(uploaded_file)

            # Verify file was saved
            assert Path(result_path).exists()
            assert Path(result_path).read_bytes() == test_content
            assert "test_video.mp4" in result_path

    def test_save_uploaded_file_invalid(self):
        """Test file upload save with invalid file."""
        test_content = b"invalid content"
        uploaded_file = MockUploadedFile("invalid.txt", test_content)

        # Mock validation to return invalid 
        with patch.object(self.manager, "validate_uploaded_file") as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "exists": True,
                "readable": False,
                "properties": {},
                "errors": ["Invalid file format"],
            }

            # Should raise ValueError
            with pytest.raises(ValueError, match="Invalid video file"):
                self.manager.save_uploaded_file(uploaded_file)

    def test_save_uploaded_file_none(self):
        """Test file upload save with None input."""
        with pytest.raises(ValueError, match="No file provided"):
            self.manager.save_uploaded_file(None)

    def test_cleanup_temp_file(self):
        """Test temporary file cleanup."""
        # Create a test file
        test_file = self.manager.temp_dir / "test_file.mp4"
        test_file.write_text("test content")
        assert test_file.exists()

        # Clean up file
        self.manager.cleanup_temp_file(str(test_file))

        # Verify file is removed
        assert not test_file.exists()

    def test_cleanup_temp_file_not_exists(self):
        """Test cleanup of non-existent file."""
        non_existent = self.manager.temp_dir / "non_existent.mp4"

        # Should not raise error
        self.manager.cleanup_temp_file(str(non_existent))

    def test_cleanup_temp_file_outside_temp_dir(self):
        """Test cleanup of file outside temp directory."""
        external_file = Path("/tmp/external_file.mp4")

        # Should not cleanup files outside temp directory
        self.manager.cleanup_temp_file(str(external_file))

    @patch("premise_cv_platform.utils.file_handler.validate_video_file")
    def test_validate_uploaded_file_valid(self, mock_validate):
        """Test file validation for valid video."""
        # Create test file
        test_file = self.manager.temp_dir / "test.mp4"
        test_file.write_bytes(b"test video content")

        # Mock video validation
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "readable": True,
            "properties": {"width": 1920, "height": 1080},
            "errors": [],
        }

        result = self.manager.validate_uploaded_file(str(test_file))

        assert result["valid"] is True
        assert "file_size_mb" in result["properties"]

    @patch("premise_cv_platform.utils.file_handler.validate_video_file")
    def test_validate_uploaded_file_large_file(self, mock_validate):
        """Test file validation for oversized file."""
        # Create test file
        test_file = self.manager.temp_dir / "large.mp4"
        # Create file larger than 100MB (simulated)
        test_file.write_bytes(b"x" * 1000)

        # Mock video validation
        mock_validate.return_value = {
            "valid": True,
            "exists": True,
            "readable": True,
            "properties": {},
            "errors": [],
        }

        # Mock file size to be large
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 200 * 1024 * 1024  # 200MB

            result = self.manager.validate_uploaded_file(str(test_file))

            assert result["valid"] is False
            assert any("exceeds limit" in error for error in result["errors"])

    def test_validate_uploaded_file_unsupported_format(self):
        """Test file validation for unsupported format."""
        # Create test file with unsupported extension
        test_file = self.manager.temp_dir / "test.txt"
        test_file.write_text("not a video")

        result = self.manager.validate_uploaded_file(str(test_file))

        assert result["valid"] is False
        assert any("Unsupported file format" in error for error in result["errors"])

    def test_cleanup_old_temp_files(self):
        """Test cleanup of old temporary files."""
        # Create some old files
        old_file1 = self.manager.temp_dir / "video_upload_old1.mp4"
        old_file2 = self.manager.temp_dir / "video_upload_old2.mp4"
        new_file = self.manager.temp_dir / "video_upload_new.mp4"

        # Write files
        old_file1.write_text("old1")
        old_file2.write_text("old2")
        new_file.write_text("new")

        # Mock file modification times
        import time

        current_time = time.time()
        old_time = current_time - (25 * 3600)  # 25 hours ago

        with patch("pathlib.Path.stat") as mock_stat:
            # Configure mock to return old times for old files, new time for new file
            def stat_side_effect():
                mock_stat_result = Mock()
                # Default to old time, will be overridden below
                mock_stat_result.st_mtime = old_time
                return mock_stat_result

            mock_stat.side_effect = stat_side_effect

            # Mock individual file stat calls
            old_file1.stat = lambda: Mock(st_mtime=old_time)
            old_file2.stat = lambda: Mock(st_mtime=old_time)
            new_file.stat = lambda: Mock(st_mtime=current_time)

            # Clean up files older than 24 hours
            cleaned_count = self.manager.cleanup_old_temp_files(max_age_hours=24)

            # Should have cleaned 2 old files
            assert cleaned_count == 2

    def test_get_temp_file_stats(self):
        """Test getting temporary file statistics."""
        # Create some test files
        file1 = self.manager.temp_dir / "video_upload_test1.mp4"
        file2 = self.manager.temp_dir / "video_upload_test2.mp4"
        other_file = self.manager.temp_dir / "other_file.txt"

        file1.write_text("content1")
        file2.write_text("content2")
        other_file.write_text("other")

        stats = self.manager.get_temp_file_stats()

        assert stats["count"] == 2  # Only video_upload_* files
        assert stats["total_size_mb"] > 0
        assert stats["temp_dir"] == str(self.manager.temp_dir)
        assert len(stats["files"]) == 2

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test normal filename
        assert TempFileManager._sanitize_filename("test_video.mp4") == "test_video.mp4"

        # Test filename with special characters
        assert (
            TempFileManager._sanitize_filename("test video!@#$%^&*().mp4")
            == "testvideo.mp4"
        )

        # Test empty filename
        assert TempFileManager._sanitize_filename("") == "uploaded_video.mp4"

        # Test filename starting with dot
        assert TempFileManager._sanitize_filename(".hidden") == "uploaded_video.mp4"