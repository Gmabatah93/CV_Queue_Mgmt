"""
File handling utilities for uploaded video files in PREMISE CV Platform.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import streamlit as st

from premise_cv_platform.config.settings import settings
from premise_cv_platform.data_ingestion.video_utils import validate_video_file


class TempFileManager:
    """Handle temporary file lifecycle for uploaded videos."""

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize TempFileManager with optional custom temp directory."""
        self.temp_dir = Path(temp_dir) if temp_dir else Path("data/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TempFileManager initialized with temp_dir: {self.temp_dir}")

    def save_uploaded_file(
        self, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile
    ) -> str:
        """
        Save Streamlit uploaded file to temporary location on disk.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            str: Path to saved temporary file

        Raises:
            ValueError: If file validation fails
            IOError: If file save operation fails
        """
        if uploaded_file is None:
            raise ValueError("No file provided")

        # Generate unique temp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = self._sanitize_filename(uploaded_file.name)
        temp_filename = f"video_upload_{timestamp}_{safe_filename}"
        temp_path = self.temp_dir / temp_filename

        try:
            # Write uploaded file to disk
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            logger.info(f"Uploaded file saved to: {temp_path}")

            # Validate the saved file
            validation_result = self.validate_uploaded_file(str(temp_path))
            if not validation_result["valid"]:
                # Clean up invalid file
                self.cleanup_temp_file(str(temp_path))
                error_msg = (
                    f"Invalid video file: {', '.join(validation_result['errors'])}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"File validation successful for: {temp_path}")
            return str(temp_path)

        except Exception as e:
            # Clean up on any error
            if temp_path.exists():
                self.cleanup_temp_file(str(temp_path))
            logger.error(f"Error saving uploaded file: {e}")
            raise IOError(f"Failed to save uploaded file: {str(e)}")

    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary file after processing.

        Args:
            file_path: Path to temporary file to cleanup
        """
        try:
            temp_path = Path(file_path)
            if temp_path.exists() and temp_path.parent == self.temp_dir:
                temp_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
            else:
                logger.warning(
                    f"Temp file not found or not in temp directory: {file_path}"
                )
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {file_path}: {e}")

    def validate_uploaded_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate uploaded video file using existing video_utils validation.

        Args:
            file_path: Path to video file to validate

        Returns:
            Dict[str, Any]: Validation result with 'valid' boolean and details
        """
        try:
            # Use existing validation from video_utils
            validation = validate_video_file(file_path)

            # Additional checks for uploaded files
            file_path_obj = Path(file_path)
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)

            # Check file size (100MB limit for uploads)
            max_size_mb = getattr(settings, "max_upload_file_size_mb", 100)
            if file_size_mb > max_size_mb:
                validation["valid"] = False
                validation["errors"].append(
                    f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)"
                )

            # Check file extension
            allowed_extensions = {".mov", ".mp4", ".mpg", ".avi", ".mkv"}
            if file_path_obj.suffix.lower() not in allowed_extensions:
                validation["valid"] = False
                error_msg = f"Unsupported file format: {file_path_obj.suffix}"
                validation["errors"].append(error_msg)

            # Add file size to properties
            if "properties" not in validation:
                validation["properties"] = {}
            validation["properties"]["file_size_mb"] = file_size_mb

            logger.info(
                f"File validation completed for {file_path}: "
                f"valid={validation['valid']}"
            )
            return validation

        except Exception as e:
            logger.error(f"Error validating uploaded file {file_path}: {e}")
            return {
                "valid": False,
                "exists": False,
                "readable": False,
                "properties": {},
                "errors": [f"Validation error: {str(e)}"],
            }

    def cleanup_old_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            int: Number of files cleaned up
        """
        try:
            cleanup_count = 0
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600

            for temp_file in self.temp_dir.glob("video_upload_*"):
                file_age_seconds = current_time - temp_file.stat().st_mtime
                if file_age_seconds > max_age_seconds:
                    temp_file.unlink()
                    cleanup_count += 1
                    logger.info(f"Cleaned up old temp file: {temp_file}")

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old temporary files")

            return cleanup_count

        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
            return 0

    def get_temp_file_stats(self) -> Dict[str, Any]:
        """
        Get statistics about temporary files.

        Returns:
            Dict[str, Any]: Statistics about temp files
        """
        try:
            temp_files = list(self.temp_dir.glob("video_upload_*"))
            total_size = sum(f.stat().st_size for f in temp_files)

            return {
                "count": len(temp_files),
                "total_size_mb": total_size / (1024 * 1024),
                "temp_dir": str(self.temp_dir),
                "files": [str(f.name) for f in temp_files],
            }
        except Exception as e:
            logger.error(f"Error getting temp file stats: {e}")
            return {
                "count": 0,
                "total_size_mb": 0,
                "temp_dir": str(self.temp_dir),
                "files": [],
            }

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to be safe for filesystem.

        Args:
            filename: Original filename

        Returns:
            str: Sanitized filename
        """
        # Keep only alphanumeric, hyphens, underscores, dots
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        sanitized = "".join(c for c in filename if c in safe_chars)

        # Ensure it's not empty and doesn't start with a dot
        if not sanitized or sanitized.startswith("."):
            sanitized = "uploaded_video.mp4"

        return sanitized
