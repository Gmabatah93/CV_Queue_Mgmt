"""
Async processing wrapper for integrating PremiseCVPipeline with Streamlit dashboard.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Literal
from datetime import datetime
from loguru import logger

# Import platform components
from premise_cv_platform.config.settings import settings
from premise_cv_platform.utils.file_handler import TempFileManager


@dataclass
class ProcessingStatus:
    """Status tracking for video processing operations."""

    status: Literal["idle", "uploading", "processing", "completed", "error"]
    message: str
    progress: float = 0.0
    video_file: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "video_file": self.video_file,
            "results": self.results,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class UploadedVideoProcessor:
    """Async wrapper for PremiseCVPipeline to integrate with Streamlit."""

    def __init__(self, temp_dir: str = "data/temp"):
        """Initialize processor with temp directory."""
        self.temp_dir = temp_dir
        self.file_manager = TempFileManager(temp_dir)
        self.current_status = ProcessingStatus(status="idle", message="Ready")
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = False

        logger.info(f"UploadedVideoProcessor initialized with temp_dir: {temp_dir}")

    def process_uploaded_video(
        self,
        uploaded_file,
        progress_callback: Optional[Callable[[ProcessingStatus], None]] = None,
        output_dir: Optional[str] = None,
    ) -> ProcessingStatus:
        """
        Process uploaded file with progress tracking.

        Args:
            uploaded_file: Streamlit UploadedFile object
            progress_callback: Optional callback for progress updates
            output_dir: Optional output directory override

        Returns:
            ProcessingStatus: Final processing status
        """
        # Reset processing state
        self.stop_processing = False
        self.current_status = ProcessingStatus(
            status="uploading",
            message="Saving uploaded file...",
            start_time=datetime.now(),
        )

        if progress_callback:
            progress_callback(self.current_status)

        try:
            # Save uploaded file to temp location
            temp_file_path = self.file_manager.save_uploaded_file(uploaded_file)
            self.current_status.video_file = temp_file_path

            # Start processing in background thread
            self.current_status.status = "processing"
            self.current_status.message = "Starting video analysis..."
            self.current_status.progress = 0.0

            if progress_callback:
                progress_callback(self.current_status)

            # Run processing in thread
            processing_args = (temp_file_path, progress_callback, output_dir)
            self.processing_thread = threading.Thread(
                target=self._run_processing_thread, args=processing_args, daemon=True
            )
            self.processing_thread.start()

            # Wait for processing to complete or timeout
            self.processing_thread.join(timeout=3600)  # 1 hour timeout

            # Check if thread is still alive (timeout occurred)
            if self.processing_thread.is_alive():
                self.stop_processing = True
                self.current_status.status = "error"
                self.current_status.error = "Processing timeout (1 hour)"
                self.current_status.message = "Processing timed out"
                logger.error("Video processing timed out after 1 hour")

            return self.current_status

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)

            self.current_status.status = "error"
            self.current_status.error = error_msg
            self.current_status.message = "Processing failed"
            self.current_status.end_time = datetime.now()

            # Clean up temp file on error
            if (
                hasattr(self.current_status, "video_file")
                and self.current_status.video_file
            ):
                self.file_manager.cleanup_temp_file(self.current_status.video_file)

            if progress_callback:
                progress_callback(self.current_status)

            return self.current_status

    def _run_processing_thread(
        self,
        temp_file_path: str,
        progress_callback: Optional[Callable[[ProcessingStatus], None]],
        output_dir: Optional[str],
    ):
        """Run video processing in background thread."""
        try:
            # Import here to avoid circular imports
            from main import PremiseCVPipeline

            # Initialize pipeline with uploaded video
            output_path = output_dir or settings.output_csv_dir
            pipeline = PremiseCVPipeline(
                video_path=temp_file_path, output_dir=output_path
            )

            # Update status
            self.current_status.message = "Analyzing video frames..."
            self.current_status.progress = 10.0
            if progress_callback:
                progress_callback(self.current_status)

            # Custom progress tracking for pipeline
            original_process_method = pipeline.process_video

            def progress_wrapper(*args, **kwargs):
                """Wrapper to track progress during processing."""

                def frame_progress_callback(frame_num, total_frames):
                    if self.stop_processing:
                        return False  # Signal to stop processing

                    progress_percent = min(
                        90.0, 10.0 + (frame_num / total_frames * 80.0)
                    )
                    self.current_status.progress = progress_percent
                    self.current_status.message = (
                        f"Processing frame {frame_num} of {total_frames}..."
                    )

                    if progress_callback:
                        progress_callback(self.current_status)

                    return True  # Continue processing

                # Monkey patch progress tracking if possible
                if hasattr(pipeline.video_processor, "set_progress_callback"):
                    pipeline.video_processor.set_progress_callback(
                        frame_progress_callback
                    )

                return original_process_method(*args, **kwargs)

            # Execute video processing with visualization enabled for dashboard
            logger.info(f"Starting video processing for: {temp_file_path}")
            
            # Enable debug frame saving for visual feedback
            original_save_debug = settings.save_debug_frames
            original_debug_interval = settings.debug_frame_interval
            settings.save_debug_frames = True
            settings.debug_frame_interval = 30  # Save every 30th frame
            
            try:
                results = progress_wrapper(save_visualization=True)
            finally:
                # Restore original settings
                settings.save_debug_frames = original_save_debug
                settings.debug_frame_interval = original_debug_interval

            if self.stop_processing:
                self.current_status.status = "error"
                self.current_status.error = "Processing was cancelled"
                self.current_status.message = "Processing cancelled"
            else:
                # Update final status
                self.current_status.status = "completed"
                self.current_status.message = "Video analysis completed successfully"
                self.current_status.progress = 100.0
                self.current_status.results = results
                self.current_status.end_time = datetime.now()

                logger.info(
                    f"Video processing completed successfully for: {temp_file_path}"
                )

            # Clean up temporary file
            self.file_manager.cleanup_temp_file(temp_file_path)

            if progress_callback:
                progress_callback(self.current_status)

        except Exception as e:
            error_msg = f"Processing thread error: {str(e)}"
            logger.error(error_msg)

            self.current_status.status = "error"
            self.current_status.error = error_msg
            self.current_status.message = "Processing failed with error"
            self.current_status.end_time = datetime.now()

            # Clean up on error
            try:
                self.file_manager.cleanup_temp_file(temp_file_path)
            except Exception:
                pass

            if progress_callback:
                progress_callback(self.current_status)

    def get_current_status(self) -> ProcessingStatus:
        """Get current processing status."""
        return self.current_status

    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self.current_status.status == "processing"

    def cancel_processing(self) -> None:
        """Cancel current processing operation."""
        if self.is_processing():
            logger.info("Cancelling video processing...")
            self.stop_processing = True
            self.current_status.message = "Cancelling processing..."

    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up old temporary files."""
        return self.file_manager.cleanup_old_temp_files(max_age_hours)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        temp_stats = self.file_manager.get_temp_file_stats()

        processing_time = None
        if self.current_status.start_time and self.current_status.end_time:
            processing_time = (
                self.current_status.end_time - self.current_status.start_time
            ).total_seconds()

        return {
            "current_status": self.current_status.to_dict(),
            "temp_files": temp_stats,
            "processing_time_seconds": processing_time,
            "is_processing": self.is_processing(),
        }


class ProcessingManager:
    """Manager for multiple concurrent processing operations."""

    def __init__(self):
        """Initialize processing manager."""
        self.processors: Dict[str, UploadedVideoProcessor] = {}
        self.active_sessions: Dict[str, str] = {}  # session_id -> processor_id

    def get_processor(self, session_id: str) -> UploadedVideoProcessor:
        """Get or create processor for session."""
        if session_id not in self.active_sessions:
            processor_id = f"processor_{session_id}_{int(time.time())}"
            processor = UploadedVideoProcessor()
            self.processors[processor_id] = processor
            self.active_sessions[session_id] = processor_id
            logger.info(
                f"Created new processor {processor_id} for session {session_id}"
            )

        processor_id = self.active_sessions[session_id]
        return self.processors[processor_id]

    def cleanup_session(self, session_id: str) -> None:
        """Clean up processor for session."""
        if session_id in self.active_sessions:
            processor_id = self.active_sessions[session_id]
            if processor_id in self.processors:
                processor = self.processors[processor_id]
                processor.cancel_processing()
                processor.cleanup_temp_files()
                del self.processors[processor_id]
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all processors."""
        return {
            "total_processors": len(self.processors),
            "active_sessions": len(self.active_sessions),
            "processors": {
                proc_id: processor.get_processing_stats()
                for proc_id, processor in self.processors.items()
            },
        }


# Global processing manager instance
processing_manager = ProcessingManager()
