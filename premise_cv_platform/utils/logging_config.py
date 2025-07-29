"""
Logging configuration for PREMISE CV Platform.
"""

import sys
from pathlib import Path
from loguru import logger
from premise_cv_platform.config.settings import settings


def setup_logger() -> None:
    """Setup structured logging with proper formatting and rotation."""

    # Remove default handler
    logger.remove()

    # Ensure log directory exists
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler with color formatting
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True if settings.debug_mode else False,
    )

    # File handler for general logs
    logger.add(
        log_dir / "premise_cv.log",
        level=settings.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=settings.log_rotation_size,
        retention=settings.log_retention_count,
        compression="zip",
        backtrace=True,
        diagnose=True if settings.debug_mode else False,
    )

    # Separate file for errors only
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation="10 MB",
        retention=10,
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    # Performance monitoring log if enabled
    if settings.enable_performance_monitoring:
        logger.add(
            log_dir / "performance.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            rotation="5 MB",
            retention=5,
            filter=lambda record: "PERF" in record["extra"],
        )

    # Video processing specific log
    logger.add(
        log_dir / "video_processing.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="20 MB",
        retention=3,
        filter=lambda record: "VIDEO" in record["extra"],
    )

    # Detection and tracking log
    logger.add(
        log_dir / "detection_tracking.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="15 MB",
        retention=3,
        filter=lambda record: "DETECTION" in record["extra"],
    )

    # Zone events log
    logger.add(
        log_dir / "zone_events.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="10 MB",
        retention=5,
        filter=lambda record: "ZONE" in record["extra"],
    )

    logger.info("PREMISE CV Platform logging initialized")


def get_video_logger():
    """Get logger for video processing with VIDEO tag."""
    return logger.bind(VIDEO=True)


def get_detection_logger():
    """Get logger for detection and tracking with DETECTION tag."""
    return logger.bind(DETECTION=True)


def get_zone_logger():
    """Get logger for zone events with ZONE tag."""
    return logger.bind(ZONE=True)


def get_performance_logger():
    """Get logger for performance monitoring with PERF tag."""
    return logger.bind(PERF=True)


# Context manager for performance timing
class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.perf_logger = get_performance_logger()
        self.start_time = None

    def __enter__(self):
        import time

        self.start_time = time.perf_counter()
        self.perf_logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        end_time = time.perf_counter()
        duration = end_time - self.start_time

        if exc_type is None:
            self.perf_logger.info(f"Completed {self.operation_name} in {duration:.4f}s")
        else:
            self.perf_logger.error(
                f"Failed {self.operation_name} after {duration:.4f}s: {exc_val}"
            )


def log_system_info():
    """Log system information for debugging."""
    import platform
    import psutil
    import torch

    logger.info("System Information:")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    # GPU Information
    if torch.cuda.is_available():
        logger.info("CUDA Available: True")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("CUDA Available: False")


def log_configuration_summary():
    """Log key configuration settings."""
    logger.info("Configuration Summary:")
    logger.info(f"Video Path: {settings.video_path}")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"GPU Enabled: {settings.gpu_enabled}")
    logger.info(f"Debug Mode: {settings.debug_mode}")
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"Face Recognition: {settings.enable_face_recognition}")
    logger.info(f"Data Anonymization: {settings.anonymize_data}")


# Initialize logging when module is imported
setup_logger()
