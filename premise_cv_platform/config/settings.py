"""
Configuration management for PREMISE CV Platform using Pydantic Settings.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main configuration class for PREMISE CV Platform."""
    
    # Video Processing Configuration
    video_path: str = Field(default="videos/bank_sample.MOV", description="Path to input video file")
    video_processing_fps: int = Field(default=30, description="FPS for video processing")
    video_buffer_size: int = Field(default=100, description="Video frame buffer size")
    
    # YOLO Model Configurationa
    model_name: str = Field(default="yolo11n.pt", description="YOLO model filename")
    model_confidence_threshold: float = Field(default=0.5, description="Confidence threshold for detections")
    model_iou_threshold: float = Field(default=0.5, description="IoU threshold for NMS")
    tracking_persistence: bool = Field(default=True, description="Enable tracking persistence")
    
    # Zone Configuration
    line_zone_points: str = Field(
        default="400,300,600,300,650,500,350,500",
        description="Line zone coordinates as comma-separated x,y pairs"
    )
    teller_zone_points: str = Field(
        default="300,100,500,100,500,200,300,200",
        description="Teller zone coordinates as comma-separated x,y pairs"
    )
    teller_dwell_time_threshold: int = Field(
        default=10,
        description="Seconds threshold for teller interaction detection"
    )
    
    # Data Storage Configuration
    output_csv_dir: str = Field(default="data/csv_exports", description="Directory for CSV exports")
    log_dir: str = Field(default="data/logs", description="Directory for log files")
    processed_video_dir: str = Field(default="data/processed", description="Directory for processed videos")
    
    # CSV Export Settings
    csv_timestamp_format: str = Field(default="%Y-%m-%d %H:%M:%S.%f", description="Timestamp format for CSV")
    csv_export_batch_size: int = Field(default=1000, description="Batch size for CSV exports")
    
    # Dashboard Configuration
    dashboard_host: str = Field(default="localhost", description="Dashboard host")
    dashboard_port: int = Field(default=8501, description="Dashboard port")
    dashboard_update_interval: int = Field(default=5, description="Dashboard update interval in seconds")
    dashboard_cache_ttl: int = Field(default=60, description="Dashboard cache TTL in seconds")
    
    # Security & Privacy Settings
    enable_face_recognition: bool = Field(default=False, description="Enable facial recognition")
    anonymize_data: bool = Field(default=True, description="Anonymize exported data")
    data_retention_days: int = Field(default=30, description="Data retention period in days")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Performance Settings
    gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration")
    batch_processing: bool = Field(default=False, description="Enable batch processing")
    max_concurrent_streams: int = Field(default=4, description="Maximum concurrent video streams")
    memory_optimization: bool = Field(default=True, description="Enable memory optimization")
    
    # Alert Thresholds
    queue_length_alert_threshold: int = Field(default=5, description="Queue length alert threshold")
    wait_time_alert_threshold: int = Field(default=300, description="Wait time alert threshold in seconds")
    abandonment_rate_alert_threshold: float = Field(default=20.0, description="Abandonment rate alert threshold percentage")
    
    # Integration Settings
    enable_email_alerts: bool = Field(default=False, description="Enable email alerts")
    email_smtp_server: Optional[str] = Field(default=None, description="SMTP server for email alerts")
    email_smtp_port: int = Field(default=587, description="SMTP port")
    email_username: Optional[str] = Field(default=None, description="Email username")
    email_password: Optional[str] = Field(default=None, description="Email password")
    alert_recipients: Optional[str] = Field(default=None, description="Comma-separated alert recipients")
    
    # Database Configuration
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    database_connection_pool_size: int = Field(default=10, description="Database connection pool size")
    
    # Cloud Configuration
    cloud_storage_enabled: bool = Field(default=False, description="Enable cloud storage")
    cloud_storage_bucket: Optional[str] = Field(default=None, description="Cloud storage bucket")
    cloud_storage_region: Optional[str] = Field(default=None, description="Cloud storage region")
    cloud_access_key: Optional[str] = Field(default=None, description="Cloud access key")
    cloud_secret_key: Optional[str] = Field(default=None, description="Cloud secret key")
    
    # Development Settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    save_debug_frames: bool = Field(default=False, description="Save debug frames")
    debug_frame_interval: int = Field(default=30, description="Debug frame save interval")
    
    # API Configuration
    api_enabled: bool = Field(default=False, description="Enable API server")
    api_host: str = Field(default="localhost", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="API worker processes")
    api_secret_key: str = Field(default="your-secret-key-here", description="API secret key")
    
    # Monitoring & Logging
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    log_rotation_size: str = Field(default="10MB", description="Log rotation size")
    log_retention_count: int = Field(default=5, description="Log retention count")
    metrics_export_interval: int = Field(default=60, description="Metrics export interval in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @field_validator("line_zone_points")
    @classmethod
    def validate_line_zone_points(cls, v):
        """Validate line zone points format."""
        try:
            points = [float(x) for x in v.split(",")]
            if len(points) % 2 != 0:
                raise ValueError("Zone points must be pairs of x,y coordinates")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid zone points format: {e}")
    
    @field_validator("teller_zone_points")
    @classmethod
    def validate_teller_zone_points(cls, v):
        """Validate teller zone points format."""
        try:
            points = [float(x) for x in v.split(",")]
            if len(points) % 2 != 0:
                raise ValueError("Zone points must be pairs of x,y coordinates")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid zone points format: {e}")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def get_line_zone_coordinates(self) -> List[Tuple[float, float]]:
        """Parse line zone points into coordinate tuples."""
        points = [float(x) for x in self.line_zone_points.split(",")]
        return [(points[i], points[i+1]) for i in range(0, len(points), 2)]
    
    def get_teller_zone_coordinates(self) -> List[Tuple[float, float]]:
        """Parse teller zone points into coordinate tuples."""
        points = [float(x) for x in self.teller_zone_points.split(",")]
        return [(points[i], points[i+1]) for i in range(0, len(points), 2)]
    
    def get_alert_recipients_list(self) -> List[str]:
        """Parse alert recipients into list."""
        if not self.alert_recipients:
            return []
        return [email.strip() for email in self.alert_recipients.split(",")]
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.output_csv_dir,
            self.log_dir,
            self.processed_video_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()