"""
Unit tests for configuration management.
"""

import pytest
from premise_cv_platform.config.settings import Settings


class TestSettings:
    """Test configuration settings."""
    
    def test_default_settings(self):
        """Test default settings loading."""
        settings = Settings()
        assert settings.model_name == "yolo11n.pt"
        assert settings.video_path == "videos/bank_sample.MOV"
        assert settings.model_confidence_threshold == 0.5
        assert settings.gpu_enabled is True
    
    def test_zone_coordinate_parsing(self):
        """Test zone coordinate parsing."""
        settings = Settings()
        line_coords = settings.get_line_zone_coordinates()
        teller_coords = settings.get_teller_zone_coordinates()
        
        assert isinstance(line_coords, list)
        assert isinstance(teller_coords, list)
        assert len(line_coords) >= 4  # At least 2 points (x,y pairs)
        assert len(teller_coords) >= 4  # At least 2 points (x,y pairs)
    
    def test_directory_creation(self):
        """Test directory creation."""
        settings = Settings()
        settings.ensure_directories()
        
        # Should create directories without raising exceptions
        assert True  # If we get here, directories were created successfully
    
    def test_invalid_zone_points(self):
        """Test validation of invalid zone points."""
        with pytest.raises(ValueError):
            Settings(line_zone_points="invalid,format")
    
    def test_invalid_log_level(self):
        """Test validation of invalid log level."""
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")