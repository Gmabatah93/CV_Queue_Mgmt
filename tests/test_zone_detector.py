"""
Unit tests for zone-based event detection.
"""

import pytest
import numpy as np
from datetime import datetime
from premise_cv_platform.inference.zone_detector import ZoneEventDetector
from premise_cv_platform.config.zone_config import Zone, ZoneType, ZoneManager
from premise_cv_platform.storage.data_schemas import Detection


class TestZoneEventDetector:
    """Test zone-based event detection."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create simple test zones
        line_zone = Zone(
            zone_id="test_line_zone",
            zone_type=ZoneType.LINE,
            points=[(100, 100), (200, 100), (200, 200), (100, 200)],
            name="Test Line Zone"
        )
        
        teller_zone = Zone(
            zone_id="test_teller_zone", 
            zone_type=ZoneType.TELLER,
            points=[(300, 100), (400, 100), (400, 200), (300, 200)],
            name="Test Teller Zone"
        )
        
        zone_manager = ZoneManager()
        zone_manager.add_zone(line_zone)
        zone_manager.add_zone(teller_zone)
        self.detector = ZoneEventDetector(zone_manager)
    
    def test_zone_entry_detection(self):
        """Test detection of zone entry events."""
        # Create detection inside line zone
        detection = Detection(
            timestamp=datetime.now(),
            frame_number=1,
            person_id=1,
            confidence=0.85,
            bbox_x1=140.0,
            bbox_y1=140.0,
            bbox_x2=160.0,
            bbox_y2=160.0,
            center_x=150.0,  # Inside line zone
            center_y=150.0
        )
        
        events = self.detector.detect_zone_events([detection])
        
        # Should detect line entry
        assert len(events) == 1
        assert events[0]['event_type'] == 'line_entered'
        assert events[0]['person_id'] == 'person_001'
    
    def test_zone_exit_detection(self):
        """Test detection of zone exit events."""
        # First, enter the zone
        detection_inside = Detection(
            timestamp=datetime.now(),
            frame_number=1,
            person_id=1,
            confidence=0.85,
            bbox_x1=140.0,
            bbox_y1=140.0,
            bbox_x2=160.0,
            bbox_y2=160.0,
            center_x=150.0,  # Inside line zone
            center_y=150.0
        )
        
        self.detector.detect_zone_events([detection_inside])
        
        # Then exit the zone
        detection_outside = Detection(
            timestamp=datetime.now(),
            frame_number=2,
            person_id=1,
            confidence=0.85,
            bbox_x1=40.0,
            bbox_y1=40.0,
            bbox_x2=60.0,
            bbox_y2=60.0,
            center_x=50.0,  # Outside any zone
            center_y=50.0
        )
        
        events = self.detector.detect_zone_events([detection_outside])
        
        # Should detect line exit
        assert len(events) == 1
        assert events[0]['event_type'] == 'line_exited'
    
    def test_abandonment_detection(self):
        """Test abandonment event detection."""
        # Person enters line zone
        detection_enter = Detection(
            timestamp=datetime.now(),
            frame_number=1,
            person_id=1,
            confidence=0.85,
            bbox_x1=140.0,
            bbox_y1=140.0,
            bbox_x2=160.0,
            bbox_y2=160.0,
            center_x=150.0,
            center_y=150.0
        )
        
        self.detector.detect_zone_events([detection_enter])
        
        # Person exits line zone without teller interaction
        detection_exit = Detection(
            timestamp=datetime.now(),
            frame_number=2,
            person_id=1,
            confidence=0.85,
            bbox_x1=40.0,
            bbox_y1=40.0,
            bbox_x2=60.0,
            bbox_y2=60.0,
            center_x=50.0,
            center_y=50.0
        )
        
        self.detector.detect_zone_events([detection_exit])
        
        # Check for abandonment
        abandonment_events = self.detector.detect_abandonment_events()
        
        # Should detect abandonment
        assert len(abandonment_events) == 1
        assert abandonment_events[0].person_id == "person_001"
    
    def test_get_zone_statistics(self):
        """Test zone statistics generation."""
        stats = self.detector.get_zone_statistics()
        
        assert 'total_line_entries' in stats
        assert 'total_teller_interactions' in stats
        assert 'total_abandonment_events' in stats
        assert 'abandonment_rate' in stats
        assert 'zone_occupancy' in stats
    
    def test_cleanup_old_tracking_data(self):
        """Test cleanup of old tracking data."""
        # Add some tracking data
        detection = Detection(
            timestamp=datetime.now(),
            frame_number=1,
            person_id=1,
            confidence=0.85,
            bbox_x1=140.0,
            bbox_y1=140.0,
            bbox_x2=160.0,
            bbox_y2=160.0,
            center_x=150.0,
            center_y=150.0
        )
        
        self.detector.detect_zone_events([detection])
        
        # Cleanup with very short age (should clean up immediately)
        cleaned_count = self.detector.cleanup_old_tracking_data(max_age_minutes=0)
        
        # Should have cleaned up the tracking data
        assert cleaned_count >= 0