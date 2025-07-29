"""
Unit tests for data schemas and validation.
"""

import pytest
from datetime import datetime
from premise_cv_platform.storage.data_schemas import (
    Detection, LineEvent, TellerInteractionEvent, AbandonmentEvent,
    EventType, ProcessingSummary
)


class TestDetection:
    """Test Detection schema."""
    
    def test_valid_detection(self):
        """Test valid detection creation."""
        detection = Detection(
            timestamp=datetime.now(),
            frame_number=100,
            person_id=1,
            confidence=0.85,
            bbox_x1=100.0,
            bbox_y1=200.0,
            bbox_x2=150.0,
            bbox_y2=250.0,
            center_x=125.0,
            center_y=225.0
        )
        
        assert detection.person_id == 1
        assert detection.confidence == 0.85
        assert detection.center_x == 125.0
    
    def test_invalid_confidence(self):
        """Test invalid confidence values."""
        with pytest.raises(ValueError):
            Detection(
                timestamp=datetime.now(),
                frame_number=100,
                person_id=1,
                confidence=1.5,  # Invalid: > 1.0
                bbox_x1=100.0,
                bbox_y1=200.0,
                bbox_x2=150.0,
                bbox_y2=250.0,
                center_x=125.0,
                center_y=225.0
            )


class TestLineEvent:
    """Test LineEvent schema."""
    
    def test_valid_line_event(self):
        """Test valid line event creation."""
        event = LineEvent(
            timestamp=datetime.now(),
            event_type=EventType.LINE_ENTERED,
            person_id="person_001",
            line_zone_id="line_zone_1"
        )
        
        assert event.event_type == EventType.LINE_ENTERED
        assert event.person_id == "person_001"


class TestTellerInteractionEvent:
    """Test TellerInteractionEvent schema."""
    
    def test_valid_teller_event(self):
        """Test valid teller interaction event creation."""
        event = TellerInteractionEvent(
            timestamp=datetime.now(),
            event_type=EventType.TELLER_INTERACTED,
            person_id="person_001",
            teller_zone_id="teller_zone_1"
        )
        
        assert event.event_type == EventType.TELLER_INTERACTED
        assert event.teller_zone_id == "teller_zone_1"


class TestAbandonmentEvent:
    """Test AbandonmentEvent schema."""
    
    def test_valid_abandonment_event(self):
        """Test valid abandonment event creation."""
        now = datetime.now()
        event = AbandonmentEvent(
            timestamp=now,
            event_type=EventType.LEFT_LINE_NO_TELLER_INTERACTION,
            person_id="person_001",
            line_entered_timestamp=now,
            line_exited_timestamp=now
        )
        
        assert event.event_type == EventType.LEFT_LINE_NO_TELLER_INTERACTION
        assert event.person_id == "person_001"


class TestProcessingSummary:
    """Test ProcessingSummary schema."""
    
    def test_valid_processing_summary(self):
        """Test valid processing summary creation."""
        summary = ProcessingSummary(
            video_file="test_video.mp4",
            processing_date=datetime.now(),
            total_frames=1000,
            processing_duration=60.5,
            fps=30.0,
            total_detections=50,
            unique_individuals=5,
            average_confidence=0.85,
            line_entries=3,
            line_exits=2,
            teller_interactions=2,
            abandonment_events=1,
            zone_statistics={}
        )
        
        assert summary.video_file == "test_video.mp4"
        assert summary.total_frames == 1000
        assert summary.unique_individuals == 5