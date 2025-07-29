"""
Unit tests for CSV data management.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime
from premise_cv_platform.storage.csv_manager import CSVManager
from premise_cv_platform.storage.data_schemas import (
    LineEvent, TellerInteractionEvent, AbandonmentEvent, EventType, ProcessingSummary
)


class TestCSVManager:
    """Test CSV data export and management."""
    
    def setup_method(self):
        """Setup test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.csv_manager = CSVManager(self.test_dir)
    
    def test_write_line_events(self):
        """Test writing line events to CSV."""
        events = [
            LineEvent(
                timestamp=datetime.now(),
                event_type=EventType.LINE_ENTERED,
                person_id="person_001",
                line_zone_id="line_zone_1"
            ),
            LineEvent(
                timestamp=datetime.now(),
                event_type=EventType.LINE_EXITED,
                person_id="person_001",
                line_zone_id="line_zone_1"
            )
        ]
        
        file_path = self.csv_manager.write_line_events(events)
        assert file_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(file_path)
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "event_type" in df.columns
        assert "person_id" in df.columns
        assert "line_zone_id" in df.columns
    
    def test_write_teller_interaction_events(self):
        """Test writing teller interaction events to CSV."""
        events = [
            TellerInteractionEvent(
                timestamp=datetime.now(),
                event_type=EventType.TELLER_INTERACTED,
                person_id="person_001",
                teller_zone_id="teller_zone_1"
            )
        ]
        
        file_path = self.csv_manager.write_teller_interaction_events(events)
        assert file_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(file_path)
        assert len(df) == 1
        assert "teller_zone_id" in df.columns
    
    def test_write_abandonment_events(self):
        """Test writing abandonment events to CSV."""
        now = datetime.now()
        events = [
            AbandonmentEvent(
                timestamp=now,
                event_type=EventType.LEFT_LINE_NO_TELLER_INTERACTION,
                person_id="person_001",
                line_entered_timestamp=now,
                line_exited_timestamp=now
            )
        ]
        
        file_path = self.csv_manager.write_abandonment_events(events)
        assert file_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(file_path)
        assert len(df) == 1
        assert "line_entered_timestamp" in df.columns
        assert "line_exited_timestamp" in df.columns
    
    def test_export_summary_to_csv(self):
        """Test exporting processing summary to CSV."""
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
        
        file_path = self.csv_manager.export_summary_to_csv(summary)
        assert file_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(file_path)
        assert len(df) == 1
        assert df.iloc[0]['video_file'] == "test_video.mp4"
        assert df.iloc[0]['unique_individuals'] == 5
    
    def test_get_csv_file_stats(self):
        """Test getting CSV file statistics."""
        # Create some test files first
        events = [
            LineEvent(
                timestamp=datetime.now(),
                event_type=EventType.LINE_ENTERED,
                person_id="person_001",
                line_zone_id="line_zone_1"
            )
        ]
        self.csv_manager.write_line_events(events)
        
        stats = self.csv_manager.get_csv_file_stats()
        assert stats['total_files'] >= 1
        assert 'total_size_mb' in stats
        assert 'file_types' in stats