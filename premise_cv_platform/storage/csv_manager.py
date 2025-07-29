"""
CSV file operations manager for PREMISE CV Platform with exact schema matching.
"""

import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from premise_cv_platform.config.settings import settings
from premise_cv_platform.storage.data_schemas import (
    LineEvent, TellerInteractionEvent, AbandonmentEvent, 
    ProcessingSummary, ValidationError, validate_csv_row
)


class CSVManager:
    """Manager for CSV file operations with schema validation."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or settings.output_csv_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV schemas matching examples exactly
        self.line_events_schema = ["timestamp", "event_type", "person_id", "line_zone_id"]
        self.teller_interaction_schema = ["timestamp", "event_type", "person_id", "teller_zone_id"]
        self.abandonment_schema = [
            "timestamp", "event_type", "person_id", 
            "line_entered_timestamp", "line_exited_timestamp"
        ]
        
        logger.info(f"CSV Manager initialized with output directory: {self.output_dir}")
    
    def write_line_events(self, events: List[LineEvent], filename: Optional[str] = None) -> Path:
        """Write line events to CSV file matching examples/line_events.csv schema."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"line_events_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert events to dictionaries
        rows = []
        for event in events:
            rows.append({
                "timestamp": event.timestamp.strftime(settings.csv_timestamp_format),
                "event_type": event.event_type.value,
                "person_id": event.person_id,
                "line_zone_id": event.line_zone_id
            })
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.line_events_schema)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Written {len(events)} line events to {filepath}")
        return filepath
    
    def write_teller_interaction_events(self, events: List[TellerInteractionEvent], 
                                      filename: Optional[str] = None) -> Path:
        """Write teller interaction events to CSV file matching examples/teller_interaction_events.csv schema."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"teller_interaction_events_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert events to dictionaries
        rows = []
        for event in events:
            rows.append({
                "timestamp": event.timestamp.strftime(settings.csv_timestamp_format),
                "event_type": event.event_type.value,
                "person_id": event.person_id,
                "teller_zone_id": event.teller_zone_id
            })
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.teller_interaction_schema)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Written {len(events)} teller interaction events to {filepath}")
        return filepath
    
    def write_abandonment_events(self, events: List[AbandonmentEvent], 
                               filename: Optional[str] = None) -> Path:
        """Write abandonment events to CSV file matching examples/abandonment_events.csv schema."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"abandonment_events_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert events to dictionaries
        rows = []
        for event in events:
            rows.append({
                "timestamp": event.timestamp.strftime(settings.csv_timestamp_format),
                "event_type": event.event_type.value,
                "person_id": event.person_id,
                "line_entered_timestamp": event.line_entered_timestamp.strftime(settings.csv_timestamp_format),
                "line_exited_timestamp": event.line_exited_timestamp.strftime(settings.csv_timestamp_format)
            })
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.abandonment_schema)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Written {len(events)} abandonment events to {filepath}")
        return filepath
    
    def read_line_events(self, filepath: Path) -> List[LineEvent]:
        """Read line events from CSV file with validation."""
        events = []
        errors = []
        
        try:
            df = pd.read_csv(filepath, dtype={'confidence': 'float32'})
            
            for index, row in df.iterrows():
                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(row['timestamp'], settings.csv_timestamp_format)
                    
                    event = LineEvent(
                        timestamp=timestamp,
                        event_type=row['event_type'],
                        person_id=row['person_id'],
                        line_zone_id=row['line_zone_id']
                    )
                    events.append(event)
                    
                except Exception as e:
                    error = ValidationError(
                        field="row",
                        value=dict(row),
                        error_message=f"Row {index}: {str(e)}"
                    )
                    errors.append(error)
                    logger.error(f"Error parsing line event row {index}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
        
        if errors:
            logger.warning(f"Found {len(errors)} validation errors in {filepath}")
        
        logger.info(f"Read {len(events)} valid line events from {filepath}")
        return events
    
    def read_teller_interaction_events(self, filepath: Path) -> List[TellerInteractionEvent]:
        """Read teller interaction events from CSV file with validation."""
        events = []
        errors = []
        
        try:
            df = pd.read_csv(filepath)
            
            for index, row in df.iterrows():
                try:
                    # Parse timestamp
                    timestamp = datetime.strptime(row['timestamp'], settings.csv_timestamp_format)
                    
                    event = TellerInteractionEvent(
                        timestamp=timestamp,
                        event_type=row['event_type'],
                        person_id=row['person_id'],
                        teller_zone_id=row['teller_zone_id']
                    )
                    events.append(event)
                    
                except Exception as e:
                    error = ValidationError(
                        field="row",
                        value=dict(row),
                        error_message=f"Row {index}: {str(e)}"
                    )
                    errors.append(error)
                    logger.error(f"Error parsing teller interaction event row {index}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
        
        if errors:
            logger.warning(f"Found {len(errors)} validation errors in {filepath}")
        
        logger.info(f"Read {len(events)} valid teller interaction events from {filepath}")
        return events
    
    def read_abandonment_events(self, filepath: Path) -> List[AbandonmentEvent]:
        """Read abandonment events from CSV file with validation."""
        events = []
        errors = []
        
        try:
            df = pd.read_csv(filepath)
            
            for index, row in df.iterrows():
                try:
                    # Parse timestamps
                    timestamp = datetime.strptime(row['timestamp'], settings.csv_timestamp_format)
                    line_entered = datetime.strptime(row['line_entered_timestamp'], settings.csv_timestamp_format)
                    line_exited = datetime.strptime(row['line_exited_timestamp'], settings.csv_timestamp_format)
                    
                    event = AbandonmentEvent(
                        timestamp=timestamp,
                        event_type=row['event_type'],
                        person_id=row['person_id'],
                        line_entered_timestamp=line_entered,
                        line_exited_timestamp=line_exited
                    )
                    events.append(event)
                    
                except Exception as e:
                    error = ValidationError(
                        field="row",
                        value=dict(row),
                        error_message=f"Row {index}: {str(e)}"
                    )
                    errors.append(error)
                    logger.error(f"Error parsing abandonment event row {index}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
        
        if errors:
            logger.warning(f"Found {len(errors)} validation errors in {filepath}")
        
        logger.info(f"Read {len(events)} valid abandonment events from {filepath}")
        return events
    
    def validate_csv_file(self, filepath: Path, expected_schema: List[str]) -> List[ValidationError]:
        """Validate CSV file structure and data."""
        errors = []
        
        try:
            # Check if file exists
            if not filepath.exists():
                errors.append(ValidationError(
                    field="file",
                    value=str(filepath),
                    error_message="File does not exist"
                ))
                return errors
            
            # Read CSV and check schema
            df = pd.read_csv(filepath)
            
            # Check columns
            if list(df.columns) != expected_schema:
                errors.append(ValidationError(
                    field="schema",
                    value=list(df.columns),
                    error_message=f"Expected columns: {expected_schema}, got: {list(df.columns)}"
                ))
            
            # Check for empty file
            if len(df) == 0:
                errors.append(ValidationError(
                    field="data",
                    value=len(df),
                    error_message="CSV file is empty"
                ))
            
            # Check for required columns
            for col in expected_schema:
                if col not in df.columns:
                    errors.append(ValidationError(
                        field="column",
                        value=col,
                        error_message=f"Missing required column: {col}"
                    ))
                elif df[col].isna().any():
                    null_count = df[col].isna().sum()
                    errors.append(ValidationError(
                        field="column",
                        value=col,
                        error_message=f"Column {col} has {null_count} null values"
                    ))
        
        except Exception as e:
            errors.append(ValidationError(
                field="file",
                value=str(filepath),
                error_message=f"Error reading file: {str(e)}"
            ))
        
        return errors
    
    def export_summary_to_csv(self, summary: ProcessingSummary, 
                            filename: Optional[str] = None) -> Path:
        """Export processing summary to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processing_summary_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Convert summary to dictionary
        summary_dict = {
            "video_file": [summary.video_file],
            "processing_date": [summary.processing_date.strftime("%Y-%m-%d %H:%M:%S")],
            "total_frames": [summary.total_frames],
            "processing_duration": [summary.processing_duration],
            "fps": [summary.fps],
            "total_detections": [summary.total_detections],
            "unique_individuals": [summary.unique_individuals],
            "average_confidence": [summary.average_confidence],
            "line_entries": [summary.line_entries],
            "line_exits": [summary.line_exits],
            "teller_interactions": [summary.teller_interactions],
            "abandonment_events": [summary.abandonment_events]
        }
        
        df = pd.DataFrame(summary_dict)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported processing summary to {filepath}")
        return filepath
    
    def cleanup_old_files(self, days_old: int = None) -> int:
        """Clean up CSV files older than specified days."""
        if days_old is None:
            days_old = settings.data_retention_days
        
        current_time = datetime.now()
        cleanup_count = 0
        
        for csv_file in self.output_dir.glob("*.csv"):
            file_age = current_time - datetime.fromtimestamp(csv_file.stat().st_mtime)
            
            if file_age.days > days_old:
                try:
                    csv_file.unlink()
                    cleanup_count += 1
                    logger.info(f"Removed old CSV file: {csv_file}")
                except Exception as e:
                    logger.error(f"Error removing file {csv_file}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} old CSV files")
        return cleanup_count
    
    def get_csv_file_stats(self) -> Dict[str, Any]:
        """Get statistics about CSV files in output directory."""
        stats = {
            "total_files": 0,
            "total_size_mb": 0.0,
            "file_types": {},
            "oldest_file": None,
            "newest_file": None
        }
        
        csv_files = list(self.output_dir.glob("*.csv"))
        stats["total_files"] = len(csv_files)
        
        if csv_files:
            total_size = sum(f.stat().st_size for f in csv_files)
            stats["total_size_mb"] = total_size / (1024 * 1024)
            
            # File type counting
            for csv_file in csv_files:
                file_type = csv_file.stem.split('_')[0] if '_' in csv_file.stem else 'unknown'
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            
            # Oldest and newest files
            oldest = min(csv_files, key=lambda f: f.stat().st_mtime)
            newest = max(csv_files, key=lambda f: f.stat().st_mtime)
            
            stats["oldest_file"] = {
                "name": oldest.name,
                "modified": datetime.fromtimestamp(oldest.stat().st_mtime).isoformat()
            }
            stats["newest_file"] = {
                "name": newest.name,
                "modified": datetime.fromtimestamp(newest.stat().st_mtime).isoformat()
            }
        
        return stats