"""
Pydantic data schemas for PREMISE CV Platform event validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class EventType(str, Enum):
    """Types of events that can be detected."""
    LINE_ENTERED = "line_entered"
    LINE_EXITED = "line_exited"
    TELLER_INTERACTED = "teller_interacted"
    LEFT_LINE_NO_TELLER_INTERACTION = "left_line_no_teller_interaction"
    PERSON_DETECTED = "person_detected"
    LOITERING_DETECTED = "loitering_detected"
    ZONE_VIOLATION = "zone_violation"


class Detection(BaseModel):
    """Individual person detection data."""
    timestamp: datetime = Field(..., description="Detection timestamp")
    frame_number: int = Field(..., description="Video frame number")
    person_id: int = Field(..., description="Unique person tracking ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    
    # Bounding box coordinates
    bbox_x1: float = Field(..., description="Bounding box top-left x coordinate")
    bbox_y1: float = Field(..., description="Bounding box top-left y coordinate") 
    bbox_x2: float = Field(..., description="Bounding box bottom-right x coordinate")
    bbox_y2: float = Field(..., description="Bounding box bottom-right y coordinate")
    
    # Center point coordinates
    center_x: float = Field(..., description="Person center x coordinate")
    center_y: float = Field(..., description="Person center y coordinate")
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @field_validator("bbox_x2", "bbox_y2")
    @classmethod
    def validate_bbox(cls, v):
        """Ensure bounding box coordinates are positive."""
        if v < 0:
            raise ValueError("Bounding box coordinates must be non-negative")
        return v


class LineEvent(BaseModel):
    """Line entry/exit event data matching examples/line_events.csv schema."""
    timestamp: datetime = Field(..., description="Event timestamp with microsecond precision")
    event_type: EventType = Field(..., description="Type of line event")
    person_id: str = Field(..., description="Unique person identifier (e.g., person_001)")
    line_zone_id: str = Field(..., description="Zone identifier where event occurred")
    
    @field_validator("event_type")
    @classmethod
    def validate_line_event_type(cls, v):
        """Ensure event type is valid for line events."""
        valid_types = [EventType.LINE_ENTERED, EventType.LINE_EXITED]
        if v not in valid_types:
            raise ValueError(f"Line event type must be one of: {valid_types}")
        return v
    
    # Pydantic v2 compatible


class TellerInteractionEvent(BaseModel):
    """Teller interaction event data matching examples/teller_interaction_events.csv schema."""
    timestamp: datetime = Field(..., description="Interaction timestamp")
    event_type: EventType = Field(..., description="Type of teller event")  
    person_id: str = Field(..., description="Unique person identifier")
    teller_zone_id: str = Field(..., description="Teller zone identifier")
    
    @field_validator("event_type")
    @classmethod
    def validate_teller_event_type(cls, v):
        """Ensure event type is valid for teller events."""
        if v != EventType.TELLER_INTERACTED:
            raise ValueError("Teller event type must be 'teller_interacted'")
        return v
    
    # Pydantic v2 compatible


class AbandonmentEvent(BaseModel):
    """Line abandonment event data matching examples/abandonment_events.csv schema."""
    timestamp: datetime = Field(..., description="Abandonment timestamp")
    event_type: EventType = Field(..., description="Type of abandonment event")
    person_id: str = Field(..., description="Unique person identifier")
    line_entered_timestamp: datetime = Field(..., description="When person entered the line")
    line_exited_timestamp: datetime = Field(..., description="When person exited the line")
    
    @field_validator("event_type")
    @classmethod
    def validate_abandonment_event_type(cls, v):
        """Ensure event type is valid for abandonment events."""
        if v != EventType.LEFT_LINE_NO_TELLER_INTERACTION:
            raise ValueError("Abandonment event type must be 'left_line_no_teller_interaction'")
        return v
    
    @field_validator("line_exited_timestamp")
    @classmethod
    def validate_timeline(cls, v):
        """Ensure exit timestamp is valid."""
        if v is None:
            raise ValueError("Exit timestamp cannot be None")
        return v
    
    # Pydantic v2 compatible


class PersonTrackingState(BaseModel):
    """State tracking for individual persons across frames."""
    person_id: str = Field(..., description="Unique person identifier")
    first_detected: datetime = Field(..., description="First detection timestamp")
    last_detected: datetime = Field(..., description="Last detection timestamp")
    current_zones: List[str] = Field(default_factory=list, description="Currently occupied zones")
    zone_history: List[Dict[str, Any]] = Field(default_factory=list, description="Zone entry/exit history")
    line_entered_time: Optional[datetime] = Field(None, description="Time when entered line")
    teller_interaction_time: Optional[datetime] = Field(None, description="Time of teller interaction")
    total_detections: int = Field(default=0, description="Total number of detections")
    average_confidence: float = Field(default=0.0, description="Average detection confidence")
    
    def add_zone_entry(self, zone_id: str, timestamp: datetime):
        """Record zone entry."""
        if zone_id not in self.current_zones:
            self.current_zones.append(zone_id)
            self.zone_history.append({
                "zone_id": zone_id,
                "event": "entered",
                "timestamp": timestamp.isoformat()
            })
    
    def add_zone_exit(self, zone_id: str, timestamp: datetime):
        """Record zone exit."""
        if zone_id in self.current_zones:
            self.current_zones.remove(zone_id)
            self.zone_history.append({
                "zone_id": zone_id,
                "event": "exited", 
                "timestamp": timestamp.isoformat()
            })
    
    def update_detection_stats(self, confidence: float):
        """Update detection statistics."""
        self.total_detections += 1
        self.average_confidence = (
            (self.average_confidence * (self.total_detections - 1) + confidence) / 
            self.total_detections
        )


class ProcessingSummary(BaseModel):
    """Summary data for video processing session."""
    video_file: str = Field(..., description="Processed video filename")
    processing_date: datetime = Field(..., description="Processing date and time")
    total_frames: int = Field(..., description="Total frames processed")
    processing_duration: float = Field(..., description="Processing duration in seconds")
    fps: float = Field(..., description="Video frames per second")
    
    # Detection statistics
    total_detections: int = Field(..., description="Total person detections")
    unique_individuals: int = Field(..., description="Number of unique individuals detected")
    average_confidence: float = Field(..., description="Average detection confidence")
    
    # Event statistics
    line_entries: int = Field(default=0, description="Number of line entry events")
    line_exits: int = Field(default=0, description="Number of line exit events") 
    teller_interactions: int = Field(default=0, description="Number of teller interactions")
    abandonment_events: int = Field(default=0, description="Number of abandonment events")
    
    # Zone statistics
    zone_statistics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-zone statistics"
    )
    
    # Pydantic v2 compatible


class ValidationError(BaseModel):
    """Data validation error information."""
    field: str = Field(..., description="Field that failed validation")
    value: Any = Field(..., description="Invalid value")
    error_message: str = Field(..., description="Error description")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class EventBatch(BaseModel):
    """Batch of events for efficient processing."""
    batch_id: str = Field(..., description="Unique batch identifier")
    events: List[Dict[str, Any]] = Field(..., description="List of events in batch")
    batch_timestamp: datetime = Field(default_factory=datetime.now, description="Batch creation time")
    event_count: int = Field(..., description="Number of events in batch")
    
    @field_validator("event_count")
    @classmethod
    def validate_event_count(cls, v):
        """Ensure event count is positive."""
        if v < 0:
            raise ValueError("Event count must be non-negative")
        return v


def validate_csv_row(row_data: Dict[str, Any], event_type: str) -> Optional[ValidationError]:
    """Validate a CSV row against the appropriate schema."""
    try:
        if event_type == "line_events":
            LineEvent(**row_data)
        elif event_type == "teller_interaction_events":
            TellerInteractionEvent(**row_data)
        elif event_type == "abandonment_events":
            AbandonmentEvent(**row_data)
        else:
            return ValidationError(
                field="event_type",
                value=event_type,
                error_message=f"Unknown event type: {event_type}"
            )
        return None
    except ValueError as e:
        return ValidationError(
            field="unknown",
            value=row_data,
            error_message=str(e)
        )