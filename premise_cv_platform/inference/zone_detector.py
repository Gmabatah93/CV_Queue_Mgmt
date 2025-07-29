"""
Zone-based event detection and abandonment analysis for PREMISE CV Platform.
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from loguru import logger

from premise_cv_platform.config.settings import settings
from premise_cv_platform.config.zone_config import ZoneManager, Zone, ZoneType, create_default_zone_manager
from premise_cv_platform.storage.data_schemas import (
    Detection, LineEvent, TellerInteractionEvent, AbandonmentEvent, EventType
)
from premise_cv_platform.utils.logging_config import get_zone_logger


class ZoneEventDetector:
    """Zone-based event detection and abandonment analysis."""
    
    def __init__(self, zone_manager: Optional[ZoneManager] = None):
        self.zone_manager = zone_manager or self._create_default_zones()
        self.tracking_states: Dict[str, Dict[str, Any]] = {}  # person_id -> state
        self.zone_states: Dict[str, Dict[str, bool]] = {}  # person_id -> zone_id -> in_zone
        self.zone_logger = get_zone_logger()
        
        # Event storage
        self.line_events: List[LineEvent] = []
        self.teller_events: List[TellerInteractionEvent] = []
        self.abandonment_events: List[AbandonmentEvent] = []
        
        # Dwell time tracking
        self.zone_entry_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)  # person_id -> zone_id -> entry_time
        
        self.zone_logger.info("ZoneEventDetector initialized")
    
    def _create_default_zones(self) -> ZoneManager:
        """Create default zone manager from settings."""
        line_points = settings.get_line_zone_coordinates()
        teller_points = settings.get_teller_zone_coordinates()
        
        return create_default_zone_manager(line_points, teller_points)
    
    def detect_zone_events(self, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Detect zone-based events from person detections.
        
        Args:
            detections: List of person detections with coordinates
            
        Returns:
            List of events (line_entered, line_exited, teller_interacted, etc.)
        """
        events = []
        
        for detection in detections:
            person_id = f"person_{detection.person_id:03d}"
            center_point = (detection.center_x, detection.center_y)
            timestamp = detection.timestamp
            
            # Initialize tracking states if needed
            if person_id not in self.tracking_states:
                self.tracking_states[person_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'line_entries': [],
                    'line_exits': [],
                    'teller_interactions': []
                }
                self.zone_states[person_id] = {}
            
            # Update last seen
            self.tracking_states[person_id]['last_seen'] = timestamp
            
            # Check each zone
            for zone in self.zone_manager.get_active_zones():
                zone_id = zone.zone_id
                in_zone = zone.contains_point(center_point)
                
                # Get previous state
                was_in_zone = self.zone_states[person_id].get(zone_id, False)
                
                # Detect zone entry
                if in_zone and not was_in_zone:
                    event = self._handle_zone_entry(person_id, zone, timestamp)
                    if event:
                        events.append(event)
                    
                    # Record entry time for dwell time calculation
                    self.zone_entry_times[person_id][zone_id] = timestamp
                
                # Detect zone exit
                elif not in_zone and was_in_zone:
                    event = self._handle_zone_exit(person_id, zone, timestamp)
                    if event:
                        events.append(event)
                    
                    # Remove entry time
                    if zone_id in self.zone_entry_times[person_id]:
                        del self.zone_entry_times[person_id][zone_id]
                
                # Check for dwell time threshold (teller interaction)
                elif in_zone and was_in_zone and zone.zone_type == ZoneType.TELLER:
                    if self._check_teller_dwell_time(person_id, zone_id, timestamp):
                        event = self._create_teller_interaction_event(person_id, zone_id, timestamp)
                        if event:
                            events.append(event)
                
                # Update zone state
                self.zone_states[person_id][zone_id] = in_zone
        
        return events
    
    def _handle_zone_entry(self, person_id: str, zone: Zone, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Handle zone entry event."""
        if zone.zone_type == ZoneType.LINE:
            # Create line entry event
            line_event = LineEvent(
                timestamp=timestamp,
                event_type=EventType.LINE_ENTERED,
                person_id=person_id,
                line_zone_id=zone.zone_id
            )
            
            self.line_events.append(line_event)
            self.tracking_states[person_id]['line_entries'].append(timestamp)
            
            self.zone_logger.info(f"Line entry: {person_id} entered {zone.zone_id} at {timestamp}")
            
            return {
                'event_type': 'line_entered',
                'person_id': person_id,
                'zone_id': zone.zone_id,
                'timestamp': timestamp,
                'data': line_event
            }
        
        return None
    
    def _handle_zone_exit(self, person_id: str, zone: Zone, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Handle zone exit event."""
        if zone.zone_type == ZoneType.LINE:
            # Create line exit event
            line_event = LineEvent(
                timestamp=timestamp,
                event_type=EventType.LINE_EXITED,
                person_id=person_id,
                line_zone_id=zone.zone_id
            )
            
            self.line_events.append(line_event)
            self.tracking_states[person_id]['line_exits'].append(timestamp)
            
            self.zone_logger.info(f"Line exit: {person_id} exited {zone.zone_id} at {timestamp}")
            
            return {
                'event_type': 'line_exited',
                'person_id': person_id,
                'zone_id': zone.zone_id,
                'timestamp': timestamp,
                'data': line_event
            }
        
        return None
    
    def _check_teller_dwell_time(self, person_id: str, zone_id: str, current_time: datetime) -> bool:
        """Check if person has been in teller zone long enough for interaction."""
        if zone_id not in self.zone_entry_times[person_id]:
            return False
        
        entry_time = self.zone_entry_times[person_id][zone_id]
        dwell_seconds = (current_time - entry_time).total_seconds()
        
        # Check if already recorded interaction for this visit
        if person_id in self.tracking_states:
            for interaction_time in self.tracking_states[person_id].get('teller_interactions', []):
                if abs((interaction_time - entry_time).total_seconds()) < 5:  # Within 5 seconds
                    return False  # Already recorded
        
        return dwell_seconds >= settings.teller_dwell_time_threshold
    
    def _create_teller_interaction_event(self, person_id: str, zone_id: str, 
                                       timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Create teller interaction event."""
        teller_event = TellerInteractionEvent(
            timestamp=timestamp,
            event_type=EventType.TELLER_INTERACTED,
            person_id=person_id,
            teller_zone_id=zone_id
        )
        
        self.teller_events.append(teller_event)
        self.tracking_states[person_id]['teller_interactions'].append(timestamp)
        
        self.zone_logger.info(f"Teller interaction: {person_id} interacted at {zone_id} at {timestamp}")
        
        return {
            'event_type': 'teller_interacted',
            'person_id': person_id,
            'zone_id': zone_id,
            'timestamp': timestamp,
            'data': teller_event
        }
    
    def detect_abandonment_events(self) -> List[AbandonmentEvent]:
        """
        Analyze tracking data to detect abandonment events.
        Complex logic to correlate line entry/exit with teller interaction.
        """
        abandonment_events = []
        
        for person_id, state in self.tracking_states.items():
            line_entries = state.get('line_entries', [])
            line_exits = state.get('line_exits', [])
            teller_interactions = state.get('teller_interactions', [])
            
            # Analyze each line entry for potential abandonment
            for entry_time in line_entries:
                # Find corresponding exit
                corresponding_exit = self._find_corresponding_exit(entry_time, line_exits)
                
                if corresponding_exit:
                    # Check if teller interaction occurred between entry and exit
                    had_interaction = self._check_interaction_between_times(
                        teller_interactions, entry_time, corresponding_exit
                    )
                    
                    if not had_interaction:
                        # This is an abandonment event
                        abandonment_event = AbandonmentEvent(
                            timestamp=corresponding_exit,
                            event_type=EventType.LEFT_LINE_NO_TELLER_INTERACTION,
                            person_id=person_id,
                            line_entered_timestamp=entry_time,
                            line_exited_timestamp=corresponding_exit
                        )
                        
                        abandonment_events.append(abandonment_event)
                        self.abandonment_events.append(abandonment_event)
                        
                        self.zone_logger.warning(
                            f"Abandonment detected: {person_id} left line without interaction "
                            f"(entered: {entry_time}, exited: {corresponding_exit})"
                        )
        
        return abandonment_events
    
    def _find_corresponding_exit(self, entry_time: datetime, 
                               exits: List[datetime]) -> Optional[datetime]:
        """Find the exit time that corresponds to a given entry time."""
        # Find the first exit after the entry
        valid_exits = [exit_time for exit_time in exits if exit_time > entry_time]
        
        if valid_exits:
            return min(valid_exits)  # Return the earliest exit after entry
        
        return None
    
    def _check_interaction_between_times(self, interactions: List[datetime],
                                       start_time: datetime, end_time: datetime) -> bool:
        """Check if any teller interaction occurred between start and end times."""
        for interaction_time in interactions:
            if start_time <= interaction_time <= end_time:
                return True
        return False
    
    def get_zone_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive zone statistics."""
        # Calculate abandonment rate
        total_line_entries = len([e for e in self.line_events if e.event_type == EventType.LINE_ENTERED])
        abandonment_rate = 0.0
        if total_line_entries > 0:
            abandonment_rate = (len(self.abandonment_events) / total_line_entries) * 100
        
        # Create nested structure for zone statistics
        stats = {
            'overall': {
                'total_line_entries': len(self.line_events),
                'total_teller_interactions': len(self.teller_events),
                'total_abandonment_events': len(self.abandonment_events),
                'abandonment_rate': abandonment_rate
            },
            'zone_occupancy': {},
            'average_dwell_times': {}
        }
        
        # Zone occupancy statistics
        for zone in self.zone_manager.get_active_zones():
            zone_id = zone.zone_id
            
            # Count people currently in zone
            current_occupancy = 0
            for person_states in self.zone_states.values():
                if person_states.get(zone_id, False):
                    current_occupancy += 1
            
            stats['zone_occupancy'][zone_id] = current_occupancy
        
        # Calculate average dwell times
        for zone in self.zone_manager.get_active_zones():
            zone_id = zone.zone_id
            dwell_times = []
            
            for person_id, zone_times in self.zone_entry_times.items():
                if zone_id in zone_times:
                    entry_time = zone_times[zone_id]
                    current_time = datetime.now()
                    dwell_seconds = (current_time - entry_time).total_seconds()
                    dwell_times.append(dwell_seconds)
            
            if dwell_times:
                stats['average_dwell_times'][zone_id] = np.mean(dwell_times)
            else:
                stats['average_dwell_times'][zone_id] = 0.0
        
        return stats
    
    def visualize_zones_on_frame(self, frame: np.ndarray, 
                               show_zone_names: bool = True) -> np.ndarray:
        """Draw zones on frame for visualization.""" 
        return self.zone_manager.draw_zones_on_frame(frame)
    
    def get_current_zone_occupancy(self) -> Dict[str, List[str]]:
        """Get current occupancy for each zone."""
        occupancy = defaultdict(list)
        
        for person_id, zone_states in self.zone_states.items():
            for zone_id, in_zone in zone_states.items():
                if in_zone:
                    occupancy[zone_id].append(person_id)
        
        return dict(occupancy)
    
    def export_events_data(self) -> Dict[str, Any]:
        """Export all events data for analysis."""
        return {
            'line_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'person_id': event.person_id,
                    'line_zone_id': event.line_zone_id
                }
                for event in self.line_events
            ],
            'teller_interaction_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'person_id': event.person_id,
                    'teller_zone_id': event.teller_zone_id
                }
                for event in self.teller_events
            ],
            'abandonment_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'person_id': event.person_id,
                    'line_entered_timestamp': event.line_entered_timestamp.isoformat(),
                    'line_exited_timestamp': event.line_exited_timestamp.isoformat()
                }
                for event in self.abandonment_events
            ],
            'statistics': self.get_zone_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def cleanup_old_tracking_data(self, max_age_minutes: int = 30) -> int:
        """Clean up tracking data for persons not seen recently."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=max_age_minutes)
        
        persons_to_remove = []
        for person_id, state in self.tracking_states.items():
            if state['last_seen'] < cutoff_time:
                persons_to_remove.append(person_id)
        
        # Remove old tracking data
        for person_id in persons_to_remove:
            if person_id in self.tracking_states:
                del self.tracking_states[person_id]
            if person_id in self.zone_states:
                del self.zone_states[person_id]
            if person_id in self.zone_entry_times:
                del self.zone_entry_times[person_id]
        
        if persons_to_remove:
            self.zone_logger.info(f"Cleaned up tracking data for {len(persons_to_remove)} persons")
        
        return len(persons_to_remove)