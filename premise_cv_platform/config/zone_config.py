"""
Zone configuration utilities for PREMISE CV Platform.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum


class ZoneType(Enum):
    """Types of zones in the banking environment."""
    LINE = "line"
    TELLER = "teller"  
    ENTRANCE = "entrance"
    EXIT = "exit"
    WAITING = "waiting"
    ATM = "atm"
    RESTRICTED = "restricted"


@dataclass
class Zone:
    """Zone definition with polygon points and metadata."""
    zone_id: str
    zone_type: ZoneType
    points: List[Tuple[float, float]]
    name: str
    description: str = ""
    active: bool = True
    
    def __post_init__(self):
        """Validate zone after initialization."""
        if len(self.points) < 3:
            raise ValueError(f"Zone {self.zone_id} must have at least 3 points")
        
        # Convert points to numpy array for cv2 operations
        self.polygon = np.array(self.points, dtype=np.int32)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside this zone using cv2.pointPolygonTest."""
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0
    
    def get_center(self) -> Tuple[float, float]:
        """Calculate the center point of the zone."""
        center_x = sum(p[0] for p in self.points) / len(self.points)
        center_y = sum(p[1] for p in self.points) / len(self.points)
        return (center_x, center_y)
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box coordinates (x_min, y_min, x_max, y_max)."""
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert zone to dictionary for serialization."""
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type.value,
            "points": self.points,
            "name": self.name,
            "description": self.description,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Zone':
        """Create zone from dictionary."""
        return cls(
            zone_id=data["zone_id"],
            zone_type=ZoneType(data["zone_type"]),
            points=data["points"],
            name=data["name"],
            description=data.get("description", ""),
            active=data.get("active", True)
        )


class ZoneManager:
    """Manager for all zones in the banking environment."""
    
    def __init__(self):
        self.zones: Dict[str, Zone] = {}
    
    def add_zone(self, zone: Zone) -> None:
        """Add a zone to the manager."""
        self.zones[zone.zone_id] = zone
    
    def remove_zone(self, zone_id: str) -> None:
        """Remove a zone from the manager."""
        if zone_id in self.zones:
            del self.zones[zone_id]
    
    def get_zone(self, zone_id: str) -> Zone:
        """Get a zone by ID."""
        if zone_id not in self.zones:
            raise KeyError(f"Zone {zone_id} not found")
        return self.zones[zone_id]
    
    def get_zones_by_type(self, zone_type: ZoneType) -> List[Zone]:
        """Get all zones of a specific type."""
        return [zone for zone in self.zones.values() if zone.zone_type == zone_type]
    
    def find_zones_containing_point(self, point: Tuple[float, float]) -> List[Zone]:
        """Find all zones that contain a given point."""
        return [zone for zone in self.zones.values() 
                if zone.active and zone.contains_point(point)]
    
    def get_active_zones(self) -> List[Zone]:
        """Get all active zones."""
        return [zone for zone in self.zones.values() if zone.active]
    
    def validate_all_zones(self) -> Dict[str, List[str]]:
        """Validate all zones and return any errors."""
        errors = {}
        
        for zone_id, zone in self.zones.items():
            zone_errors = []
            
            # Check minimum points
            if len(zone.points) < 3:
                zone_errors.append("Zone must have at least 3 points")
            
            # Check for self-intersecting polygons (basic check)
            if len(zone.points) >= 4:
                # Simple check for obvious self-intersections
                x_coords = [p[0] for p in zone.points]
                y_coords = [p[1] for p in zone.points]
                
                if len(set(x_coords)) == 1 or len(set(y_coords)) == 1:
                    zone_errors.append("Zone points form a line, not a polygon")
            
            if zone_errors:
                errors[zone_id] = zone_errors
        
        return errors
    
    def load_default_zones(self, line_points: List[Tuple[float, float]], 
                          teller_points: List[Tuple[float, float]]) -> None:
        """Load default zones from configuration points."""
        # Add line zone
        line_zone = Zone(
            zone_id="line_zone_a",
            zone_type=ZoneType.LINE,
            points=line_points,
            name="Main Line Zone",
            description="Primary queue formation area around blue logo"
        )
        self.add_zone(line_zone)
        
        # Add teller zone
        teller_zone = Zone(
            zone_id="teller_zone_1",
            zone_type=ZoneType.TELLER,
            points=teller_points,
            name="Teller Interaction Zone",
            description="Area in front of red chairs for teller interactions"
        )
        self.add_zone(teller_zone)
    
    def draw_zones_on_frame(self, frame: np.ndarray, 
                           zone_types: List[ZoneType] = None,
                           colors: Dict[ZoneType, Tuple[int, int, int]] = None) -> np.ndarray:
        """Draw zones on a video frame for visualization."""
        if colors is None:
            colors = {
                ZoneType.LINE: (0, 255, 0),        # Green
                ZoneType.TELLER: (255, 0, 0),      # Red
                ZoneType.ENTRANCE: (0, 0, 255),    # Blue
                ZoneType.EXIT: (255, 255, 0),      # Cyan
                ZoneType.WAITING: (255, 0, 255),   # Magenta
                ZoneType.ATM: (0, 255, 255),       # Yellow
                ZoneType.RESTRICTED: (128, 128, 128)  # Gray
            }
        
        frame_copy = frame.copy()
        
        for zone in self.get_active_zones():
            if zone_types is None or zone.zone_type in zone_types:
                color = colors.get(zone.zone_type, (255, 255, 255))
                
                # Draw polygon
                cv2.polylines(frame_copy, [zone.polygon], True, color, 2)
                
                # Add zone label
                center = zone.get_center()
                label_pos = (int(center[0]), int(center[1]))
                
                # Add background rectangle for text
                text_size = cv2.getTextSize(zone.name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame_copy, 
                            (label_pos[0] - text_size[0]//2 - 5, label_pos[1] - text_size[1] - 5),
                            (label_pos[0] + text_size[0]//2 + 5, label_pos[1] + 5),
                            color, -1)
                
                # Add text
                cv2.putText(frame_copy, zone.name, 
                          (label_pos[0] - text_size[0]//2, label_pos[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    def export_zones_config(self) -> Dict[str, Any]:
        """Export all zones configuration to dictionary."""
        return {
            "zones": [zone.to_dict() for zone in self.zones.values()],
            "version": "1.0",
            "description": "PREMISE CV Platform zone configuration"
        }
    
    def import_zones_config(self, config: Dict[str, Any]) -> None:
        """Import zones configuration from dictionary."""
        self.zones.clear()
        
        for zone_data in config.get("zones", []):
            zone = Zone.from_dict(zone_data)
            self.add_zone(zone)


def create_default_zone_manager(line_points: List[Tuple[float, float]], 
                               teller_points: List[Tuple[float, float]]) -> ZoneManager:
    """Create a default zone manager with line and teller zones."""
    manager = ZoneManager()
    manager.load_default_zones(line_points, teller_points)
    return manager