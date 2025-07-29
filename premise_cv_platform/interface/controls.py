"""
Interactive Zone Configuration Controls

This module provides interactive widgets for configuring zones in real-time,
including boundary adjustment, zone type selection, and live preview updates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.patches as patches
from matplotlib.widgets import PolygonSelector
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2

from premise_cv_platform.config.settings import settings
from premise_cv_platform.config.zone_config import (
    ZoneManager, Zone, ZoneType, create_default_zone_manager
)
from premise_cv_platform.utils.logging_config import get_zone_logger


class InteractiveZoneControls:
    """Interactive zone configuration controls for Streamlit dashboard."""
    
    def __init__(self):
        self.logger = get_zone_logger()
        
        # Initialize zone manager
        self.zone_manager = create_default_zone_manager(
            settings.get_line_zone_coordinates(),
            settings.get_teller_zone_coordinates()
        )
        
        # Initialize session state for zone configurations
        if "zone_configs" not in st.session_state:
            st.session_state.zone_configs = self._load_initial_configs()
        
        if "zone_preview_enabled" not in st.session_state:
            st.session_state.zone_preview_enabled = True
        
        if "zone_editor_mode" not in st.session_state:
            st.session_state.zone_editor_mode = "simple"  # simple or advanced
        
        self.canvas_width = 1280
        self.canvas_height = 720
        
        self.logger.info("InteractiveZoneControls initialized")
    
    def _load_initial_configs(self) -> Dict[str, Any]:
        """Load initial zone configurations from settings."""
        return {
            "line_zone": {
                "type": "line",
                "coordinates": self._parse_coordinates(settings.line_zone_points),
                "enabled": True,
                "color": "#00FFFF",  # Cyan
                "thickness": 3,
                "name": "Teller Access Line"
            },
            "teller_zone": {
                "type": "rectangle",
                "coordinates": self._parse_coordinates(settings.teller_zone_points),
                "enabled": True,
                "color": "#FF00FF",  # Magenta
                "thickness": 2,
                "name": "Teller Interaction Zone"
            },
            "abandonment_zone": {
                "type": "polygon",
                "coordinates": [(100, 400), (500, 400), (500, 600), (100, 600)],
                "enabled": False,
                "color": "#FF0000",  # Red
                "thickness": 2,
                "name": "Abandonment Detection Zone"
            }
        }
    
    def _parse_coordinates(self, coord_string: str) -> List[Tuple[int, int]]:
        """Parse coordinate string into list of tuples."""
        try:
            if not coord_string:
                return []
            
            # Handle comma-separated coordinate pairs
            coords = []
            pairs = coord_string.split()
            for pair in pairs:
                if ',' in pair:
                    x, y = map(int, pair.split(','))
                    coords.append((x, y))
            
            return coords if coords else [(0, 0), (100, 100)]
        except Exception as e:
            self.logger.error(f"Error parsing coordinates {coord_string}: {e}")
            return [(0, 0), (100, 100)]
    
    def render_zone_configuration_panel(self) -> Dict[str, Any]:
        """
        Render the main zone configuration panel.
        
        Returns:
            Updated zone configurations
        """
        st.header("🗺️ Interactive Zone Configuration")
        
        # Mode selector
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            editor_mode = st.selectbox(
                "Configuration Mode",
                ["simple", "advanced"],
                index=0 if st.session_state.zone_editor_mode == "simple" else 1,
                help="Simple mode for basic adjustments, Advanced for detailed control"
            )
            st.session_state.zone_editor_mode = editor_mode
        
        with col2:
            preview_enabled = st.checkbox(
                "Live Preview",
                value=st.session_state.zone_preview_enabled,
                help="Show zone overlays on video in real-time"
            )
            st.session_state.zone_preview_enabled = preview_enabled
        
        with col3:
            if st.button("🔄 Reset to Defaults", help="Reset all zones to default configuration"):
                st.session_state.zone_configs = self._load_initial_configs()
                st.success("Zone configuration reset to defaults")
                st.rerun()
        
        # Zone configuration tabs
        if editor_mode == "simple":
            return self._render_simple_zone_controls()
        else:
            return self._render_advanced_zone_controls()
    
    def _render_simple_zone_controls(self) -> Dict[str, Any]:
        """Render simple zone configuration controls."""
        st.subheader("📐 Simple Zone Configuration")
        
        # Configuration columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_line_zone_simple_controls()
        
        with col2:
            self._render_teller_zone_simple_controls()
        
        # Zone preview
        if st.session_state.zone_preview_enabled:
            self._render_zone_preview()
        
        # Save/Load controls
        self._render_save_load_controls()
        
        return st.session_state.zone_configs
    
    def _render_line_zone_simple_controls(self):
        """Render simple controls for line zone."""
        st.write("**🚶 Teller Access Line**")
        
        line_config = st.session_state.zone_configs["line_zone"]
        
        # Enable/disable toggle
        enabled = st.checkbox(
            "Enable Line Zone",
            value=line_config["enabled"],
            key="line_zone_enabled"
        )
        line_config["enabled"] = enabled
        
        if enabled:
            # Y position (horizontal line)
            y_pos = st.slider(
                "Line Y Position",
                min_value=100,
                max_value=600,
                value=line_config["coordinates"][0][1] if line_config["coordinates"] else 400,
                step=10,
                help="Vertical position of the teller access line",
                key="line_y_pos"
            )
            
            # X range
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                x_start = st.number_input(
                    "Start X",
                    min_value=0,
                    max_value=self.canvas_width,
                    value=300,
                    step=10,
                    key="line_x_start"
                )
            
            with col_x2:
                x_end = st.number_input(
                    "End X",
                    min_value=x_start,
                    max_value=self.canvas_width,
                    value=800,
                    step=10,
                    key="line_x_end"
                )
            
            # Update coordinates
            line_config["coordinates"] = [(x_start, y_pos), (x_end, y_pos)]
            
            # Visual properties
            line_config["color"] = st.color_picker(
                "Line Color",
                value=line_config["color"],
                key="line_color"
            )
            
            line_config["thickness"] = st.slider(
                "Line Thickness",
                min_value=1,
                max_value=10,
                value=line_config["thickness"],
                key="line_thickness"
            )
    
    def _render_teller_zone_simple_controls(self):
        """Render simple controls for teller zone."""
        st.write("**🏪 Teller Interaction Zone**")
        
        teller_config = st.session_state.zone_configs["teller_zone"]
        
        # Enable/disable toggle
        enabled = st.checkbox(
            "Enable Teller Zone",
            value=teller_config["enabled"],
            key="teller_zone_enabled"
        )
        teller_config["enabled"] = enabled
        
        if enabled:
            # Zone bounds
            col_x, col_y = st.columns(2)
            
            with col_x:
                x_min = st.number_input(
                    "Left X", 
                    min_value=0,
                    max_value=self.canvas_width//2,
                    value=400,
                    step=10,
                    key="teller_x_min"
                )
                
                x_max = st.number_input(
                    "Right X",
                    min_value=x_min,
                    max_value=self.canvas_width,
                    value=700,
                    step=10,
                    key="teller_x_max"
                )
            
            with col_y:
                y_min = st.number_input(
                    "Top Y",
                    min_value=0,
                    max_value=self.canvas_height//2,
                    value=100,
                    step=10,
                    key="teller_y_min"
                )
                
                y_max = st.number_input(
                    "Bottom Y",
                    min_value=y_min,
                    max_value=self.canvas_height,
                    value=300,
                    step=10,
                    key="teller_y_max"
                )
            
            # Update coordinates (rectangle)
            teller_config["coordinates"] = [
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max)
            ]
            
            # Visual properties
            teller_config["color"] = st.color_picker(
                "Zone Color",
                value=teller_config["color"],
                key="teller_color"
            )
            
            teller_config["thickness"] = st.slider(
                "Border Thickness",
                min_value=1,
                max_value=10,
                value=teller_config["thickness"],
                key="teller_thickness"
            )
    
    def _render_advanced_zone_controls(self) -> Dict[str, Any]:
        """Render advanced zone configuration controls."""
        st.subheader("🔧 Advanced Zone Configuration")
        
        # Zone selector
        zone_names = list(st.session_state.zone_configs.keys())
        selected_zone = st.selectbox(
            "Select Zone to Configure",
            zone_names,
            format_func=lambda x: st.session_state.zone_configs[x]["name"]
        )
        
        if selected_zone:
            self._render_advanced_zone_editor(selected_zone)
        
        # Global settings
        self._render_global_zone_settings()
        
        return st.session_state.zone_configs
    
    def _render_advanced_zone_editor(self, zone_key: str):
        """Render advanced editor for specific zone."""
        zone_config = st.session_state.zone_configs[zone_key]
        
        st.write(f"**Editing: {zone_config['name']}**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Coordinate editor
            st.write("Zone Coordinates:")
            
            # Add/remove coordinate points
            coords = zone_config["coordinates"]
            
            for i, (x, y) in enumerate(coords):
                coord_col1, coord_col2, coord_col3 = st.columns([1, 1, 0.3])
                
                with coord_col1:
                    new_x = st.number_input(
                        f"Point {i+1} X",
                        value=x,
                        min_value=0,
                        max_value=self.canvas_width,
                        step=1,
                        key=f"{zone_key}_x_{i}"
                    )
                
                with coord_col2:
                    new_y = st.number_input(
                        f"Point {i+1} Y",
                        value=y,
                        min_value=0,
                        max_value=self.canvas_height,
                        step=1,
                        key=f"{zone_key}_y_{i}"
                    )
                
                with coord_col3:
                    if st.button("🗑️", key=f"delete_{zone_key}_{i}", help="Delete point"):
                        if len(coords) > 2:  # Minimum 2 points
                            coords.pop(i)
                            st.rerun()
                
                # Update coordinate
                coords[i] = (new_x, new_y)
            
            # Add new point
            if st.button(f"➕ Add Point", key=f"add_point_{zone_key}"):
                coords.append((100, 100))
                st.rerun()
        
        with col2:
            # Zone properties
            st.write("Zone Properties:")
            
            zone_config["name"] = st.text_input(
                "Zone Name",
                value=zone_config["name"],
                key=f"{zone_key}_name"
            )
            
            zone_config["enabled"] = st.checkbox(
                "Enabled",
                value=zone_config["enabled"],
                key=f"{zone_key}_enabled_adv"
            )
            
            zone_config["type"] = st.selectbox(
                "Zone Type",
                ["line", "rectangle", "polygon"],
                index=["line", "rectangle", "polygon"].index(zone_config["type"]),
                key=f"{zone_key}_type"
            )
            
            zone_config["color"] = st.color_picker(
                "Color",
                value=zone_config["color"],
                key=f"{zone_key}_color_adv"
            )
            
            zone_config["thickness"] = st.slider(
                "Thickness",
                min_value=1,
                max_value=10,
                value=zone_config["thickness"],
                key=f"{zone_key}_thickness_adv"
            )
    
    def _render_global_zone_settings(self):
        """Render global zone settings."""
        st.subheader("🌐 Global Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Detection Thresholds**")
            
            # These would integrate with actual detection logic
            st.slider("Line Crossing Sensitivity", 0.1, 1.0, 0.5, 0.1)
            st.slider("Zone Dwell Time (seconds)", 1, 10, 2)
            st.slider("Minimum Person Size", 0.01, 0.1, 0.02, 0.01)
        
        with col2:
            st.write("**Visual Settings**")
            
            st.checkbox("Show Zone Labels", value=True)
            st.checkbox("Show Coordinate Grid", value=False)
            st.selectbox("Label Font Size", ["Small", "Medium", "Large"])
    
    def _render_zone_preview(self):
        """Render zone configuration preview."""
        st.subheader("🔍 Zone Preview")
        
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set canvas dimensions
            ax.set_xlim(0, self.canvas_width)
            ax.set_ylim(self.canvas_height, 0)  # Flip Y for image coordinates
            ax.set_aspect('equal')
            
            # Draw background grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Draw each zone
            for zone_key, zone_config in st.session_state.zone_configs.items():
                if not zone_config["enabled"]:
                    continue
                
                coords = zone_config["coordinates"]
                if not coords:
                    continue
                
                # Convert hex color to RGB
                color = zone_config["color"]
                
                if zone_config["type"] == "line":
                    # Draw line
                    if len(coords) >= 2:
                        xs, ys = zip(*coords)
                        ax.plot(xs, ys, color=color, linewidth=zone_config["thickness"],
                               label=zone_config["name"])
                
                elif zone_config["type"] == "rectangle":
                    # Draw rectangle
                    if len(coords) >= 4:
                        # Assume rectangle coordinates
                        x_min = min(coord[0] for coord in coords)
                        y_min = min(coord[1] for coord in coords)
                        width = max(coord[0] for coord in coords) - x_min
                        height = max(coord[1] for coord in coords) - y_min
                        
                        rect = Rectangle(
                            (x_min, y_min), width, height,
                            linewidth=zone_config["thickness"],
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.3,
                            label=zone_config["name"]
                        )
                        ax.add_patch(rect)
                
                elif zone_config["type"] == "polygon":
                    # Draw polygon
                    if len(coords) >= 3:
                        polygon = Polygon(
                            coords,
                            linewidth=zone_config["thickness"],
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.3,
                            label=zone_config["name"]
                        )
                        ax.add_patch(polygon)
                
                # Add zone labels
                if coords:
                    center_x = sum(coord[0] for coord in coords) / len(coords)
                    center_y = sum(coord[1] for coord in coords) / len(coords)
                    ax.text(center_x, center_y, zone_config["name"],
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_title("Zone Configuration Preview", fontsize=14, fontweight='bold')
            ax.set_xlabel("X Coordinate (pixels)")
            ax.set_ylabel("Y Coordinate (pixels)")
            ax.legend(loc='upper right')
            
            # Display in Streamlit
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error rendering zone preview: {e}")
            self.logger.error(f"Zone preview error: {e}")
    
    def _render_save_load_controls(self):
        """Render save/load configuration controls."""
        st.subheader("💾 Save/Load Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Save current configuration
            config_name = st.text_input(
                "Configuration Name",
                value=f"zone_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for saved configuration"
            )
            
            if st.button("💾 Save Configuration", type="primary"):
                self._save_zone_configuration(config_name)
        
        with col2:
            # Load saved configuration
            saved_configs = self._get_saved_configurations()
            
            if saved_configs:
                selected_config = st.selectbox(
                    "Saved Configurations",
                    saved_configs,
                    help="Select a saved configuration to load"
                )
                
                if st.button("📂 Load Configuration"):
                    if self._load_zone_configuration(selected_config):
                        st.success(f"Loaded configuration: {selected_config}")
                        st.rerun()
        
        with col3:
            # Export/Import
            if st.button("📤 Export as JSON"):
                config_json = json.dumps(st.session_state.zone_configs, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=config_json,
                    file_name=f"zone_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            uploaded_config = st.file_uploader(
                "📥 Import JSON",
                type="json",
                help="Upload a previously exported zone configuration"
            )
            
            if uploaded_config is not None:
                try:
                    config_data = json.loads(uploaded_config.getvalue())
                    st.session_state.zone_configs = config_data
                    st.success("Configuration imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing configuration: {e}")
    
    def _save_zone_configuration(self, config_name: str) -> bool:
        """Save current zone configuration."""
        try:
            # In a real implementation, this would save to a database or file
            # For now, we'll store in session state
            if "saved_zone_configs" not in st.session_state:
                st.session_state.saved_zone_configs = {}
            
            st.session_state.saved_zone_configs[config_name] = st.session_state.zone_configs.copy()
            st.success(f"Configuration '{config_name}' saved successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error saving configuration: {e}")
            self.logger.error(f"Error saving zone configuration {config_name}: {e}")
            return False
    
    def _load_zone_configuration(self, config_name: str) -> bool:
        """Load saved zone configuration."""
        try:
            if "saved_zone_configs" in st.session_state and config_name in st.session_state.saved_zone_configs:
                st.session_state.zone_configs = st.session_state.saved_zone_configs[config_name].copy()
                return True
            else:
                st.error(f"Configuration '{config_name}' not found")
                return False
                
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            self.logger.error(f"Error loading zone configuration {config_name}: {e}")
            return False
    
    def _get_saved_configurations(self) -> List[str]:
        """Get list of saved configurations."""
        if "saved_zone_configs" in st.session_state:
            return list(st.session_state.saved_zone_configs.keys())
        return []
    
    def get_current_zone_config(self) -> Dict[str, Any]:
        """Get current zone configuration."""
        return st.session_state.zone_configs.copy()
    
    def update_zone_config(self, zone_key: str, config: Dict[str, Any]):
        """Update specific zone configuration."""
        if zone_key in st.session_state.zone_configs:
            st.session_state.zone_configs[zone_key].update(config)
    
    def get_zone_overlay_data(self) -> Dict[str, Any]:
        """Get zone data formatted for video overlay."""
        overlay_data = {}
        
        for zone_key, zone_config in st.session_state.zone_configs.items():
            if zone_config["enabled"]:
                overlay_data[zone_key] = {
                    "coordinates": zone_config["coordinates"],
                    "type": zone_config["type"],
                    "color": self._hex_to_bgr(zone_config["color"]),
                    "thickness": zone_config["thickness"],
                    "name": zone_config["name"]
                }
        
        return overlay_data
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple for OpenCV."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Convert RGB to BGR for OpenCV
        return (rgb[2], rgb[1], rgb[0])


# Factory function
def create_zone_controls() -> InteractiveZoneControls:
    """Create InteractiveZoneControls instance."""
    return InteractiveZoneControls()