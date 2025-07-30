"""
PREMISE CV Platform Interface Module

This module provides web-based interfaces for real-time computer vision
analysis including Streamlit dashboard, video processing, and interactive controls.
"""

from .streamlit_dashboard import SimplePremiseDashboard
from .video_processor import StreamlitVideoProcessor
from .controls import InteractiveZoneControls
from .event_visualizer import RealTimeEventVisualizer

__all__ = [
    'SimplePremiseDashboard',
    'StreamlitVideoProcessor', 
    'InteractiveZoneControls',
    'RealTimeEventVisualizer'
]