"""
Real-time Event Visualization System

This module provides comprehensive real-time event visualization capabilities
including live event logging, filtering, search, and integration with existing
event storage systems.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import deque, defaultdict
import json
import time

from premise_cv_platform.storage.data_schemas import (
    Detection, LineEvent, TellerInteractionEvent, AbandonmentEvent, EventType
)
from premise_cv_platform.storage.csv_manager import CSVManager
from premise_cv_platform.config.settings import settings
from premise_cv_platform.utils.logging_config import get_zone_logger


class RealTimeEventVisualizer:
    """Real-time event visualization and monitoring system."""
    
    def __init__(self, max_live_events: int = 100):
        """
        Initialize event visualizer.
        
        Args:
            max_live_events: Maximum number of live events to keep in memory
        """
        self.logger = get_zone_logger()
        self.csv_manager = CSVManager()
        self.max_live_events = max_live_events
        
        # Initialize session state for live events
        if "live_events_buffer" not in st.session_state:
            st.session_state.live_events_buffer = deque(maxlen=max_live_events)
        
        if "event_filters" not in st.session_state:
            st.session_state.event_filters = {
                "event_types": ["all"],
                "person_ids": ["all"],
                "time_range": "last_hour",
                "zone_filter": "all"
            }
        
        if "event_stats" not in st.session_state:
            st.session_state.event_stats = {
                "total_events": 0,
                "events_by_type": defaultdict(int),
                "events_by_hour": defaultdict(int),
                "active_persons": set(),
                "last_update": datetime.now()
            }
        
        # Event type configurations
        self.event_type_config = {
            EventType.LINE_ENTERED: {
                "name": "Line Entered",
                "emoji": "🚶",
                "color": "#4CAF50",
                "priority": "medium"
            },
            EventType.LINE_EXITED: {
                "name": "Line Exited",
                "emoji": "🚪",
                "color": "#FF9800",
                "priority": "medium"
            },
            EventType.TELLER_INTERACTED: {
                "name": "Teller Interaction",
                "emoji": "✅",
                "color": "#2196F3",
                "priority": "high"
            },
            EventType.LEFT_LINE_NO_TELLER_INTERACTION: {
                "name": "Abandonment",
                "emoji": "❌",
                "color": "#F44336",
                "priority": "high"
            },
            "teller_zone_entered": {
                "name": "Teller Zone Entered",
                "emoji": "🏪",
                "color": "#9C27B0",
                "priority": "medium"
            },
            "teller_zone_exited": {
                "name": "Teller Zone Exited",
                "emoji": "🚶‍♂️",
                "color": "#607D8B",
                "priority": "low"
            }
        }
        
        self.logger.info("RealTimeEventVisualizer initialized")
    
    def add_live_event(self, event_data: Dict[str, Any]):
        """
        Add new event to live buffer.
        
        Args:
            event_data: Event data dictionary
        """
        # Enhance event data with metadata
        enhanced_event = {
            **event_data,
            "received_at": datetime.now(),
            "event_id": f"evt_{int(time.time()*1000)}_{len(st.session_state.live_events_buffer)}"
        }
        
        # Add to live buffer
        st.session_state.live_events_buffer.append(enhanced_event)
        
        # Update statistics
        self._update_event_stats(enhanced_event)
        
        self.logger.debug(f"Added live event: {enhanced_event.get('event_type', 'unknown')}")
    
    def _update_event_stats(self, event_data: Dict[str, Any]):
        """Update event statistics."""
        stats = st.session_state.event_stats
        
        # Update counters
        stats["total_events"] += 1
        
        event_type = event_data.get("event_type", "unknown")
        stats["events_by_type"][str(event_type)] += 1
        
        # Update hourly stats
        hour_key = datetime.now().strftime("%H:00")
        stats["events_by_hour"][hour_key] += 1
        
        # Track active persons
        person_id = event_data.get("person_id")
        if person_id:
            stats["active_persons"].add(person_id)
        
        stats["last_update"] = datetime.now()
    
    def render_live_event_monitor(self) -> None:
        """Render the main live event monitoring interface."""
        st.header("📡 Real-time Event Monitor")
        
        # Monitor controls
        self._render_monitor_controls()
        
        # Live event stream
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_live_event_stream()
        
        with col2:
            self._render_live_statistics()
        
        # Event timeline
        self._render_live_event_timeline()
        
        # Auto-refresh for live updates
        if st.session_state.get("auto_refresh_events", True):
            time.sleep(2)  # Refresh every 2 seconds
            st.rerun()
    
    def _render_monitor_controls(self):
        """Render monitoring control panel."""
        st.subheader("🎛️ Monitor Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auto_refresh = st.checkbox(
                "🔄 Auto Refresh",
                value=st.session_state.get("auto_refresh_events", True),
                help="Automatically refresh event display"
            )
            st.session_state.auto_refresh_events = auto_refresh
        
        with col2:
            if st.button("🗑️ Clear Buffer", help="Clear live event buffer"):
                st.session_state.live_events_buffer.clear()
                st.success("Event buffer cleared")
        
        with col3:
            if st.button("💾 Export Events", help="Export current events to CSV"):
                self._export_live_events()
        
        with col4:
            refresh_rate = st.selectbox(
                "Refresh Rate",
                [1, 2, 5, 10],
                index=1,
                help="Refresh interval in seconds"
            )
            st.session_state.refresh_rate = refresh_rate
    
    def _render_live_event_stream(self):
        """Render live event stream display."""
        st.subheader("🌊 Live Event Stream")
        
        # Event filters
        with st.expander("🔍 Filters", expanded=False):
            self._render_event_filters()
        
        # Get filtered events
        filtered_events = self._apply_event_filters()
        
        if not filtered_events:
            st.info("No events to display. Events will appear here as they occur.")
            return
        
        # Display events in reverse chronological order
        events_container = st.container()
        
        with events_container:
            for i, event in enumerate(reversed(filtered_events[-20:])):  # Last 20 events
                self._render_event_card(event, i)
    
    def _render_event_card(self, event_data: Dict[str, Any], index: int):
        """Render individual event card."""
        event_type = event_data.get("event_type", "unknown")
        person_id = event_data.get("person_id", "Unknown")
        timestamp = event_data.get("timestamp", event_data.get("received_at", datetime.now()))
        
        # Get event configuration
        event_config = self.event_type_config.get(event_type, {
            "name": str(event_type).replace("_", " ").title(),
            "emoji": "📝",
            "color": "#757575",
            "priority": "low"
        })
        
        # Format timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
        
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Choose card styling based on priority
        if event_config["priority"] == "high":
            card_style = "border-left: 4px solid #F44336; background-color: #FFEBEE; padding: 10px; margin: 5px 0; border-radius: 5px;"
        elif event_config["priority"] == "medium":
            card_style = "border-left: 4px solid #FF9800; background-color: #FFF3E0; padding: 10px; margin: 5px 0; border-radius: 5px;"
        else:
            card_style = "border-left: 4px solid #757575; background-color: #FAFAFA; padding: 10px; margin: 5px 0; border-radius: 5px;"
        
        # Render event card
        with st.container():
            st.markdown(f"""
            <div style="{card_style}">
                <strong>{event_config['emoji']} {event_config['name']}</strong><br>
                <small>👤 Person: {person_id} | ⏰ Time: {time_str}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional event details in expander
            with st.expander(f"Details for {event_config['name']} - {person_id}", expanded=False):
                event_details = {k: v for k, v in event_data.items() 
                               if k not in ["event_id", "received_at"]}
                st.json(event_details)
    
    def _render_event_filters(self):
        """Render event filtering controls."""
        filters = st.session_state.event_filters
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event type filter
            available_types = ["all"] + list(set(
                str(event.get("event_type", "unknown")) 
                for event in st.session_state.live_events_buffer
            ))
            
            selected_types = st.multiselect(
                "Event Types",
                available_types,
                default=filters["event_types"],
                help="Filter by event types"
            )
            filters["event_types"] = selected_types or ["all"]
            
            # Time range filter
            filters["time_range"] = st.selectbox(
                "Time Range",
                ["last_minute", "last_5_minutes", "last_hour", "all_time"],
                index=["last_minute", "last_5_minutes", "last_hour", "all_time"].index(filters["time_range"])
            )
        
        with col2:
            # Person ID filter
            available_persons = ["all"] + list(set(
                str(event.get("person_id", "unknown"))
                for event in st.session_state.live_events_buffer
                if event.get("person_id")
            ))
            
            selected_persons = st.multiselect(
                "Person IDs",
                available_persons,
                default=filters["person_ids"],
                help="Filter by person IDs"
            )
            filters["person_ids"] = selected_persons or ["all"]
            
            # Zone filter
            filters["zone_filter"] = st.selectbox(
                "Zone Filter",
                ["all", "line_zones", "teller_zones", "abandonment_zones"],
                index=["all", "line_zones", "teller_zones", "abandonment_zones"].index(filters["zone_filter"])
            )
    
    def _apply_event_filters(self) -> List[Dict[str, Any]]:
        """Apply current filters to event buffer."""
        filtered_events = list(st.session_state.live_events_buffer)
        filters = st.session_state.event_filters
        
        # Filter by event type
        if "all" not in filters["event_types"]:
            filtered_events = [
                event for event in filtered_events
                if str(event.get("event_type", "unknown")) in filters["event_types"]
            ]
        
        # Filter by person ID
        if "all" not in filters["person_ids"]:
            filtered_events = [
                event for event in filtered_events
                if str(event.get("person_id", "unknown")) in filters["person_ids"]
            ]
        
        # Filter by time range
        now = datetime.now()
        if filters["time_range"] != "all_time":
            time_deltas = {
                "last_minute": timedelta(minutes=1),
                "last_5_minutes": timedelta(minutes=5),
                "last_hour": timedelta(hours=1)
            }
            
            cutoff_time = now - time_deltas[filters["time_range"]]
            
            filtered_events = [
                event for event in filtered_events
                if event.get("received_at", now) >= cutoff_time
            ]
        
        # Filter by zone
        if filters["zone_filter"] != "all":
            zone_keywords = {
                "line_zones": ["line"],
                "teller_zones": ["teller"],
                "abandonment_zones": ["abandon"]
            }
            
            keywords = zone_keywords.get(filters["zone_filter"], [])
            filtered_events = [
                event for event in filtered_events
                if any(keyword in str(event.get("event_type", "")).lower() for keyword in keywords)
            ]
        
        return filtered_events
    
    def _render_live_statistics(self):
        """Render live event statistics."""
        st.subheader("📊 Live Statistics")
        
        stats = st.session_state.event_stats
        
        # Key metrics
        st.metric("Total Events", stats["total_events"])
        st.metric("Active Persons", len(stats["active_persons"]))
        
        # Last update
        last_update = stats["last_update"]
        time_ago = (datetime.now() - last_update).total_seconds()
        st.caption(f"Last update: {int(time_ago)}s ago")
        
        # Event distribution pie chart
        if stats["events_by_type"]:
            event_types = list(stats["events_by_type"].keys())
            event_counts = list(stats["events_by_type"].values())
            
            fig = px.pie(
                values=event_counts,
                names=event_types,
                title="Event Distribution"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_live_event_timeline(self):
        """Render live event timeline."""
        st.subheader("📈 Event Timeline")
        
        if not st.session_state.live_events_buffer:
            st.info("No events for timeline display")
            return
        
        # Prepare timeline data
        timeline_data = []
        for event in st.session_state.live_events_buffer:
            timestamp = event.get("received_at", datetime.now())
            event_type = event.get("event_type", "unknown")
            person_id = event.get("person_id", "unknown")
            
            timeline_data.append({
                "timestamp": timestamp,
                "event_type": str(event_type),
                "person_id": str(person_id),
                "count": 1
            })
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Group by time intervals
            df["time_bin"] = df["timestamp"].dt.floor("1min")
            
            # Aggregate events
            agg_df = df.groupby(["time_bin", "event_type"]).agg({"count": "sum"}).reset_index()
            
            # Create timeline chart
            fig = px.line(
                agg_df,
                x="time_bin",
                y="count",
                color="event_type",
                title="Live Event Timeline (1-minute intervals)",
                labels={"time_bin": "Time", "count": "Event Count"}
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Event Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_event_history_analysis(self):
        """Render comprehensive event history analysis."""
        st.header("📚 Event History Analysis")
        
        # Load historical data
        historical_data = self._load_historical_events()
        
        if not historical_data:
            st.warning("No historical event data available")
            return
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📈 Trends",
            "🔍 Deep Dive",
            "📋 Raw Data"
        ])
        
        with tab1:
            self._render_historical_overview(historical_data)
        
        with tab2:
            self._render_historical_trends(historical_data)
        
        with tab3:
            self._render_historical_deep_dive(historical_data)
        
        with tab4:
            self._render_historical_raw_data(historical_data)
    
    def _load_historical_events(self) -> Dict[str, pd.DataFrame]:
        """Load historical event data from CSV files."""
        historical_data = {}
        
        try:
            # Load different event types
            csv_dir = Path(settings.output_csv_dir)
            
            event_files = {
                "line_events": "line_events",
                "teller_interactions": "teller_interaction_events",
                "abandonment_events": "abandonment_events"
            }
            
            for event_type, file_prefix in event_files.items():
                pattern = f"{file_prefix}_*.csv"
                csv_files = list(csv_dir.glob(pattern))
                
                if csv_files:
                    # Get most recent file
                    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                    df = pd.read_csv(latest_file, parse_dates=["timestamp"])
                    historical_data[event_type] = df
                    
        except Exception as e:
            self.logger.error(f"Error loading historical events: {e}")
        
        return historical_data
    
    def _render_historical_overview(self, data: Dict[str, pd.DataFrame]):
        """Render historical data overview."""
        st.subheader("📊 Historical Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_line_events = len(data.get("line_events", pd.DataFrame()))
        total_teller_events = len(data.get("teller_interactions", pd.DataFrame()))
        total_abandonment_events = len(data.get("abandonment_events", pd.DataFrame()))
        total_events = total_line_events + total_teller_events + total_abandonment_events
        
        with col1:
            st.metric("Total Events", total_events)
        with col2:
            st.metric("Line Events", total_line_events)
        with col3:
            st.metric("Teller Interactions", total_teller_events)
        with col4:
            st.metric("Abandonment Events", total_abandonment_events)
        
        # Event distribution chart
        if total_events > 0:
            labels = ["Line Events", "Teller Interactions", "Abandonment Events"]
            values = [total_line_events, total_teller_events, total_abandonment_events]
            
            fig = px.pie(values=values, names=labels, title="Historical Event Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_historical_trends(self, data: Dict[str, pd.DataFrame]):
        """Render historical trend analysis."""
        st.subheader("📈 Historical Trends")
        
        # Combine all events for trend analysis
        combined_events = []
        
        for event_type, df in data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy["event_category"] = event_type
                combined_events.append(df_copy[["timestamp", "event_category"]])
        
        if combined_events:
            all_events_df = pd.concat(combined_events, ignore_index=True)
            
            # Hourly trend
            all_events_df["hour"] = all_events_df["timestamp"].dt.hour
            hourly_counts = all_events_df.groupby(["hour", "event_category"]).size().reset_index(name="count")
            
            fig = px.bar(
                hourly_counts,
                x="hour",
                y="count",
                color="event_category",
                title="Events by Hour of Day",
                labels={"hour": "Hour", "count": "Event Count"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_historical_deep_dive(self, data: Dict[str, pd.DataFrame]):
        """Render detailed historical analysis."""
        st.subheader("🔍 Deep Dive Analysis")
        
        # Event type selector for detailed analysis
        available_types = [k for k, v in data.items() if not v.empty]
        
        if not available_types:
            st.info("No data available for deep dive analysis")
            return
        
        selected_type = st.selectbox("Select Event Type for Analysis", available_types)
        
        if selected_type and selected_type in data:
            df = data[selected_type]
            
            # Person-level analysis
            if "person_id" in df.columns:
                person_counts = df["person_id"].value_counts().head(10)
                
                fig = px.bar(
                    x=person_counts.index,
                    y=person_counts.values,
                    title=f"Top 10 People by {selected_type.replace('_', ' ').title()}",
                    labels={"x": "Person ID", "y": "Event Count"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Temporal patterns
            if "timestamp" in df.columns:
                df["date"] = df["timestamp"].dt.date
                daily_counts = df.groupby("date").size()
                
                fig = px.line(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    title=f"Daily {selected_type.replace('_', ' ').title()} Trend",
                    labels={"x": "Date", "y": "Event Count"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_historical_raw_data(self, data: Dict[str, pd.DataFrame]):
        """Render raw historical data tables."""
        st.subheader("📋 Raw Historical Data")
        
        for event_type, df in data.items():
            if not df.empty:
                with st.expander(f"{event_type.replace('_', ' ').title()} ({len(df)} records)"):
                    st.dataframe(df.head(100), use_container_width=True)
                    
                    if len(df) > 100:
                        st.caption(f"Showing first 100 of {len(df)} records")
    
    def _export_live_events(self):
        """Export live events to CSV."""
        if not st.session_state.live_events_buffer:
            st.warning("No live events to export")
            return
        
        try:
            # Convert events to DataFrame
            events_data = []
            for event in st.session_state.live_events_buffer:
                events_data.append({
                    "timestamp": event.get("received_at", datetime.now()),
                    "event_type": event.get("event_type", "unknown"),
                    "person_id": event.get("person_id", "unknown"),
                    "event_data": json.dumps(event)
                })
            
            df = pd.DataFrame(events_data)
            
            # Create download
            csv_data = df.to_csv(index=False)
            filename = f"live_events_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
            
            st.success(f"Prepared {len(events_data)} events for export")
            
        except Exception as e:
            st.error(f"Error exporting events: {e}")
            self.logger.error(f"Event export error: {e}")
    
    def get_live_event_count(self) -> int:
        """Get current live event count."""
        return len(st.session_state.live_events_buffer)
    
    def clear_live_events(self):
        """Clear live event buffer."""
        st.session_state.live_events_buffer.clear()
        st.session_state.event_stats = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_hour": defaultdict(int),
            "active_persons": set(),
            "last_update": datetime.now()
        }


# Factory function
def create_event_visualizer(max_live_events: int = 100) -> RealTimeEventVisualizer:
    """Create RealTimeEventVisualizer instance."""
    return RealTimeEventVisualizer(max_live_events=max_live_events)