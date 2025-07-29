"""
Streamlit dashboard for real-time PREMISE CV analytics visualization.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional

# Import platform components
from premise_cv_platform.config.settings import settings
from premise_cv_platform.storage.csv_manager import CSVManager
from premise_cv_platform.storage.data_schemas import ProcessingSummary
from premise_cv_platform.utils.async_processor import (
    processing_manager,
    ProcessingStatus,
)
from premise_cv_platform.utils.file_handler import TempFileManager


class PremiseDashboard:
    """Streamlit dashboard for real-time CV analytics."""

    def __init__(self):
        self.csv_manager = CSVManager()
        self.setup_page_config()

        # Initialize session state for upload and processing
        if "upload_processor" not in st.session_state:
            session_id = st.session_state.get(
                "session_id", f"session_{int(time.time())}"
            )
            st.session_state.session_id = session_id
            st.session_state.upload_processor = processing_manager.get_processor(
                session_id
            )
            st.session_state.processing_status = ProcessingStatus(
                status="idle", message="Ready for upload"
            )
            st.session_state.uploaded_file_info = None

    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="PREMISE Computer Vision Dashboard",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @st.cache_data(ttl=settings.dashboard_cache_ttl)
    def load_csv_data(_self, file_type: str) -> pd.DataFrame:
        """Load CSV data with caching for performance."""
        csv_dir = Path(settings.output_csv_dir)

        if not csv_dir.exists():
            return pd.DataFrame()

        # Find most recent CSV file of specified type
        pattern = f"{file_type}_*.csv"
        csv_files = list(csv_dir.glob(pattern))

        if not csv_files:
            return pd.DataFrame()

        # Get most recent file
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)

        try:
            df = pd.read_csv(latest_file, parse_dates=["timestamp"])
            return df
        except Exception as e:
            st.error(f"Error loading {file_type} data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=settings.dashboard_cache_ttl)
    def load_summary_data(_self) -> Optional[Dict[str, Any]]:
        """Load processing summary data."""
        csv_dir = Path(settings.output_csv_dir)

        # Find most recent summary file
        summary_files = list(csv_dir.glob("processing_summary_*.csv"))

        if not summary_files:
            return None

        latest_file = max(summary_files, key=lambda f: f.stat().st_mtime)

        try:
            df = pd.read_csv(latest_file)
            if len(df) > 0:
                return df.iloc[0].to_dict()
        except Exception as e:
            st.error(f"Error loading summary data: {e}")

        return None

    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with controls and configuration."""
        with st.sidebar:
            st.header("📊 Analytics Controls")

            # Refresh button
            if st.button("🔄 Refresh Data", type="primary"):
                st.cache_data.clear()
                st.rerun()

            st.subheader("⚙️ Configuration")

            # Time range selector
            time_range = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 4 Hours", "Last Day", "All Time"],
                index=2,
            )

            # Metrics to display
            st.subheader("📈 Metrics Display")
            show_line_events = st.checkbox("Line Events", value=True)
            show_teller_interactions = st.checkbox("Teller Interactions", value=True)
            show_abandonment = st.checkbox("Abandonment Events", value=True)

            # Zone configuration display
            st.subheader("🗺️ Zone Configuration")

            # Display current zone settings
            line_points = settings.get_line_zone_coordinates()
            teller_points = settings.get_teller_zone_coordinates()

            st.text_area(
                "Line Zone Points",
                value=settings.line_zone_points,
                help="Comma-separated x,y coordinates",
            )

            st.text_area(
                "Teller Zone Points",
                value=settings.teller_zone_points,
                help="Comma-separated x,y coordinates",
            )

            # Thresholds
            st.subheader("🎯 Alert Thresholds")
            queue_threshold = st.number_input(
                "Queue Length Alert",
                min_value=1,
                max_value=20,
                value=settings.queue_length_alert_threshold,
            )

            abandonment_threshold = st.number_input(
                "Abandonment Rate Alert (%)",
                min_value=1.0,
                max_value=100.0,
                value=settings.abandonment_rate_alert_threshold,
            )

        return {
            "time_range": time_range,
            "show_line_events": show_line_events,
            "show_teller_interactions": show_teller_interactions,
            "show_abandonment": show_abandonment,
            "queue_threshold": queue_threshold,
            "abandonment_threshold": abandonment_threshold,
        }

    def render_video_upload_interface(self) -> bool:
        """Render video upload interface and handle processing."""
        st.header("🎬 Video Upload & Analysis")

        # Check current processing status
        processor = st.session_state.upload_processor
        current_status = processor.get_current_status()

        # Update session state status
        st.session_state.processing_status = current_status

        # Upload section
        upload_col, status_col = st.columns([1, 1])

        with upload_col:
            st.subheader("📁 Upload Video File")

            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mov", "mp4", "mpg", "avi", "mkv"],
                help="Upload a video file for computer vision analysis",
                disabled=current_status.status == "processing",
            )

            # Start analysis button
            start_analysis = st.button(
                "🚀 Start Analysis",
                type="primary",
                disabled=uploaded_file is None or current_status.status == "processing",
                help="Begin video processing and analysis",
            )

            # Display upload file info
            if uploaded_file is not None:
                file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
                st.info(
                    f"📋 **File:** {uploaded_file.name}\n\n📏 **Size:** {file_size_mb:.1f} MB"
                )
                st.session_state.uploaded_file_info = {
                    "name": uploaded_file.name,
                    "size_mb": file_size_mb,
                }

        with status_col:
            self.render_processing_status(current_status)

        # Handle start analysis button click
        if start_analysis and uploaded_file is not None:
            try:
                st.info("🔄 Starting video analysis...")

                # Define progress callback
                def progress_callback(status: ProcessingStatus):
                    st.session_state.processing_status = status

                # Start processing
                processor.process_uploaded_video(
                    uploaded_file, progress_callback=progress_callback
                )

                # Trigger rerun to show processing status
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to start processing: {str(e)}")
                return False

        # Return True if processing completed successfully
        return current_status.status == "completed"

    def render_processing_status(self, status: ProcessingStatus) -> None:
        """Render real-time processing status display."""
        st.subheader("⚡ Processing Status")

        # Status indicator
        status_colors = {
            "idle": "gray",
            "uploading": "blue",
            "processing": "orange",
            "completed": "green",
            "error": "red",
        }

        status_icons = {
            "idle": "⏸️",
            "uploading": "📤",
            "processing": "⚙️",
            "completed": "✅",
            "error": "❌",
        }

        color = status_colors.get(status.status, "gray")
        icon = status_icons.get(status.status, "❓")

        st.markdown(f"**Status:** {icon} {status.status.title()}")
        st.markdown(f"**Message:** {status.message}")

        # Progress bar for processing
        if status.status == "processing":
            progress_bar = st.progress(status.progress / 100.0)
            st.caption(f"Progress: {status.progress:.1f}%")

            # Auto-refresh during processing
            if status.progress < 100:
                time.sleep(2)
                st.rerun()

        # Show processing time
        if status.start_time:
            if status.end_time:
                duration = (status.end_time - status.start_time).total_seconds()
                st.caption(f"⏱️ Processing time: {duration:.1f} seconds")
            else:
                current_duration = (datetime.now() - status.start_time).total_seconds()
                st.caption(f"⏱️ Processing for: {current_duration:.1f} seconds")

        # Show error details
        if status.status == "error" and status.error:
            st.error(f"Error details: {status.error}")

        # Show results summary
        if status.status == "completed" and status.results:
            results = status.results
            st.success("🎉 Processing completed successfully!")

            # Display key metrics
            if results.get("success"):
                stats = results.get("processing_stats", {})
                events = results.get("events_generated", {})

                st.markdown("**📊 Results Summary:**")
                st.write(f"• Frames processed: {stats.get('total_frames', 0):,}")
                st.write(f"• People detected: {stats.get('unique_persons', 0)}")
                st.write(f"• Line events: {events.get('line_events', 0)}")
                st.write(
                    f"• Teller interactions: {events.get('teller_interactions', 0)}"
                )
                st.write(f"• Abandonment events: {events.get('abandonment_events', 0)}")

                # Clear cache to show new results
                st.cache_data.clear()

        # Show real-time video processing with visualization
        if status.status == "processing" and status.video_file:
            self.render_live_video_analysis(status.video_file)

    def render_live_video_analysis(self, video_file_path: str) -> None:
        """Render live video analysis with zone overlays and detections."""
        st.subheader("🎬 Live Video Analysis")

        try:
            # Create a placeholder for the video frame
            video_placeholder = st.empty()

            # Get the latest processed frame with annotations
            latest_frame_path = self.get_latest_processed_frame()

            if latest_frame_path and Path(latest_frame_path).exists():
                # Display the annotated frame with timestamp
                frame_time = Path(latest_frame_path).stat().st_mtime
                readable_time = datetime.fromtimestamp(frame_time).strftime("%H:%M:%S")

                video_placeholder.image(
                    latest_frame_path,
                    caption=f"Live Processing: Detections and Zone Analysis (Frame captured at {readable_time})",
                    use_column_width=True,
                )

                # Auto-refresh to show new frames
                if st.session_state.processing_status.status == "processing":
                    time.sleep(2)  # Refresh every 2 seconds during processing
                    st.rerun()
            else:
                # Show video info while processing
                video_placeholder.info(
                    "🔄 Processing video frames... Visual analysis will appear here.\n\n"
                    "Debug frames will be saved and displayed as processing progresses."
                )

        except Exception as e:
            st.error(f"Error displaying live analysis: {e}")

    def get_latest_processed_frame(self) -> Optional[str]:
        """Get the path to the most recently processed frame with annotations."""
        try:
            # Look for debug frames in processed video directory
            debug_frame_dir = Path(settings.processed_video_dir)
            if not debug_frame_dir.exists():
                return None

            # Find the most recent visualization frame
            vis_frames = list(debug_frame_dir.glob("frame_*_vis.jpg"))
            if vis_frames:
                latest_frame = max(vis_frames, key=lambda f: f.stat().st_mtime)
                return str(latest_frame)

        except Exception:
            pass
        return None

    def render_zone_visualization(self) -> None:
        """Render zone configuration visualization."""
        st.subheader("🗺️ Zone Configuration Visualization")

        try:
            from matplotlib.patches import Polygon

            # Create a blank canvas for zone visualization
            fig, ax = plt.subplots(figsize=(10, 6))

            # Set canvas size (representative of video dimensions)
            canvas_width, canvas_height = 1920, 1080
            ax.set_xlim(0, canvas_width)
            ax.set_ylim(canvas_height, 0)  # Flip Y axis for image coordinates

            # Draw line zone
            line_coords = settings.get_line_zone_coordinates()
            if line_coords:
                line_polygon = Polygon(
                    line_coords,
                    alpha=0.3,
                    facecolor="green",
                    edgecolor="darkgreen",
                    linewidth=2,
                )
                ax.add_patch(line_polygon)
                ax.text(
                    line_coords[0][0],
                    line_coords[0][1],
                    "Line Zone",
                    fontsize=12,
                    fontweight="bold",
                    color="darkgreen",
                )

            # Draw teller zone
            teller_coords = settings.get_teller_zone_coordinates()
            if teller_coords:
                teller_polygon = Polygon(
                    teller_coords,
                    alpha=0.3,
                    facecolor="blue",
                    edgecolor="darkblue",
                    linewidth=2,
                )
                ax.add_patch(teller_polygon)
                ax.text(
                    teller_coords[0][0],
                    teller_coords[0][1],
                    "Teller Zone",
                    fontsize=12,
                    fontweight="bold",
                    color="darkblue",
                )

            ax.set_title("Zone Configuration Overlay", fontsize=14, fontweight="bold")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, alpha=0.3)

            # Display in Streamlit
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error rendering zone visualization: {e}")

    def render_main_metrics(self, summary_data: Optional[Dict[str, Any]]):
        """Render main KPI metrics."""
        st.title("🏦 PREMISE - Banking Computer Vision Analytics")

        if summary_data is None:
            st.warning(
                "No processing data available. Please run video processing first."
            )
            return

        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "👥 Total Individuals",
                value=summary_data.get("unique_individuals", 0),
                help="Number of unique people detected",
            )

        with col2:
            line_entries = summary_data.get("line_entries", 0)
            st.metric(
                "🚶 Entered Line",
                value=line_entries,
                help="Total number of people who entered the line",
            )

        with col3:
            teller_interactions = summary_data.get("teller_interactions", 0)
            st.metric(
                "✅ Teller Interactions",
                value=teller_interactions,
                help="Number of successful teller interactions",
            )

        with col4:
            abandonment_events = summary_data.get("abandonment_events", 0)
            abandonment_rate = (
                (abandonment_events / line_entries * 100) if line_entries > 0 else 0
            )

            st.metric(
                "❌ Abandonment Events",
                value=abandonment_events,
                delta=f"{abandonment_rate:.1f}% rate",
                delta_color="inverse",
                help="Number of people who left without teller interaction",
            )

    def render_timeline_chart(self, config: Dict[str, Any]):
        """Render timeline chart of events."""
        st.subheader("📈 Event Timeline")

        # Load event data
        line_events_df = self.load_csv_data("line_events")
        teller_events_df = self.load_csv_data("teller_interaction_events")
        abandonment_df = self.load_csv_data("abandonment_events")

        if line_events_df.empty and teller_events_df.empty and abandonment_df.empty:
            st.info("No event data available for timeline.")
            return

        # Combine all events for timeline
        timeline_data = []

        if config["show_line_events"] and not line_events_df.empty:
            for _, row in line_events_df.iterrows():
                timeline_data.append(
                    {
                        "timestamp": row["timestamp"],
                        "event_type": "Line "
                        + row["event_type"].replace("line_", "").title(),
                        "person_id": row["person_id"],
                        "count": 1,
                    }
                )

        if config["show_teller_interactions"] and not teller_events_df.empty:
            for _, row in teller_events_df.iterrows():
                timeline_data.append(
                    {
                        "timestamp": row["timestamp"],
                        "event_type": "Teller Interaction",
                        "person_id": row["person_id"],
                        "count": 1,
                    }
                )

        if config["show_abandonment"] and not abandonment_df.empty:
            for _, row in abandonment_df.iterrows():
                timeline_data.append(
                    {
                        "timestamp": row["timestamp"],
                        "event_type": "Abandonment",
                        "person_id": row["person_id"],
                        "count": 1,
                    }
                )

        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)

            # Group by time intervals for aggregation
            timeline_df["time_bin"] = pd.to_datetime(timeline_df["timestamp"]).dt.floor(
                "1min"
            )

            # Aggregate by time and event type
            agg_timeline = (
                timeline_df.groupby(["time_bin", "event_type"])
                .agg({"count": "sum"})
                .reset_index()
            )

            # Create timeline chart
            fig = px.line(
                agg_timeline,
                x="time_bin",
                y="count",
                color="event_type",
                title="Events Over Time (1-minute intervals)",
                labels={"time_bin": "Time", "count": "Event Count"},
            )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Number of Events",
                legend_title="Event Type",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events to display in timeline.")

    def render_zone_analysis(self):
        """Render zone-based analysis."""
        st.subheader("🗺️ Zone Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Line Zone Activity**")

            line_events_df = self.load_csv_data("line_events")

            if not line_events_df.empty:
                # Count events by type
                event_counts = line_events_df["event_type"].value_counts()

                fig = px.pie(
                    values=event_counts.values,
                    names=event_counts.index,
                    title="Line Zone Events Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No line zone data available.")

        with col2:
            st.write("**Teller Zone Activity**")

            teller_events_df = self.load_csv_data("teller_interaction_events")

            if not teller_events_df.empty:
                # Interactions by hour
                teller_events_df["hour"] = pd.to_datetime(
                    teller_events_df["timestamp"]
                ).dt.hour
                hourly_interactions = teller_events_df.groupby("hour").size()

                fig = px.bar(
                    x=hourly_interactions.index,
                    y=hourly_interactions.values,
                    title="Teller Interactions by Hour",
                    labels={"x": "Hour of Day", "y": "Interactions"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No teller interaction data available.")

    def render_abandonment_analysis(self):
        """Render detailed abandonment analysis."""
        st.subheader("⚠️ Abandonment Analysis")

        abandonment_df = self.load_csv_data("abandonment_events")

        if abandonment_df.empty:
            st.info("No abandonment events recorded.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Wait Time Before Abandonment**")

            # Calculate wait times
            abandonment_df["line_entered"] = pd.to_datetime(
                abandonment_df["line_entered_timestamp"]
            )
            abandonment_df["line_exited"] = pd.to_datetime(
                abandonment_df["line_exited_timestamp"]
            )
            abandonment_df["wait_time_minutes"] = (
                abandonment_df["line_exited"] - abandonment_df["line_entered"]
            ).dt.total_seconds() / 60

            fig = px.histogram(
                abandonment_df,
                x="wait_time_minutes",
                nbins=10,
                title="Distribution of Wait Times Before Abandonment",
                labels={
                    "wait_time_minutes": "Wait Time (minutes)",
                    "count": "Number of People",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            avg_wait = abandonment_df["wait_time_minutes"].mean()
            max_wait = abandonment_df["wait_time_minutes"].max()
            min_wait = abandonment_df["wait_time_minutes"].min()

            st.write(f"**Wait Time Statistics:**")
            st.write(f"- Average: {avg_wait:.1f} minutes")
            st.write(f"- Maximum: {max_wait:.1f} minutes")
            st.write(f"- Minimum: {min_wait:.1f} minutes")

        with col2:
            st.write("**Abandonment Timeline**")

            # Abandonment events over time
            abandonment_df["hour"] = pd.to_datetime(abandonment_df["timestamp"]).dt.hour
            hourly_abandonment = abandonment_df.groupby("hour").size()

            fig = px.bar(
                x=hourly_abandonment.index,
                y=hourly_abandonment.values,
                title="Abandonment Events by Hour",
                labels={"x": "Hour of Day", "y": "Abandonment Count"},
                color=hourly_abandonment.values,
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_data_table(self):
        """Render recent events data table."""
        st.subheader("📋 Recent Events")

        tab1, tab2, tab3 = st.tabs(
            ["Line Events", "Teller Interactions", "Abandonment Events"]
        )

        with tab1:
            line_df = self.load_csv_data("line_events")
            if not line_df.empty:
                # Show most recent events
                recent_line = line_df.sort_values("timestamp", ascending=False).head(10)
                st.dataframe(recent_line, use_container_width=True)
            else:
                st.info("No line events data available.")

        with tab2:
            teller_df = self.load_csv_data("teller_interaction_events")
            if not teller_df.empty:
                recent_teller = teller_df.sort_values(
                    "timestamp", ascending=False
                ).head(10)
                st.dataframe(recent_teller, use_container_width=True)
            else:
                st.info("No teller interaction data available.")

        with tab3:
            abandonment_df = self.load_csv_data("abandonment_events")
            if not abandonment_df.empty:
                recent_abandonment = abandonment_df.sort_values(
                    "timestamp", ascending=False
                ).head(10)
                st.dataframe(recent_abandonment, use_container_width=True)
            else:
                st.info("No abandonment events data available.")

    def render_summary_report(self, summary_data: Optional[Dict[str, Any]]):
        """Render summary report matching examples/summary_report_sample.txt format."""
        st.subheader("📄 Summary Report")

        if summary_data is None:
            st.info("No summary data available.")
            return

        # Generate report text
        report_text = f"""Video Processing Summary Report
-------------------------------
Video Processed: {summary_data.get('video_file', 'N/A')}
Processing Date: {summary_data.get('processing_date', 'N/A')}

Overall Metrics:
- Total unique individuals detected: {summary_data.get('unique_individuals', 0)}
- Total individuals who entered the line: {summary_data.get('line_entries', 0)}

Line & Interaction Metrics:
- Individuals who successfully interacted with a teller: {summary_data.get('teller_interactions', 0)}
- Individuals who left the line without teller interaction: {summary_data.get('abandonment_events', 0)}"""

        # Display in text area for easy copying
        st.text_area(
            "Report Content",
            value=report_text,
            height=300,
            help="Copy this report for external use",
        )

        # Download button
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name=f"premise_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    def render_system_status(self):
        """Render system status and health metrics."""
        st.subheader("🔧 System Status")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Data Files Status**")
            csv_stats = self.csv_manager.get_csv_file_stats()

            st.metric("Total CSV Files", csv_stats["total_files"])
            st.metric("Total Size (MB)", f"{csv_stats['total_size_mb']:.1f}")

            if csv_stats["file_types"]:
                st.write("File Types:")
                for file_type, count in csv_stats["file_types"].items():
                    st.write(f"- {file_type}: {count}")

        with col2:
            st.write("**Configuration**")
            st.write(f"- Model: {settings.model_name}")
            st.write(f"- GPU Enabled: {settings.gpu_enabled}")
            st.write(f"- Debug Mode: {settings.debug_mode}")
            st.write(f"- Face Recognition: {settings.enable_face_recognition}")
            st.write(f"- Data Retention: {settings.data_retention_days} days")

    def run(self):
        """Main dashboard application."""
        # Render sidebar and get configuration
        config = self.render_sidebar()

        # Render video upload interface at the top
        processing_completed = self.render_video_upload_interface()

        # Add separator
        st.markdown("---")

        # Load summary data (refresh if processing just completed)
        if processing_completed:
            st.cache_data.clear()
        summary_data = self.load_summary_data()

        # Main content area
        self.render_main_metrics(summary_data)

        # Zone visualization
        self.render_zone_visualization()

        # Charts and analysis
        self.render_timeline_chart(config)

        col1, col2 = st.columns(2)

        with col1:
            self.render_zone_analysis()

        with col2:
            self.render_abandonment_analysis()

        # Data tables
        self.render_data_table()

        # Summary report
        col1, col2 = st.columns(2)

        with col1:
            self.render_summary_report(summary_data)

        with col2:
            self.render_system_status()

        # Auto-refresh option
        st.sidebar.markdown("---")
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)

        if auto_refresh:
            import time

            time.sleep(30)
            st.rerun()


# Run the dashboard
if __name__ == "__main__":
    dashboard = PremiseDashboard()
    dashboard.run()
