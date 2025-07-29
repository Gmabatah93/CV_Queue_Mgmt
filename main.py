"""
Main application entry point for PREMISE Computer Vision Platform.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import click
from rich.console import Console
from rich.progress import Progress, TaskID
from loguru import logger

# Import platform components
from premise_cv_platform.config.settings import settings
from premise_cv_platform.utils.logging_config import (
    setup_logger, log_system_info, log_configuration_summary, PerformanceTimer
)
from premise_cv_platform.data_ingestion.process_video import VideoProcessor, process_video_file
from premise_cv_platform.inference.track_people import PersonTracker
from premise_cv_platform.inference.zone_detector import ZoneEventDetector
from premise_cv_platform.storage.csv_manager import CSVManager
from premise_cv_platform.storage.data_schemas import ProcessingSummary

console = Console()


class PremiseCVPipeline:
    """Main pipeline orchestrator for end-to-end video processing."""
    
    def __init__(self, video_path: Optional[str] = None, output_dir: Optional[str] = None):
        self.video_path = video_path or settings.video_path
        self.output_dir = output_dir or settings.output_csv_dir
        
        # Initialize components
        self.video_processor = VideoProcessor(self.video_path)
        self.person_tracker = PersonTracker()
        self.zone_detector = ZoneEventDetector()
        self.csv_manager = CSVManager(self.output_dir)
        
        # Processing statistics
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_frames_processed = 0
        self.total_detections = 0
        
        logger.info(f"PremiseCVPipeline initialized for video: {self.video_path}")
    
    def process_video(self, save_visualization: bool = False) -> Dict[str, Any]:
        """
        Process video through complete pipeline: detection -> tracking -> zone analysis -> CSV export.
        """

        #1. Start timing and setup
        self.start_time = datetime.now()
        
        console.print(f"\n🏦 [bold blue]PREMISE Computer Vision Analytics[/bold blue]")
        console.print(f"Processing: {Path(self.video_path).name}")
        console.print(f"Output: {self.output_dir}\n")
        
        results = {
            "success": False,
            "video_info": {},
            "processing_stats": {},
            "events_generated": {},
            "files_created": [],
            "errors": []
        }
        
        try:
            #2. Open video file
            with self.video_processor as processor:
                # - get video information
                video_info = processor.get_video_info()
                results["video_info"] = video_info
                
                total_frames = video_info.get("properties", {}).get("total_frames", 0)
                
                # - setup progress tracking
                with Progress() as progress:
                    task = progress.add_task("[green]Processing frames...", total=total_frames)
                    
                    all_detections = []
                    all_events = []
                    
                    #3. Process each frame
                    for frame_number, frame, timestamp in processor.get_frame_generator():
                        #4. Detect people and track
                        detections = self.person_tracker.detect_and_track(frame, frame_number, timestamp)
                        all_detections.extend(detections)
                        
                        #5. Analyze zone events (line entries/exits, teller interactions, abandonment events)
                        zone_events = self.zone_detector.detect_zone_events(detections)
                        all_events.extend(zone_events)
                        
                        #6. Optional: Save visualization frame if requested
                        if save_visualization and settings.save_debug_frames:
                            if frame_number % settings.debug_frame_interval == 0:
                                vis_frame = self._create_visualization_frame(frame, detections)
                                vis_path = Path(settings.processed_video_dir) / f"frame_{frame_number:06d}_vis.jpg"
                                vis_path.parent.mkdir(parents=True, exist_ok=True)
                                processor.save_frame(vis_frame, str(vis_path), timestamp)
                        
                        self.total_frames_processed = frame_number + 1
                        progress.update(task, advance=1)
                        
                        # Cleanup tracking data periodically
                        if frame_number % 1000 == 0:
                            self.person_tracker.cleanup_old_tracks()
                            self.zone_detector.cleanup_old_tracking_data()
                
                # POST-PROCESSING Analysis ================================
                # - detect abandonment events after processing all frames
                console.print("\n[yellow]Analyzing abandonment events...[/yellow]")
                abandonment_events = self.zone_detector.detect_abandonment_events()
                
                # Export to CSV files
                console.print("[yellow]Exporting data to CSV files...[/yellow]")
                csv_files = self._export_to_csv()
                results["files_created"] = csv_files
                
                # Generate processing summary
                summary = self._generate_processing_summary(video_info, all_detections)
                summary_file = self.csv_manager.export_summary_to_csv(summary)
                results["files_created"].append(str(summary_file))
                
                # Collect final statistics
                results["processing_stats"] = {
                    "total_frames": self.total_frames_processed,
                    "total_detections": len(all_detections),
                    "unique_persons": len(self.person_tracker.tracking_states),
                    "processing_time": (datetime.now() - self.start_time).total_seconds(),
                    "fps_processed": self.total_frames_processed / (datetime.now() - self.start_time).total_seconds()
                }
                
                results["events_generated"] = {
                    "line_events": len(self.zone_detector.line_events),
                    "teller_interactions": len(self.zone_detector.teller_events),
                    "abandonment_events": len(abandonment_events)
                }
                
                results["success"] = True
                
                # Display summary
                self._display_summary(results)
                
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            results["errors"].append(str(e))
            console.print(f"\n[red]❌ Processing failed: {e}[/red]")
        
        finally:
            self.end_time = datetime.now()
        
        return results
    
    def _create_visualization_frame(self, frame, detections):
        """Create visualization frame with detections and zones."""
        # Draw zones
        vis_frame = self.zone_detector.visualize_zones_on_frame(frame)
        
        # Draw detections
        vis_frame = self.person_tracker.visualize_detections(vis_frame, detections)
        
        return vis_frame
    
    def _export_to_csv(self) -> List[str]:
        """Export all events to CSV files with exact schema matching."""
        csv_files = []
        
        # Export line events
        if self.zone_detector.line_events:
            line_file = self.csv_manager.write_line_events(self.zone_detector.line_events)
            csv_files.append(str(line_file))
            console.print(f"  ✓ Line events: {line_file}")
        
        # Export teller interaction events
        if self.zone_detector.teller_events:
            teller_file = self.csv_manager.write_teller_interaction_events(self.zone_detector.teller_events)
            csv_files.append(str(teller_file))
            console.print(f"  ✓ Teller interactions: {teller_file}")
        
        # Export abandonment events
        if self.zone_detector.abandonment_events:
            abandonment_file = self.csv_manager.write_abandonment_events(self.zone_detector.abandonment_events)
            csv_files.append(str(abandonment_file))
            console.print(f"  ✓ Abandonment events: {abandonment_file}")
        
        return csv_files
    
    def _generate_processing_summary(self, video_info: Dict, detections: list) -> ProcessingSummary:
        """Generate processing summary matching examples/summary_report_sample.txt format."""
        zone_stats = self.zone_detector.get_zone_statistics()
        tracking_stats = self.person_tracker.get_tracking_statistics()
        
        return ProcessingSummary(
            video_file=Path(self.video_path).name,
            processing_date=self.start_time or datetime.now(),
            total_frames=self.total_frames_processed,
            processing_duration=(self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            fps=video_info.get("properties", {}).get("fps", 0),
            total_detections=len(detections),
            unique_individuals=tracking_stats["unique_persons_tracked"],
            average_confidence=tracking_stats.get("tracking_quality", 0),
            line_entries=len([e for e in self.zone_detector.line_events if e.event_type.value == "line_entered"]),
            line_exits=len([e for e in self.zone_detector.line_events if e.event_type.value == "line_exited"]),
            teller_interactions=len(self.zone_detector.teller_events),
            abandonment_events=len(self.zone_detector.abandonment_events),
            zone_statistics=zone_stats
        )
    
    def _display_summary(self, results: Dict[str, Any]):
        """Display processing summary to console."""
        console.print("\n" + "="*50)
        console.print("[bold green]✅ PROCESSING COMPLETE[/bold green]")
        console.print("="*50)
        
        stats = results["processing_stats"]
        events = results["events_generated"]
        
        console.print(f"📊 [bold]Processing Statistics:[/bold]")
        console.print(f"   • Total frames processed: {stats['total_frames']:,}")
        console.print(f"   • Total detections: {stats['total_detections']:,}")
        console.print(f"   • Unique individuals: {stats['unique_persons']:,}")
        console.print(f"   • Processing time: {stats['processing_time']:.1f}s")
        console.print(f"   • Processing speed: {stats['fps_processed']:.1f} fps")
        
        console.print(f"\n🎯 [bold]Events Detected:[/bold]")
        console.print(f"   • Line entries/exits: {events['line_events']}")
        console.print(f"   • Teller interactions: {events['teller_interactions']}")
        console.print(f"   • Abandonment events: {events['abandonment_events']}")
        
        if events['line_events'] > 0:
            abandonment_rate = (events['abandonment_events'] / events['line_events']) * 100
            console.print(f"   • Abandonment rate: {abandonment_rate:.1f}%")
        
        console.print(f"\n📁 [bold]Files Created:[/bold]")
        for file_path in results["files_created"]:
            console.print(f"   • {file_path}")


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', type=str, help='Path to configuration file')
def cli(debug: bool, config: Optional[str]):
    """PREMISE Computer Vision Platform - Banking Analytics with AI."""
    if debug:
        settings.debug_mode = True
        settings.log_level = "DEBUG"
    
    # Initialize logging
    setup_logger()
    
    if debug:
        log_system_info()
        log_configuration_summary()


@cli.command()
@click.option('--video', type=str, help='Path to video file')
@click.option('--output', type=str, help='Output directory for CSV files')
@click.option('--visualize', is_flag=True, help='Save visualization frames')
@click.option('--benchmark', is_flag=True, help='Run in benchmark mode')
def process(video: Optional[str], output: Optional[str], visualize: bool, benchmark: bool):
    """Process video through complete CV pipeline."""
    
    # Ensure required directories exist
    settings.ensure_directories()
    
    # Initialize pipeline
    pipeline = PremiseCVPipeline(video, output)
    
    with PerformanceTimer("Complete video processing pipeline"):
        results = pipeline.process_video(save_visualization=visualize)
    
    if benchmark:
        console.print(f"\n[bold yellow]Benchmark Results:[/bold yellow]")
        stats = results["processing_stats"]
        console.print(f"Processing Rate: {stats['fps_processed']:.2f} fps")
        console.print(f"Memory Usage: {stats.get('memory_usage', 'N/A')}")
    
    # Exit with error code if processing failed
    if not results["success"]:
        sys.exit(1)


@cli.command()
@click.option('--port', default=8501, help='Dashboard port')
def dashboard(port):
    """Launch Streamlit dashboard."""
    import subprocess
    
    dashboard_script = "premise_cv_platform/reporting/streamlit_dashboard.py"
    
    if not Path(dashboard_script).exists():
        console.print("[red]❌ Dashboard script not found[/red]")
        sys.exit(1)
    
    console.print(f"🚀 [bold blue]Launching dashboard on port {port}...[/bold blue]")
    
    try:
        subprocess.run([
            "streamlit", "run", dashboard_script,
            "--server.port", str(port),
            "--server.address", settings.dashboard_host
        ])
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard shutdown[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Dashboard failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
def validate(video_path: str):
    """Validate video file compatibility."""
    from premise_cv_platform.data_ingestion.video_utils import validate_video_file
    
    console.print(f"🔍 [bold]Validating video file:[/bold] {video_path}")
    
    validation = validate_video_file(video_path)
    
    if validation['valid']:
        console.print("[green]✅ Video file is valid[/green]")
        
        props = validation['properties']
        console.print(f"\n📋 [bold]Video Properties:[/bold]")
        console.print(f"   • Resolution: {props['width']}x{props['height']}")
        console.print(f"   • FPS: {props['fps']}")
        console.print(f"   • Duration: {props.get('duration', 'Unknown')} seconds")
        console.print(f"   • Total frames: {props['frame_count']:,}")
    else:
        console.print("[red]❌ Video file validation failed[/red]")
        for error in validation['errors']:
            console.print(f"   • {error}")
        sys.exit(1)


@cli.command()
def info():
    """Display system and configuration information."""
    console.print("[bold blue]🏦 PREMISE Computer Vision Platform[/bold blue]")
    console.print("Banking Analytics with AI\n")
    
    # System information
    console.print("[bold]System Information:[/bold]")
    log_system_info()
    
    console.print("\n[bold]Configuration:[/bold]")
    log_configuration_summary()


@cli.command()
@click.option('--days', default=7, help='Days of data to clean up')
def cleanup(days: int):
    """Clean up old CSV files and logs."""
    csv_manager = CSVManager()
    
    console.print(f"🧹 [bold]Cleaning up files older than {days} days...[/bold]")
    
    # Clean CSV files
    csv_count = csv_manager.cleanup_old_files(days)
    console.print(f"   • Cleaned {csv_count} CSV files")
    
    # Clean log files (if implemented)
    console.print(f"✅ [green]Cleanup complete[/green]")


if __name__ == "__main__":
    cli()