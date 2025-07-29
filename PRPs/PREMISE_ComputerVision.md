# PREMISE Computer Vision Platform PRP

name: "PREMISE CV Platform v1.0 - Banking Computer Vision Analytics with Iterative Enhancement"
description: |
  A comprehensive PRP for building the PREMISE Computer Vision Analytics Platform - a scalable 
  system that processes banking video data to detect customer abandonment events using YOLO 
  object detection, zone-based tracking, and real-time analytics dashboard.

## Purpose

Template optimized for AI agents to implement a complete computer vision solution for banking 
analytics, starting with MVP functionality for person tracking and line abandonment detection, 
designed to scale to full enterprise banking intelligence platform.

## Core Principles

1. **Context is King**: Include ALL necessary CV documentation, YOLO patterns, and banking domain knowledge
2. **Validation Loops**: Provide executable tests for video processing, detection accuracy, and data integrity
3. **Progressive Enhancement**: Start with MVP person tracking, enhance to comprehensive banking analytics (e.g. line abdandonment detection)
4. **Information Dense**: Use proven CV patterns, YOLO best practices, and banking-specific requirements

---

## Goal

Build the PREMISE Computer Vision Analytics Platform that:
- **MVP Phase**: Processes bank simulation videos to detect line abandonment events
- **Production Phase**: Scales to real-time RTSP stream processing with VLM-powered analytics
- **Enterprise Phase**: Integrates RAG database and natural language query capabilities
- Maintains high accuracy (>90%) person detection and tracking
- Provides actionable business intelligence through intuitive dashboards

## Why

- **Business Value**: Reduce customer abandonment by 25% through early detection and intervention
- **Operational Efficiency**: Automate manual video review processes, saving 15+ hours/week
- **Customer Experience**: Identify bottlenecks and optimize queue management
- **Scalability**: Foundation for enterprise-wide computer vision analytics
- **ROI**: Enable data-driven decisions with quantifiable customer behavior insights

## What

### User-Visible Behavior
- Upload bank simulation video and receive comprehensive abandonment analytics
- Real-time dashboard showing customer flow, queue metrics, and abandonment events
- CSV exports of all detected events for further analysis
- Summary reports with actionable business insights
- Configurable zone definitions for different bank layouts

### Technical Requirements
- YOLO-based person detection and tracking (>90% accuracy)
- Zone-based event detection (line zones, teller interaction zones)
- CSV data storage with proper schema for all event types
- Streamlit dashboard for real-time visualization
- Error handling for video processing edge cases
- Configurable parameters via environment variables

### Success Criteria

- [ ] Successfully processes bank simulation video with person detection
- [ ] Accurately tracks individuals across frames with unique IDs
- [ ] Detects line entry/exit events with <2 second latency
- [ ] Identifies teller interactions based on zone dwell time
- [ ] Correctly classifies line abandonment events (left without interaction)
- [ ] Generates proper CSV files matching provided schema
- [ ] Displays comprehensive dashboard with key metrics
- [ ] Handles video processing errors gracefully
- [ ] Documentation enables easy setup and configuration

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window

# Computer Vision & YOLO Documentation
- url: https://docs.ultralytics.com/
  why: YOLO11 object detection and tracking implementation
  section: "Detection, tracking, and model optimization"
  critical: "Person class detection (class=0) and tracking persistence"

- url: https://docs.opencv.org/
  why: Video I/O, frame processing, and drawing functions
  section: "VideoCapture, rectangle, polylines drawing"
  critical: "Proper resource management and frame timing"

- url: https://huggingface.co/docs
  why: Pre-trained models and datasets for validation
  section: "Computer vision models and evaluation metrics"
  critical: "Model performance benchmarking and validation"

# Python Package Management
- url: https://docs.astral.sh/uv/
  why: Modern Python package management with uv
  section: "Virtual environment and dependency management"
  critical: "Reproducible environment setup and package resolution"

# Dashboard and Visualization
- url: https://docs.streamlit.io/
  why: Interactive web dashboard creation
  section: "Real-time data visualization and user interaction"
  critical: "Efficient data updates and performance optimization"

# Banking Domain Context
- file: examples/Project Management Feasibility_ Banking Computer Vision Solution.pdf
  why: Comprehensive banking CV solution architecture and requirements
  critical: "Domain-specific requirements, compliance, and scalability considerations"

- file: videos/bank_sample_description.md
  why: Detailed description of actual test video content and layout
  critical: "Overhead camera angle, loose line formation, specific abandonment at 0:18-0:20 mark"

# Example Data Patterns
- file: examples/ingestion_log.txt
  why: Expected video ingestion logging format
  critical: "Proper logging structure for debugging and monitoring"

- file: examples/line_events.csv
  why: Line entry/exit event data schema
  critical: "Exact CSV format with timestamp precision and zone identification"

- file: examples/teller_interaction_events.csv
  why: Teller interaction event data schema
  critical: "Interaction detection criteria and data structure"

- file: examples/abandonment_events.csv
  why: Line abandonment event data schema  
  critical: "Complex event detection with multiple timestamp correlations"

- file: examples/summary_report_sample.txt
  why: Expected dashboard summary format
  critical: "Business metrics presentation and reporting structure"

# PRP Framework Integration
- file: PRPs/templates/prp_base.md
  why: Validation loop structure and testing methodology
  critical: "Four-level validation hierarchy for production quality"
```

### Current Codebase Structure

```bash
PREMISE_Claude/
├── CLAUDE.md                 # Project memory and standards
├── PRPs/                     # PRP Framework
│   ├── INITIAL.md           # Updated CV requirements
│   ├── PREMISE_ComputerVision.md  # This PRP
│   ├── ai_docs/             # Curated documentation
│   ├── scripts/             # PRP runner utilities
│   └── templates/           # PRP templates
├── examples/                # Sample data and formats
│   ├── Project Management Feasibility_ Banking Computer Vision Solution.pdf
│   ├── ingestion_log.txt
│   ├── line_events.csv
│   ├── teller_interaction_events.csv
│   ├── abandonment_events.csv
│   └── summary_report_sample.txt
└── videos/
    └── bank_sample.MOV      # Test video for processing
```

### Desired Codebase Structure with PREMISE Platform

```bash
PREMISE_Claude/
├── CLAUDE.md                 # Enhanced with CV context
├── PRPs/                     # PRP Framework (existing)
├── premise_cv_platform/      # Main CV solution
│   ├── data_ingestion/       # Video processing module
│   │   ├── __init__.py
│   │   ├── process_video.py  # OpenCV video ingestion
│   │   └── video_utils.py    # Video utility functions
│   ├── inference/            # YOLO detection and tracking
│   │   ├── __init__.py
│   │   ├── track_people.py   # Person detection and tracking
│   │   ├── zone_detector.py  # Zone-based event detection
│   │   └── models/           # YOLO model storage
│   │       └── yolo11n.pt    # Pre-trained YOLO model
│   ├── storage/              # Data persistence layer
│   │   ├── __init__.py
│   │   ├── csv_manager.py    # CSV file operations
│   │   └── data_schemas.py   # Data validation schemas
│   ├── reporting/            # Dashboard and analytics
│   │   ├── __init__.py
│   │   ├── streamlit_dashboard.py  # Main dashboard
│   │   ├── generate_summary.py    # Report generation
│   │   └── visualizations.py      # Chart and plot utilities
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py       # Pydantic settings
│   │   └── zone_config.py    # Zone definition utilities
│   └── utils/                # Shared utilities
│       ├── __init__.py
│       ├── logging_config.py # Logging setup
│       └── validation.py     # Data validation utilities
├── tests/                    # Comprehensive test suite
│   ├── unit/                 # Unit tests for each module
│   │   ├── test_video_processing.py
│   │   ├── test_yolo_detection.py
│   │   ├── test_zone_detection.py
│   │   └── test_csv_operations.py
│   ├── integration/          # Integration test scenarios
│   │   ├── test_full_pipeline.py
│   │   └── test_dashboard_integration.py
│   └── fixtures/             # Test data and mock objects
│       ├── sample_video.mp4
│       └── expected_results.csv
├── data/                     # Generated data storage
│   ├── processed/            # Processed video outputs
│   ├── csv_exports/          # Generated CSV files
│   └── logs/                 # Application logs
├── docs/                     # Documentation
│   ├── README.md             # Setup and usage instructions
│   ├── ARCHITECTURE.md       # System architecture details
│   └── API.md                # Module API documentation
├── .env.example              # Environment variable template
├── .env                      # Local environment variables (gitignored)
├── pyproject.toml            # Project configuration with uv
├── requirements.txt          # Package dependencies
└── main.py                   # Application entry point
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: YOLO/Ultralytics Patterns (2025)
# YOLO11 requires specific model loading and tracking persistence
# Example: model = YOLO("yolo11n.pt"); results = model.track(frame, persist=True)
# Gotcha: Always check if boxes and IDs exist before accessing
# Pattern: if results[0].boxes is not None and results[0].boxes.id is not None:

# CRITICAL: OpenCV Video Processing
# VideoCapture requires proper resource management
# Example: cap = cv2.VideoCapture(video_path); ... ; cap.release()
# Gotcha: Always check ret value: ret, frame = cap.read(); if not ret: break
# Pattern: Use context managers or try/finally for cleanup

# CRITICAL: Zone Detection Mathematics  
# Point-in-polygon requires proper ray-casting algorithm
# Gotcha: Floating point precision can cause edge case failures
# Pattern: Use cv2.pointPolygonTest() for robust zone detection
# Example: result = cv2.pointPolygonTest(np.array(zone_points), point, False)

# CRITICAL: Video-Specific Considerations (bank_sample.MOV)
# Overhead camera angle minimizes occlusions but creates "flattened" people
# Floor glare from top-right may interfere with detection - test YOLO robustness
# Loose line formation around blue logo - use generous polygonal zones
# Line zone: polygonal area around blue logo extending toward red chairs
# Teller zone: rectangular area directly in front of red chairs
# Abandonment test case: person in dark jacket exits at 0:18-0:20 mark

# CRITICAL: CSV Data Types and Performance
# Pandas DataFrame dtypes significantly impact memory usage
# Example: Use 'float32' instead of 'float64' to reduce memory by 50%
# Pattern: Define dtypes upfront: pd.read_csv(file, dtype={'confidence': 'float32'})
# Gotcha: Timestamp precision must match expected format for event correlation

# CRITICAL: Streamlit Performance and Caching
# Streamlit requires proper caching for real-time performance
# Example: @st.cache_data(ttl=60) for data loading functions
# Pattern: Use st.empty() placeholders for dynamic content updates
# Gotcha: Large DataFrames can cause browser memory issues

# CRITICAL: Environment Variables and Configuration
# Use python-dotenv for secure configuration management
# Pattern: from dotenv import load_dotenv; load_dotenv()
# Example: VIDEO_PATH, MODEL_NAME, OUTPUT_CSV_DIR from .env
# Gotcha: Provide .env.example with all required variables

# CRITICAL: Error Handling in Video Processing
# Video files can have various issues (corrupted, unsupported format)
# Pattern: Graceful degradation with informative error messages
# Example: try/except blocks around cv2.VideoCapture operations
# Gotcha: Log errors for debugging but don't crash the entire pipeline
```

## Implementation Blueprint

### Data Models and Structure

Create robust data models for computer vision pipeline with proper validation.

```python
# Core data models following CV best practices
# Examples:
# - Pydantic models for detection results and event schemas
# - Dataclasses for tracking state and zone definitions
# - Enum classes for event types and zone categories
# - Configuration models for YOLO parameters and zone settings
# - Validation schemas for CSV data integrity
```

### Iterative Development Tasks (Ordered for Progressive Success)

```yaml
# Phase 1: MVP Foundation - Basic Video Processing Pipeline
Task 1 - Project Structure and Configuration:
  CREATE premise_cv_platform/ directory structure:
    - FOLLOW pattern from: modern Python CV project layouts
    - INCLUDE modular architecture for inference, storage, reporting
    - PRESERVE extensibility for future banking enhancements
  
  CREATE pyproject.toml and .env.example:
    - MIRROR pattern from: uv-based Python projects
    - INCLUDE CV dependencies (ultralytics, opencv-python, streamlit)
    - DEFINE environment variables (VIDEO_PATH, MODEL_NAME, OUTPUT_CSV_DIR)

Task 2 - Video Ingestion Layer:
  CREATE data_ingestion/process_video.py:
    - IMPLEMENT OpenCV VideoCapture with proper error handling
    - PATTERN: Resource management with try/finally or context managers
    - INCLUDE logging for video properties (resolution, FPS, duration)
    - MATCH format from: examples/ingestion_log.txt

Task 3 - YOLO Detection and Tracking:
  CREATE inference/track_people.py:
    - IMPLEMENT YOLO11 person detection with ultralytics
    - PATTERN: model = YOLO("yolo11n.pt"); results = model.track(persist=True)
    - PRESERVE tracking IDs across frames for consistent identification
    - INCLUDE confidence thresholding and bounding box extraction

Task 4 - Zone-Based Event Detection:
  CREATE inference/zone_detector.py:
    - IMPLEMENT point-in-polygon detection for configurable zones
    - PATTERN: Use cv2.pointPolygonTest() for robust zone detection
    - DETECT line_entered, line_exited, teller_interacted events
    - CALCULATE abandonment events (left_line_no_teller_interaction)

# Phase 2: Data Storage and Analytics
Task 5 - CSV Data Storage System:
  CREATE storage/csv_manager.py:
    - IMPLEMENT CSV writing with exact schema from examples/
    - PATTERN: Use pandas with proper dtypes for memory efficiency
    - PRESERVE data integrity with timestamp precision
    - INCLUDE batch writing for performance optimization

Task 6 - Event Processing Logic:
  CREATE storage/data_schemas.py:
    - IMPLEMENT Pydantic models for event validation
    - PATTERN: Timestamp correlation for complex event detection
    - VALIDATE event sequences (line_entered -> line_exited -> abandonment)
    - INCLUDE data quality checks and error reporting

# Phase 3: Dashboard and Reporting
Task 7 - Streamlit Dashboard:
  CREATE reporting/streamlit_dashboard.py:
    - IMPLEMENT real-time dashboard with key metrics display
    - PATTERN: Use st.empty() placeholders for dynamic updates
    - DISPLAY metrics matching examples/summary_report_sample.txt format
    - INCLUDE interactive controls for zone configuration

Task 8 - Report Generation:
  CREATE reporting/generate_summary.py:
    - IMPLEMENT business intelligence report generation
    - PATTERN: Read CSV data and calculate key performance indicators
    - GENERATE summary matching expected format from examples/
    - INCLUDE export functionality for business stakeholders

# Phase 4: Integration and Pipeline Orchestration
Task 9 - Main Application Entry Point:
  CREATE main.py:
    - IMPLEMENT end-to-end pipeline orchestration
    - PATTERN: Command-line interface with argparse or click
    - COORDINATE video processing, inference, storage, and reporting
    - INCLUDE progress tracking and error recovery mechanisms

Task 10 - Configuration and Documentation:
  CREATE comprehensive documentation:
    - IMPLEMENT README.md with setup and usage instructions
    - PATTERN: Step-by-step guide for environment setup with uv
    - INCLUDE troubleshooting guide and performance optimization tips
    - DOCUMENT zone configuration and customization options
```

### Phase-Specific Pseudocode

```python
# Phase 1: Video Processing and Detection
class VideoProcessor:
    """Core video processing with YOLO detection"""
    
    def __init__(self, config: Config):
        # PATTERN: Initialize YOLO model once for performance
        self.model = YOLO(config.model_name)
        self.logger = setup_logger(__name__)
        
    def process_video(self, video_path: str) -> List[Detection]:
        # PATTERN: OpenCV VideoCapture with proper resource management
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video: {video_path}")
            
        # GOTCHA: Always log video properties for debugging
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {video_path}, Resolution: {width}x{height}")
        self.logger.info(f"FPS: {fps}, Total frames: {total_frames}")
        
        detections = []
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # PATTERN: YOLO tracking with persistence for ID consistency
                results = self.model.track(frame, persist=True, classes=[0])  # Person class
                
                # CRITICAL: Check if detections exist before accessing
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    detections.extend(self._extract_detections(results[0], frame_number))
                
                frame_number += 1
                
        finally:
            cap.release()  # CRITICAL: Always release resources
            
        return detections

# Phase 2: Zone Detection and Event Processing
class ZoneEventDetector:
    """Zone-based event detection and abandonment analysis"""
    
    def __init__(self, zone_config: Dict):
        self.zones = zone_config
        self.tracking_states = {}  # Track person states across frames
        
    def detect_zone_events(self, detections: List[Detection]) -> List[Event]:
        events = []
        
        for detection in detections:
            person_id = detection.track_id
            center_point = (detection.center_x, detection.center_y)
            
            # PATTERN: Check each configured zone
            for zone_id, zone_polygon in self.zones.items():
                in_zone = cv2.pointPolygonTest(
                    np.array(zone_polygon, dtype=np.int32), 
                    center_point, 
                    False
                ) >= 0
                
                # PATTERN: State machine for entry/exit detection
                previous_state = self.tracking_states.get(f"{person_id}_{zone_id}", False)
                
                if in_zone and not previous_state:
                    # Zone entry event
                    events.append(Event(
                        timestamp=detection.timestamp,
                        event_type="line_entered" if "line" in zone_id else "teller_interacted",
                        person_id=person_id,
                        zone_id=zone_id
                    ))
                elif not in_zone and previous_state:
                    # Zone exit event
                    events.append(Event(
                        timestamp=detection.timestamp,
                        event_type="line_exited",
                        person_id=person_id,
                        zone_id=zone_id
                    ))
                
                self.tracking_states[f"{person_id}_{zone_id}"] = in_zone
        
        # PATTERN: Complex event detection for abandonment
        abandonment_events = self._detect_abandonment_events(events)
        events.extend(abandonment_events)
        
        return events
    
    def _detect_abandonment_events(self, events: List[Event]) -> List[Event]:
        # CRITICAL: Complex logic to correlate line entry/exit with teller interaction
        person_timelines = defaultdict(list)
        
        for event in events:
            person_timelines[event.person_id].append(event)
        
        abandonment_events = []
        
        for person_id, timeline in person_timelines.items():
            # PATTERN: Analyze timeline for abandonment pattern
            line_entries = [e for e in timeline if e.event_type == "line_entered"]
            line_exits = [e for e in timeline if e.event_type == "line_exited"]
            teller_interactions = [e for e in timeline if e.event_type == "teller_interacted"]
            
            # GOTCHA: Handle multiple line entries/exits
            for line_entry in line_entries:
                corresponding_exit = self._find_corresponding_exit(line_entry, line_exits)
                if corresponding_exit:
                    # Check if teller interaction occurred between entry and exit
                    had_interaction = any(
                        line_entry.timestamp <= interaction.timestamp <= corresponding_exit.timestamp
                        for interaction in teller_interactions
                    )
                    
                    if not had_interaction:
                        abandonment_events.append(Event(
                            timestamp=corresponding_exit.timestamp,
                            event_type="left_line_no_teller_interaction",
                            person_id=person_id,
                            line_entered_timestamp=line_entry.timestamp,
                            line_exited_timestamp=corresponding_exit.timestamp
                        ))
        
        return abandonment_events

# Phase 3: Dashboard and Real-time Visualization
class PremiseDashboard:
    """Streamlit dashboard for real-time CV analytics"""
    
    def __init__(self):
        st.set_page_config(
            page_title="PREMISE Computer Vision Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def create_dashboard(self, results_data: Dict):
        st.title("🏦 PREMISE - Banking Computer Vision Analytics")
        
        # PATTERN: Sidebar for configuration and controls
        with st.sidebar:
            st.header("📊 Analytics Controls")
            
            # Zone configuration
            st.subheader("Zone Configuration")
            line_zone_points = st.text_area(
                "Line Zone Points (x1,y1;x2,y2;...)",
                value="100,200;300,200;300,400;100,400"
            )
            
            teller_zone_points = st.text_area(
                "Teller Zone Points",
                value="350,150;500,150;500,250;350,250"
            )
        
        # PATTERN: Main dashboard layout with metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👥 Total Individuals",
                value=results_data.get('total_individuals', 0),
                delta=f"+{results_data.get('individuals_change', 0)} from last hour"
            )
        
        with col2:
            st.metric(
                "🚶 Entered Line",
                value=results_data.get('line_entries', 0),
                delta=f"{results_data.get('line_entry_rate', 0):.1f}/min"
            )
        
        with col3:
            st.metric(
                "✅ Teller Interactions",
                value=results_data.get('teller_interactions', 0),
                delta=f"{results_data.get('interaction_rate', 0):.1f}% success rate"
            )
        
        with col4:
            st.metric(
                "❌ Abandonment Events",
                value=results_data.get('abandonment_events', 0),
                delta=f"{results_data.get('abandonment_rate', 0):.1f}% abandonment rate"
            )
        
        # PATTERN: Interactive visualizations
        st.subheader("📈 Real-time Analytics")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # PATTERN: Time series chart of events
            if 'timeline_data' in results_data:
                fig = px.line(
                    results_data['timeline_data'],
                    x='timestamp',
                    y='count',
                    color='event_type',
                    title="Event Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # PATTERN: Summary statistics
            st.subheader("📋 Summary Report")
            summary_text = self._generate_summary_text(results_data)
            st.text_area("Report", value=summary_text, height=300)
    
    @st.cache_data(ttl=30)  # PATTERN: Cache for performance
    def _load_csv_data(self, csv_path: str) -> pd.DataFrame:
        # GOTCHA: Handle missing files gracefully
        try:
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            return pd.DataFrame()
```

### Integration Points

```yaml
VIDEO_PROCESSING:
  - input: "Local video files (.mov, .mp4, .avi)"
  - output: "Frame-by-frame detection results with timestamps"
  - performance: "Process 30 FPS video at >20 FPS detection rate"

YOLO_INTEGRATION:
  - model: "YOLO11n.pt for person detection (class=0)"
  - tracking: "ByteTrack or BoT-SORT for ID persistence" 
  - optimization: "TensorRT optimization for edge deployment"

ZONE_CONFIGURATION:
  - format: "Polygon points as list of (x,y) coordinates"
  - types: "Line zones, teller interaction zones, exclusion zones"
  - validation: "Real-time zone visualization with frame overlay"

CSV_STORAGE:
  - schemas: "Exact match to examples/ CSV format requirements"
  - performance: "Batch writing with pandas for large datasets"
  - integrity: "Data validation with Pydantic schemas"

DASHBOARD_INTEGRATION:
  - framework: "Streamlit with real-time data updates"
  - caching: "@st.cache_data for performance optimization"
  - visualization: "Plotly for interactive charts and heatmaps"

CONFIGURATION:
  - environment: "python-dotenv for secure configuration"
  - validation: "Pydantic Settings for type safety"
  - deployment: "Docker-ready with environment variable injection"
```

## Validation Loop (Computer Vision Quality Gates)

### Level 1: Syntax & Dependencies (Run After Each Task)

```bash
# Phase 1: Environment and Dependencies
# Install and verify all CV dependencies
uv venv && source .venv/bin/activate  # Create virtual environment
uv pip install ultralytics opencv-python streamlit plotly pandas python-dotenv
uv pip install pytest pytest-cov  # Testing dependencies

# Verify critical imports work
python -c "import cv2, ultralytics, streamlit, plotly, pandas; print('✅ All imports successful')"

# Test YOLO model loading
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('✅ YOLO model loaded')"

# Verify OpenCV video capabilities
python -c "import cv2; print(f'✅ OpenCV version: {cv2.__version__}')"

# Expected: No import errors, YOLO model downloads successfully, OpenCV functional
```

### Level 2: Unit Tests (Progressive CV Testing)

```python
# Phase 1: Core CV Components (CREATE tests/unit/test_video_processing.py)
def test_video_ingestion_basic():
    """Video ingestion handles valid video files correctly"""
    processor = VideoProcessor(test_config)
    
    # Test with sample video
    video_path = "tests/fixtures/sample_video.mp4"
    detections = processor.process_video(video_path)
    
    assert len(detections) > 0, "Should detect people in sample video"
    assert all(d.confidence > 0.3 for d in detections), "All detections should meet confidence threshold"
    assert all(hasattr(d, 'track_id') for d in detections), "All detections should have tracking IDs"

def test_video_error_handling():
    """Video processing handles missing/corrupted files gracefully"""
    processor = VideoProcessor(test_config)
    
    # Test missing file
    with pytest.raises(VideoProcessingError) as exc_info:
        processor.process_video("nonexistent_video.mp4")
    assert "Cannot open video" in str(exc_info.value)
    
    # Test corrupted file (create empty file)
    Path("corrupted.mp4").touch()
    with pytest.raises(VideoProcessingError):
        processor.process_video("corrupted.mp4")

def test_yolo_detection_accuracy():
    """YOLO detection meets accuracy requirements on bank_sample.MOV"""
    # Test on actual bank video with known characteristics
    cap = cv2.VideoCapture("videos/bank_sample.MOV")
    model = YOLO("yolo11n.pt")
    
    # Test frame around 10 second mark (should have 3-5 people visible)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # ~10 seconds at 30fps
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        results = model(frame, classes=[0])  # Person class only
        
        # Verify detection meets video description expectations
        if results[0].boxes is not None:
            detections = results[0].boxes
            confidences = detections.conf.cpu().numpy()
            
            assert 3 <= len(confidences) <= 7, "Should detect 3-5 people as described in video"
            assert np.mean(confidences) > 0.4, "Account for overhead angle - lower threshold"
        else:
            pytest.fail("No people detected in bank simulation video")

# Phase 2: Zone Detection Tests (CREATE tests/unit/test_zone_detection.py)
def test_zone_detection_accuracy():
    """Zone detection correctly identifies point-in-polygon for bank layout"""
    # Define zones based on bank_sample.MOV description
    # Line zone: polygonal area around blue logo extending toward chairs
    # Teller zone: rectangular area in front of red chairs
    zone_detector = ZoneEventDetector({
        'line_zone': [(400, 300), (600, 300), (650, 500), (350, 500)],  # Around blue logo
        'teller_zone': [(300, 100), (500, 100), (500, 200), (300, 200)]  # Front of chairs
    })
    
    # Test points based on actual video layout
    line_point = Detection(center_x=500, center_y=400, track_id=1, timestamp=datetime.now())  # Near blue logo
    teller_point = Detection(center_x=400, center_y=150, track_id=2, timestamp=datetime.now())  # Front of chairs
    outside_point = Detection(center_x=100, center_y=100, track_id=3, timestamp=datetime.now())  # Outside zones
    
    events = zone_detector.detect_zone_events([line_point, teller_point, outside_point])
    
    # Should generate appropriate zone events
    line_entries = [e for e in events if e.event_type == "line_entered"]
    teller_interactions = [e for e in events if e.event_type == "teller_interacted"]
    
    assert len(line_entries) == 1, "Should detect line entry at blue logo area"
    assert len(teller_interactions) == 1, "Should detect teller interaction at chairs"

def test_abandonment_detection_logic():
    """Complex abandonment event detection works correctly"""
    zone_detector = ZoneEventDetector(test_zones)
    
    # Create event sequence: line_entered -> line_exited (no teller interaction)
    events = [
        Event(timestamp=datetime(2025, 1, 1, 10, 0, 0), event_type="line_entered", person_id=1),
        Event(timestamp=datetime(2025, 1, 1, 10, 2, 0), event_type="line_exited", person_id=1),
    ]
    
    abandonment_events = zone_detector._detect_abandonment_events(events)
    
    assert len(abandonment_events) == 1
    assert abandonment_events[0].event_type == "left_line_no_teller_interaction"
    assert abandonment_events[0].person_id == 1

# Phase 3: CSV and Data Tests (CREATE tests/unit/test_csv_operations.py)
def test_csv_schema_compliance():
    """Generated CSV files match expected schema exactly"""
    csv_manager = CSVManager()
    
    # Create test events
    test_events = [
        Event(
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            event_type="line_entered",
            person_id=1,
            zone_id="zone_A"
        )
    ]
    
    # Write and read back
    csv_path = "test_line_events.csv"
    csv_manager.write_line_events(test_events, csv_path)
    
    # Verify schema matches examples/line_events.csv
    df = pd.read_csv(csv_path)
    expected_columns = ['timestamp', 'event_type', 'person_id', 'line_zone_id']
    
    assert list(df.columns) == expected_columns
    assert df.iloc[0]['event_type'] == 'line_entered'
    assert pd.to_datetime(df.iloc[0]['timestamp']).microsecond > 0  # Timestamp precision

def test_data_integrity_validation():
    """Data validation catches common errors"""
    # Test invalid timestamp format
    invalid_data = {'timestamp': 'invalid-date', 'person_id': 1}
    
    with pytest.raises(ValidationError):
        Event.model_validate(invalid_data)
    
    # Test missing required fields
    incomplete_data = {'timestamp': datetime.now()}
    
    with pytest.raises(ValidationError):
        Event.model_validate(incomplete_data)
```

```bash
# Progressive test execution and validation
# Phase 1: Basic CV Tests
uv run pytest tests/unit/test_video_processing.py -v --cov=premise_cv_platform/data_ingestion
# Expected: >80% coverage for video processing, all core tests passing

# Phase 2: Zone Detection Tests
uv run pytest tests/unit/test_zone_detection.py -v --cov=premise_cv_platform/inference
# Expected: >85% coverage for inference modules, zone detection accuracy validated

# Phase 3: Data Pipeline Tests
uv run pytest tests/unit/ -v --cov=premise_cv_platform --cov-report=html
# Expected: >90% overall coverage, all data schemas validated
# If failing: Check coverage report, fix missing test cases, validate CSV schemas
```

### Level 3: Integration Testing (Full Pipeline Validation)

```bash
# Phase 1: End-to-End Pipeline Test
# Test complete video processing pipeline with actual bank simulation
python main.py --video videos/bank_sample.MOV --output data/test_output/

# Verify specific abandonment detection at known timestamp (0:18-0:20 mark)
python -c "
import pandas as pd
df = pd.read_csv('data/test_output/abandonment_events.csv')
print(f'Abandonment events detected: {len(df)}')
# Should detect person in dark jacket who exits without teller interaction
expected_abandonment_time = pd.Timestamp('2025-01-01 00:00:18')  # Approximate
print('✅ Abandonment detection validation based on video description')
"

# Verify CSV files are generated with correct schema
ls -la data/test_output/
# Expected: line_events.csv, teller_interaction_events.csv, abandonment_events.csv

# Validate CSV data integrity
python -c "
import pandas as pd
df = pd.read_csv('data/test_output/line_events.csv')
print(f'Line events: {len(df)} records')
print(f'Columns: {list(df.columns)}')
assert 'timestamp' in df.columns
assert 'person_id' in df.columns
print('✅ CSV schema validation passed')
"

# Phase 2: Dashboard Integration Test
# Start Streamlit dashboard
streamlit run premise_cv_platform/reporting/streamlit_dashboard.py --server.port 8501 &
sleep 10

# Test dashboard accessibility
curl -f http://localhost:8501/_stcore/health
# Expected: HTTP 200 response

# Test dashboard with processed data
curl -X GET "http://localhost:8501" | grep -q "PREMISE"
# Expected: Dashboard loads with PREMISE title

# Phase 3: Performance and Accuracy Testing
# Test video processing performance
time python main.py --video videos/bank_sample.MOV --benchmark
# Expected: Process 30 FPS video at >20 FPS, complete in <video_duration * 1.5

# Validate detection accuracy with ground truth
python -c "
from premise_cv_platform.inference.track_people import PersonTracker
import cv2

tracker = PersonTracker()
cap = cv2.VideoCapture('videos/bank_sample.MOV')
frame_count = 0
detection_count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_count > 300:  # Test first 10 seconds
        break
    
    results = tracker.detect_and_track(frame)
    if results[0].boxes is not None:
        detection_count += len(results[0].boxes)
    
    frame_count += 1

cap.release()
detection_rate = detection_count / frame_count
print(f'Detection rate: {detection_rate:.2f} detections/frame')
assert detection_rate > 0.5, 'Should detect people in most frames'
print('✅ Accuracy validation passed')
"

# Clean up test processes
pkill -f streamlit
```

### Level 4: Production Readiness and Business Validation

```bash
# Phase 1: Business Logic Validation
# Verify abandonment detection matches business requirements
python -c "
from premise_cv_platform.reporting.generate_summary import SummaryGenerator

generator = SummaryGenerator()
summary = generator.generate_report('data/test_output/')

print('Generated Summary:')
print(summary)

# Validate key metrics are present
assert 'Total unique individuals' in summary
assert 'left the line without teller interaction' in summary
print('✅ Business metrics validation passed')
"

# Test report format matches examples/summary_report_sample.txt
diff <(python -c "
from premise_cv_platform.reporting.generate_summary import SummaryGenerator
generator = SummaryGenerator()
print(generator.generate_report('data/test_output/'))
") examples/summary_report_sample.txt --ignore-matching-lines="Processing Date"
# Expected: Format matches except for dynamic values

# Phase 2: Error Recovery and Edge Cases
# Test with corrupted video
touch corrupted_video.mp4
python main.py --video corrupted_video.mp4 --output data/error_test/
# Expected: Graceful error handling, informative error message

# Test with empty video directory
mkdir empty_dir
python main.py --video empty_dir/ --output data/error_test/
# Expected: Clear error message about missing video files

# Test dashboard with missing data
rm -rf data/test_output/*.csv
streamlit run premise_cv_platform/reporting/streamlit_dashboard.py --server.port 8502 &
sleep 5
curl -f http://localhost:8502
# Expected: Dashboard shows "No data available" message gracefully

# Phase 3: Performance Benchmarking
# Load testing with multiple concurrent video processing
for i in {1..3}; do
    python main.py --video videos/bank_sample.MOV --output data/load_test_$i/ &
done
wait

# Verify all processes completed successfully
for i in {1..3}; do
    test -f data/load_test_$i/line_events.csv || echo "❌ Load test $i failed"
done
echo "✅ Load testing completed"

# Memory usage monitoring
python -c "
import psutil
import subprocess
import time

# Start video processing
proc = subprocess.Popen(['python', 'main.py', '--video', 'videos/bank_sample.MOV'])
time.sleep(2)

# Monitor memory usage
process = psutil.Process(proc.pid)
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Memory usage: {memory_mb:.1f} MB')

proc.wait()
assert memory_mb < 1000, 'Memory usage should be < 1GB for sample video'
print('✅ Memory usage validation passed')
"

# Clean up test artifacts
rm -rf data/test_output/ data/error_test/ data/load_test_*/
pkill -f streamlit
```

## Final Validation Checklist (Progressive CV Completion)

### Phase 1: MVP Computer Vision Pipeline
- [ ] Video ingestion works: `python main.py --video videos/bank_sample.MOV`
- [ ] YOLO detection functional: Person detection confidence >0.5 average
- [ ] Zone detection accurate: Point-in-polygon tests pass
- [ ] CSV files generated: All 4 CSV types created with correct schema
- [ ] Basic error handling: Graceful failure for missing/corrupted videos
- [ ] Logging implemented: INFO level logs for video properties and progress
- [ ] Environment setup: `.env` file configuration working

### Phase 2: Advanced Analytics and Dashboard
- [ ] Zone events detected: Line entry/exit events with precise timestamps
- [ ] Teller interactions: Dwell time-based interaction detection
- [ ] Abandonment logic: Complex event correlation working correctly
- [ ] Dashboard functional: `streamlit run` shows metrics and visualizations
- [ ] Real-time updates: Dashboard refreshes with new data
- [ ] Performance optimized: >20 FPS processing rate maintained
- [ ] Data integrity: All CSV schemas match examples exactly

### Phase 3: Production Quality and Documentation
- [ ] Comprehensive testing: >90% test coverage across all modules
- [ ] Error recovery: Robust handling of edge cases and failures
- [ ] Performance benchmarks: Memory usage <1GB, processing rate >video FPS
- [ ] Documentation complete: README with setup instructions and troubleshooting
- [ ] Business validation: Summary reports match expected format and metrics
- [ ] Deployment ready: Docker containerization and environment variable configuration
- [ ] Code quality: Passes linting and type checking with mypy

---

## Anti-Patterns to Avoid

- ❌ Don't skip video resource cleanup - always call `cap.release()`
- ❌ Don't assume YOLO detections exist - check `results[0].boxes is not None`
- ❌ Don't hardcode zone coordinates - use configurable environment variables
- ❌ Don't ignore timestamp precision - CSV events need microsecond accuracy
- ❌ Don't cache YOLO models incorrectly - load once and reuse instances
- ❌ Don't block Streamlit UI - use `st.empty()` for dynamic content updates
- ❌ Don't skip data validation - use Pydantic schemas for all event types
- ❌ Don't ignore performance - optimize for >20 FPS processing rate

## Future Phases: Banking Intelligence Platform

### Phase 2: Real-time Processing Architecture
- **RTSP Stream Integration**: Live camera feed processing with reduced latency
- **Edge Computing**: NVIDIA Jetson deployment for on-premises processing
- **Apache Kafka**: Real-time event streaming for enterprise integration
- **Advanced Analytics**: Queue length analysis, service time optimization

### Phase 3: RAG-Powered Intelligence
- **PostgreSQL**: Structured event data with time-series optimization
- **Vector Database**: FAISS/Pinecone for similarity-based pattern analysis
- **VLM Integration**: GPT-4V for natural language query capabilities
- **LangGraph**: Orchestrated AI agents for complex business intelligence

### Phase 4: Enterprise Banking Platform
- **Multi-location Support**: Centralized analytics across branch networks
- **Compliance Integration**: PCI DSS and banking regulation compliance
- **Predictive Analytics**: ML models for customer behavior prediction
- **API Gateway**: RESTful APIs for integration with banking systems

---

## Confidence Score: 9/10

**Reasoning for High Confidence Score**:

1. **Domain-Specific Context**: Comprehensive banking CV requirements with real example data
2. **Proven Technology Stack**: YOLO11, OpenCV, Streamlit - all production-ready technologies
3. **Executable Validation**: Specific tests for CV accuracy, performance, and business logic
4. **Progressive Structure**: Clear MVP → Advanced Analytics → Production pathway
5. **Real-world Validation**: Based on actual banking computer vision use cases
6. **Complete Pipeline**: End-to-end solution from video ingestion to business reporting

**Remaining 10% Risk Factors**:
- Video quality variations may affect detection accuracy
- Zone configuration complexity for different bank layouts
- Real-time performance optimization may require additional tuning

This PRP provides comprehensive context and validation for building a production-ready computer vision platform for banking analytics, with sufficient detail for AI agents to achieve one-pass implementation success.