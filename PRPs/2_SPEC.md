# Specification Template (prompt inspired by IndyDevDan)

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Create a comprehensive Streamlit dashboard for real-time CV queue management visualization and video upload processing that transforms the existing command-line visualization into a web-based interface

## Mid-Level Objective

- Build a main Streamlit dashboard application with video streaming capabilities
- Implement video file upload functionality for on-demand analysis
- Create interactive zone configuration controls and real-time event logging
- Integrate existing CV pipeline (visualize_IDEAL.py) with web interface
- Add performance monitoring, statistics display, and processing status feedback

## Implementation Notes

- **Existing Codebase**: `visualize_IDEAL.py` contains complete working solution with `UpdatedZoneEventDetector` and `VisualUpdatedDetector`
- **Dependencies**: Streamlit, OpenCV, YOLO (yolo11n.pt), existing premise_cv_platform structure
- **Architecture**: Transform OpenCV display windows to Streamlit video components while maintaining real-time processing
- **Performance Considerations**: Memory management for video processing, UI responsiveness, frame rate optimization
- **Integration Points**: premise_cv_platform modules (config, data_ingestion, inference, reporting, storage, utils)
- **File Upload Requirements**: Support .MOV, .mp4, .mpg formats with validation and temporary file handling
- **Real-time Features**: Live event streaming, processing status updates, dynamic content refresh

## Context

### Beginning context

- `visualize_IDEAL.py` (complete working command-line visualization)
- `premise_cv_platform/` (existing modular structure)
- `videos/bank_sample.MOV` (test video file)
- `yolo11n.pt` (trained YOLO model)
- Working zone detection, person tracking, event logging system

### Ending context

- `premise_cv_platform/interface/streamlit_dashboard.py` (main dashboard application)
- `premise_cv_platform/interface/video_processor.py` (web-optimized video processing)
- `premise_cv_platform/interface/event_visualizer.py` (real-time event display)
- `premise_cv_platform/interface/controls.py` (interactive zone configuration)
- Updated `requirements.txt` with Streamlit dependencies
- Fully functional web dashboard with upload, processing, and visualization capabilities

## Low-Level Tasks

> Ordered from start to finish

1. Create main Streamlit dashboard application structure

```
CREATE premise_cv_platform/interface/streamlit_dashboard.py with:
- Main dashboard layout and navigation
- Video upload widget (st.file_uploader) with format validation
- Processing status display with progress indicators
- Real-time video streaming display area
- Integration with existing CV pipeline entry points
Validation: streamlit run premise_cv_platform/interface/streamlit_dashboard.py loads successfully
```

2. Implement web-optimized video processing module

```
CREATE premise_cv_platform/interface/video_processor.py that:
- Adapts visualize_IDEAL.py logic for Streamlit video display
- Handles uploaded video files and temporary storage
- Processes frames for web display (frame rate optimization)
- Integrates UpdatedZoneEventDetector for real-time analysis
- Provides processing status callbacks for UI updates
Validation: Video processing pipeline executes without blocking UI
```

3. Build interactive zone configuration controls

```
CREATE premise_cv_platform/interface/controls.py featuring:
- Interactive zone boundary adjustment widgets
- Zone type selection (Queue, Service, Abandonment)
- Real-time zone overlay updates on video display
- Configuration save/load functionality
- Integration with existing ZoneManager and zone_config
Validation: Zone controls update video display in real-time
```

4. Implement real-time event visualization system

```
CREATE premise_cv_platform/interface/event_visualizer.py that:
- Displays live event logging (LineEvent, TellerInteractionEvent, AbandonmentEvent)
- Shows event history with timestamps and details
- Provides event filtering and search capabilities
- Integrates with existing event storage and logging systems
- Updates event display in real-time during video processing
Validation: Events appear in dashboard as they occur during video processing
```

5. Add performance monitoring and statistics dashboard

```
MODIFY streamlit_dashboard.py to include:
- Performance metrics display (FPS, processing time, memory usage)
- Event statistics and counts by type
- Processing history and analytics
- System health monitoring indicators
- Integration with existing logging and metrics collection
Validation: Performance metrics update during video processing
```

6. Integrate video upload with processing pipeline

```
MODIFY streamlit_dashboard.py and video_processor.py to:
- Handle uploaded video file processing workflow
- Trigger main.py pipeline or equivalent processing functions
- Display processing progress and status updates
- Show analysis results after processing completion
- Implement error handling and user feedback for failed processing
Validation: Upload → Process → Display results workflow completes successfully
```

7. Update project dependencies and configuration

```
MODIFY requirements.txt to include:
- streamlit
- Additional dependencies for web interface
- Ensure compatibility with existing OpenCV, YOLO dependencies
CREATE or UPDATE configuration files for Streamlit deployment
Validation: pip install -r requirements.txt completes without conflicts
```