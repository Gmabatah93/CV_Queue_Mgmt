name: "Video Upload and Dashboard Integration PRP"
description: |
  Comprehensive specification-driven PRP for implementing video upload functionality
  in the PREMISE Computer Vision Platform Streamlit dashboard with complete pipeline integration.

---

## Goal

Implement video upload functionality directly in the `streamlit_dashboard.py` that allows users to:
1. Upload local video files through a web interface
2. Trigger the complete PREMISE CV processing pipeline on uploaded videos 
3. View real-time processing status and feedback
4. Display analysis results automatically upon completion
5. Handle errors gracefully with informative feedback

**End State**: A fully integrated dashboard where users can upload videos, monitor processing, and view results without using CLI commands.

## Why

- **User Experience**: Eliminates need for CLI interaction, making the system accessible to non-technical users
- **Operational Efficiency**: Streamlines video analysis workflow from upload to results viewing
- **Business Value**: Enables faster decision-making with immediate access to computer vision analytics
- **Integration Benefits**: Seamlessly connects existing processing pipeline with web-based interface
- **MVP Foundation**: Establishes core functionality for future dashboard enhancements

## What

### User-Visible Behavior
- File upload component in dashboard accepting common video formats (.MOV, .mp4, .mpg)
- Real-time processing status indicators (spinner, progress bar, status messages)
- Automatic display of analysis results (metrics, charts, CSV data) upon completion
- Error messaging and handling for upload/processing failures
- Processing logs visible during execution

### Technical Requirements
- Integration with existing `main.py` PremiseCVPipeline
- Temporary file handling for uploaded videos
- Asynchronous processing to prevent dashboard freezing
- Real-time status updates during video processing
- Automatic data refresh after processing completion

### Success Criteria
- [ ] Users can upload video files via dashboard interface
- [ ] Upload validation accepts only supported video formats  
- [ ] Processing pipeline executes successfully on uploaded videos
- [ ] Real-time feedback shows processing progress
- [ ] Results display automatically after processing completion
- [ ] Error states provide clear user feedback
- [ ] Dashboard remains responsive during processing
- [ ] Temporary files are cleaned up after processing

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
  why: Core file upload widget implementation patterns

- url: https://docs.streamlit.io/library/api-reference/status/st.progress  
  why: Progress indicators and status display methods

- file: /Users/isiomamabatah/Desktop/Cursor/PREMISE_Claude/main.py
  why: PremiseCVPipeline class and process_video method - core processing logic to integrate

- file: /Users/isiomamabatah/Desktop/Cursor/PREMISE_Claude/premise_cv_platform/reporting/streamlit_dashboard.py
  why: Existing dashboard structure, methods, and patterns to follow

- file: /Users/isiomamabatah/Desktop/Cursor/PREMISE_Claude/examples/sample_app.py
  why: Reference for file upload widget and processing status patterns (FRONTEND ONLY)

- docfile: PRPs/ai_docs/cc_common_workflows.md
  why: Claude Code workflow patterns for complex implementations
```

### Current State Assessment

**Current Implementation:**
```yaml
files_affected:
  - premise_cv_platform/reporting/streamlit_dashboard.py: Dashboard exists but no upload functionality
  - main.py: PremiseCVPipeline class with complete processing pipeline
  - premise_cv_platform/data_ingestion/process_video.py: Video processing logic
  - premise_cv_platform/storage/csv_manager.py: CSV export functionality

behavior: 
  - Dashboard displays pre-processed CSV data from files
  - Video processing only available via CLI: python main.py process --video path/to/video
  - Results require manual dashboard refresh after CLI processing

issues:
  - No video upload interface in dashboard
  - No integration between dashboard and processing pipeline  
  - Manual workflow requiring CLI knowledge
  - No real-time processing feedback
  - No automatic result updates
```

**Desired State:**
```yaml
files_expected:
  - premise_cv_platform/reporting/streamlit_dashboard.py: Enhanced with upload interface, processing integration, status tracking
  - premise_cv_platform/utils/file_handler.py: [NEW] Temporary file management utilities
  - premise_cv_platform/utils/async_processor.py: [NEW] Async processing wrapper for dashboard integration

behavior:
  - Dashboard includes prominent file upload section
  - Click "Start Analysis" triggers processing pipeline directly
  - Real-time status updates during processing
  - Automatic result display upon completion
  - Error handling with user-friendly messages

benefits:
  - Zero CLI knowledge required for users
  - Immediate feedback and results
  - Streamlined analytics workflow
  - Better user adoption and accessibility
```

### Current Codebase Structure

```bash
premise_cv_platform/
├── config/
│   ├── settings.py              # Configuration management
│   └── zone_config.py
├── data_ingestion/
│   ├── process_video.py         # VideoProcessor class
│   └── video_utils.py           # Video validation utilities
├── inference/
│   ├── track_people.py          # PersonTracker
│   └── zone_detector.py         # ZoneEventDetector
├── reporting/
│   └── streamlit_dashboard.py   # [MAIN TARGET] Dashboard to enhance
├── storage/
│   ├── csv_manager.py           # CSVManager for data export
│   └── data_schemas.py          # Data models
└── utils/
    └── logging_config.py        # Logging utilities
main.py                          # [INTEGRATION TARGET] PremiseCVPipeline
```

### Desired Codebase Structure

```bash
premise_cv_platform/
├── utils/
│   ├── logging_config.py        
│   ├── file_handler.py          # [NEW] Temporary file management
│   └── async_processor.py       # [NEW] Async processing wrapper
├── reporting/
│   └── streamlit_dashboard.py   # [ENHANCED] Upload interface + processing integration
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: Streamlit requires specific patterns for file uploads
# Example: st.file_uploader returns UploadedFile object, not direct file path
uploaded_file = st.file_uploader("Choose file", type=['mov', 'mp4', 'mpg'])
if uploaded_file:
    # Must save to temporary location for processing
    temp_path = save_uploaded_file(uploaded_file)

# CRITICAL: Long-running processes block Streamlit UI
# Example: PremiseCVPipeline.process_video() is synchronous and takes minutes
# Solution: Use threading or async patterns with status updates

# CRITICAL: Streamlit session state management
# Example: Use st.session_state to maintain processing status across reruns
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = 'idle'

# CRITICAL: File cleanup essential for production
# Example: Clean temporary files after processing to prevent disk bloat
try:
    result = process_video(temp_path)
finally:
    cleanup_temp_file(temp_path)

# GOTCHA: Streamlit file uploader widget behavior
# Once file uploaded, widget retains file until replaced or cleared
# Use st.session_state to track upload state

# GOTCHA: PremiseCVPipeline expects file path string, not UploadedFile
# Must write uploaded file to disk before processing
```

## Implementation Blueprint

### Data Models and Structure

```python
# New utility classes for async processing and file management

@dataclass
class ProcessingStatus:
    status: Literal['idle', 'uploading', 'processing', 'completed', 'error']
    message: str
    progress: float = 0.0
    video_file: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class UploadedVideoProcessor:
    """Async wrapper for PremiseCVPipeline to integrate with Streamlit"""
    def __init__(self, temp_dir: str = "./data/temp"):
        self.temp_dir = Path(temp_dir)
        self.status_callback: Optional[Callable] = None
    
    async def process_uploaded_video(self, uploaded_file, progress_callback=None) -> Dict[str, Any]:
        """Process uploaded file with progress tracking"""
        pass

class TempFileManager:
    """Handle temporary file lifecycle for uploaded videos"""
    @staticmethod
    def save_uploaded_file(uploaded_file, temp_dir: str) -> str:
        """Save uploaded file to temporary location"""
        pass
    
    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """Clean up temporary file after processing"""
        pass
```

### Task Implementation Order

```yaml
Task 1 - Create File Management Utilities:
  CREATE premise_cv_platform/utils/file_handler.py:
    - IMPLEMENT TempFileManager class
    - ADD save_uploaded_file method for UploadedFile -> disk 
    - ADD cleanup_temp_file method for post-processing cleanup
    - ADD validate_uploaded_file method using existing video_utils
    - PATTERN: Use pathlib.Path for cross-platform compatibility
    - ERROR_HANDLING: Validate file size, format, permissions

Task 2 - Create Async Processing Wrapper:
  CREATE premise_cv_platform/utils/async_processor.py:
    - IMPLEMENT UploadedVideoProcessor class 
    - ADD async process_uploaded_video method
    - INTEGRATE with existing PremiseCVPipeline class
    - ADD progress tracking with callback functions
    - PATTERN: Use threading.Thread for long-running processing
    - ERROR_HANDLING: Capture and return processing exceptions

Task 3 - Enhance Dashboard with Upload Interface:
  MODIFY premise_cv_platform/reporting/streamlit_dashboard.py:
    - ADD video upload section to main dashboard
    - INJECT after existing sidebar configuration
    - ADD st.file_uploader widget with video format validation
    - ADD "Start Analysis" button with processing trigger
    - PRESERVE existing dashboard methods and structure
    - PATTERN: Mirror sample_app.py frontend patterns only

Task 4 - Add Processing Status Display:
  MODIFY premise_cv_platform/reporting/streamlit_dashboard.py:
    - ADD real-time status display section
    - ADD progress bar during processing (st.progress)
    - ADD status messages and log streaming
    - ADD processing completion detection
    - PATTERN: Use st.session_state for status management
    - INTEGRATE with async_processor progress callbacks

Task 5 - Implement Automatic Result Refresh:
  MODIFY premise_cv_platform/reporting/streamlit_dashboard.py:
    - MODIFY existing data loading methods
    - ADD automatic cache clearing after processing
    - ADD result display trigger after completion
    - PRESERVE existing chart and visualization methods
    - PATTERN: Use st.rerun() for automatic refresh

Task 6 - Add Error Handling and User Feedback:
  MODIFY premise_cv_platform/reporting/streamlit_dashboard.py:
    - ADD comprehensive error handling for upload/processing failures
    - ADD user-friendly error messages
    - ADD file validation feedback
    - ADD processing timeout handling
    - PATTERN: Use st.error(), st.warning(), st.success() for user feedback
```

### Integration Points

```yaml
PROCESSING_PIPELINE:
  - integrate_with: main.py PremiseCVPipeline
  - method: "Wrap existing process_video method with async interface"
  - preserve: "All existing processing logic and CSV export"

DASHBOARD_STATE:
  - add_to: st.session_state
  - keys: "processing_status, uploaded_file_path, processing_results"
  - pattern: "Initialize in dashboard __init__ or early in run()"

TEMPORARY_FILES:
  - directory: "data/temp"  
  - pattern: "video_upload_{timestamp}_{filename}"
  - cleanup: "After processing completion or on error"

FILE_VALIDATION:
  - reuse: premise_cv_platform.data_ingestion.video_utils.validate_video_file
  - formats: ['.mov', '.mp4', '.mpg'] 
  - size_limit: "Use settings.max_upload_file_size"

ERROR_LOGGING:
  - integrate: premise_cv_platform.utils.logging_config
  - pattern: "Log upload and processing events for debugging"
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check premise_cv_platform/utils/ --fix
ruff check premise_cv_platform/reporting/streamlit_dashboard.py --fix
mypy premise_cv_platform/utils/
mypy premise_cv_platform/reporting/streamlit_dashboard.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests

```python
# CREATE tests/test_file_handler.py
def test_save_uploaded_file():
    """Test saving uploaded file to temporary location"""
    # Mock UploadedFile and verify file saved correctly
    pass

def test_cleanup_temp_file():
    """Test temporary file cleanup"""
    # Create temp file, cleanup, verify removal
    pass

def test_validate_uploaded_file():
    """Test file validation using existing video_utils"""
    # Test valid/invalid formats, sizes
    pass

# CREATE tests/test_async_processor.py  
def test_process_uploaded_video():
    """Test async video processing wrapper"""
    # Mock PremiseCVPipeline, test processing flow
    pass

def test_progress_tracking():
    """Test progress callback functionality"""
    # Verify progress updates during processing
    pass
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_file_handler.py -v
uv run pytest tests/test_async_processor.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test

```bash
# Start the dashboard
python -m streamlit run premise_cv_platform/reporting/streamlit_dashboard.py

# Test upload functionality:
# 1. Open dashboard in browser
# 2. Navigate to upload section
# 3. Upload a test video file (use videos/bank_sample.MOV)
# 4. Click "Start Analysis" button
# 5. Verify processing status updates appear
# 6. Wait for completion and verify results display
# 7. Check that temporary files are cleaned up

# Expected: Successful upload, processing, and result display
# If error: Check browser console and streamlit terminal for errors
```

### Level 4: End-to-End Workflow Validation

```bash
# Complete user journey test:
# 1. Upload bank_sample.MOV via dashboard
# 2. Monitor processing progress indicators  
# 3. Verify CSV files generated in data/csv_exports/
# 4. Confirm dashboard charts update with new data
# 5. Test error scenarios (invalid file types, corrupted files)
# 6. Verify temp file cleanup after completion

# Performance validation:
# - Upload larger video file and monitor processing time
# - Verify dashboard remains responsive during processing
# - Check memory usage doesn't exceed reasonable limits

# Data validation:
# - Compare dashboard results with CLI processing of same video
# - Verify CSV exports match expected schema
# - Confirm all metrics display correctly in dashboard
```

## Final Validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check premise_cv_platform/`
- [ ] No type errors: `uv run mypy premise_cv_platform/`
- [ ] Dashboard starts without errors: `streamlit run premise_cv_platform/reporting/streamlit_dashboard.py`
- [ ] File upload widget accepts valid video formats
- [ ] Processing pipeline executes on uploaded videos
- [ ] Real-time status updates display during processing
- [ ] Results automatically appear after processing completion
- [ ] Error cases handled gracefully with user feedback
- [ ] Temporary files cleaned up after processing
- [ ] Dashboard remains responsive during long-running processing
- [ ] CSV exports generated correctly for uploaded videos
- [ ] Integration with existing dashboard features preserved

---

## Current vs Desired State Summary

**Current State Issues:**
- No video upload capability in dashboard
- CLI-only video processing workflow
- Manual result refresh required
- No real-time processing feedback
- Technical barrier for non-CLI users

**Desired State Benefits:**
- Complete web-based workflow from upload to results
- Real-time processing status and feedback
- Automatic result display upon completion
- Accessible to non-technical users
- Streamlined analytics workflow

**Transformation Impact:**
- **High-Level**: Transform CLI-based tool into user-friendly web application
- **Mid-Level**: Integrate upload, processing, and display into single interface
- **Low-Level**: Add file handling, async processing, and status tracking components

**Risk Mitigation:**
- Preserve all existing dashboard functionality
- Maintain compatibility with existing data schemas
- Use existing processing pipeline without modification
- Implement comprehensive error handling
- Include rollback capability by preserving original dashboard methods

## Anti-Patterns to Avoid

- ❌ Don't modify core PremiseCVPipeline processing logic
- ❌ Don't break existing dashboard functionality during enhancement
- ❌ Don't use blocking synchronous processing in Streamlit
- ❌ Don't skip temporary file cleanup (causes disk bloat)
- ❌ Don't ignore file validation (security and stability risk)
- ❌ Don't copy backend logic from sample_app.py (different domain)
- ❌ Don't hardcode file paths or processing parameters
- ❌ Don't assume upload success without validation