# PREMISE CV Dashboard Deployment Guide

This guide provides instructions for deploying and running the PREMISE CV Dashboard with Streamlit.

## Quick Start

### 1. Install Dependencies

```bash
# Install all required dependencies
pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
# Simple launch
python run_dashboard.py

# Or with custom options
python run_dashboard.py --port 8502 --host 0.0.0.0 --debug
```

### 3. Access Dashboard

Open your browser and navigate to: `http://localhost:8501`

## Detailed Setup

### Environment Requirements

- Python 3.8 or higher
- 8GB+ RAM recommended for video processing
- GPU support optional but recommended for faster processing

### Configuration Files

The dashboard uses several configuration files located in `.streamlit/`:

- `config.toml` - Main Streamlit configuration
- `secrets.toml` - Sensitive configuration (keep private)

### Dashboard Features

The enhanced dashboard includes:

1. **🎬 Live Video Stream** - Real-time video processing with zone overlays
2. **📤 Video Upload** - Upload and process video files with progress tracking
3. **📊 Analytics** - Comprehensive event analytics and visualizations
4. **🗺️ Zone Config** - Interactive zone configuration with live preview
5. **📋 Event Monitor** - Real-time event monitoring with filtering
6. **📈 Performance Monitor** - System performance tracking and alerts
7. **⚙️ System Status** - Overall system health and configuration

## Running Options

### Command Line Options

```bash
python run_dashboard.py --help
```

Available options:
- `--port PORT` - Port to run dashboard on (default: 8501)
- `--host HOST` - Host to bind to (default: localhost)
- `--debug` - Enable debug mode with auto-reload
- `--no-check` - Skip dependency checking

### Direct Streamlit Launch

```bash
# Launch directly with Streamlit
streamlit run premise_cv_platform/interface/streamlit_dashboard.py

# With custom port
streamlit run premise_cv_platform/interface/streamlit_dashboard.py --server.port 8502
```

## Deployment Scenarios

### Development Mode

```bash
# Run with debug mode and auto-reload
python run_dashboard.py --debug
```

### Production Mode

```bash
# Run on all interfaces for external access
python run_dashboard.py --host 0.0.0.0 --port 8501
```

### Docker Deployment (Future)

```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["python", "run_dashboard.py", "--host", "0.0.0.0"]
```

## Configuration

### Performance Settings

Edit `.streamlit/secrets.toml` to adjust:

```toml
[performance]
enable_performance_monitoring = true
max_memory_usage_gb = 8
max_cpu_usage_percent = 80

[video_processing]
max_video_size_mb = 200
max_processing_time_minutes = 30
default_frame_skip = 1
enable_gpu_acceleration = true
```

### Security Settings

```toml
[general]
app_secret_key = "your-secure-secret-key"
debug_mode = false
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use different port
   python run_dashboard.py --port 8502
   ```

2. **Memory Issues**
   - Reduce video file size
   - Increase frame skip value
   - Close other applications

3. **GPU Not Detected**
   - Install CUDA drivers
   - Install GPU-enabled packages
   - Check GPU compatibility

4. **Slow Performance**
   - Enable GPU acceleration
   - Reduce video resolution
   - Increase frame skip
   - Check system resources

### Log Files

Logs are available in:
- `data/logs/premise_cv.log` - Main application logs
- `data/logs/performance.log` - Performance metrics
- `data/logs/errors.log` - Error tracking

### Support

For issues and support:
1. Check logs for error messages
2. Verify all dependencies are installed
3. Ensure video files are in supported formats
4. Check system resources (CPU, memory, disk space)

## Performance Optimization

### System Recommendations

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for large video processing
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: SSD recommended for faster I/O

### Processing Optimization

1. **Frame Skip**: Increase for faster processing, decrease for accuracy
2. **Resolution**: Lower resolution for faster processing
3. **Batch Size**: Adjust based on available memory
4. **GPU Acceleration**: Enable if available

## Security Considerations

1. **Network Access**: By default, dashboard binds to localhost only
2. **File Uploads**: Limited to 200MB by default
3. **Secrets**: Keep `.streamlit/secrets.toml` private
4. **HTTPS**: Use reverse proxy (nginx) for HTTPS in production

## Monitoring

The dashboard includes built-in monitoring:

- **Performance Metrics**: FPS, CPU, memory usage
- **System Health**: Automated alerts for issues
- **Event Tracking**: Real-time event monitoring
- **Error Logging**: Comprehensive error tracking

Access monitoring via the "Performance Monitor" tab in the dashboard.