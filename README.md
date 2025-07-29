# CV Queue Management System

A computer vision-based queue management system for banking environments using YOLO object detection and zone-based event tracking.

## 🎯 Features

- **Person Detection & Tracking**: YOLO-based person detection with multi-object tracking
- **Zone Analysis**: Teller Access Line and Teller Interaction Zone detection
- **Event Detection**: Real-time event logging for queue management
- **Visualization**: Real-time visualization of zone events and person tracking

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Visualization
```bash
python visualize_IDEAL.py
```

### Run Tests
```bash
python complete_updated_detector_test.py
```

## 📊 Results

- **18 events detected** across 8 people
- **Real-time processing** at 8.9 FPS
- **Simple bounding box logic** for reliable zone detection
- **Color-coded visualization** for easy monitoring

## 🏗️ Architecture

1. **Video Processing**: OpenCV-based video ingestion
2. **Person Tracking**: YOLO + custom tracking algorithm
3. **Zone Analysis**: Simple bounding box intersection logic
4. **Event Detection**: Real-time zone entry/exit detection
5. **Data Export**: CSV export of events and statistics

## 📁 Project Structure
