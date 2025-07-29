#!/usr/bin/env python3
"""
YOLO Model Comparison Tool

Test different YOLO models on your video and compare their performance:
- Detection accuracy
- Processing speed
- Confidence scores
- Tracking quality
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from premise_cv_platform.inference.track_people import PersonTracker
from premise_cv_platform.storage.data_schemas import Detection
from premise_cv_platform.data_ingestion.process_video import VideoProcessor


@dataclass
class ModelResult:
    """Results for a single model test."""
    model_name: str
    total_frames: int
    total_detections: int
    unique_persons: int
    avg_confidence: float
    processing_time: float
    fps: float
    detections_per_frame: float
    tracking_quality: float
    model_size_mb: float
    memory_usage_mb: float


class YOLOModelComparator:
    """Compare different YOLO models on the same video."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.results: List[ModelResult] = []
        
        # Available YOLO models to test
        self.models = {
            "yolo11n.pt": "YOLOv11 Nano (Fastest)",
            "yolo11s.pt": "YOLOv11 Small (Balanced)",
            "yolo11m.pt": "YOLOv11 Medium (Accurate)",
            "yolo11l.pt": "YOLOv11 Large (Most Accurate)",
            "yolo11x.pt": "YOLOv11 XLarge (Best Accuracy)",
            "yolov8n.pt": "YOLOv8 Nano (Legacy)",
            "yolov8s.pt": "YOLOv8 Small (Legacy)",
            "yolov8m.pt": "YOLOv8 Medium (Legacy)",
            "yolov8l.pt": "YOLOv8 Large (Legacy)",
            "yolov8x.pt": "YOLOv8 XLarge (Legacy)"
        }
    
    def test_model(self, model_name: str, max_frames: int = 100) -> ModelResult:
        """Test a single YOLO model on the video."""
        print(f"\n🧪 Testing {model_name} ({self.models.get(model_name, 'Unknown')})")
        print("=" * 60)
        
        # Initialize tracker with specific model
        tracker = PersonTracker(model_name=model_name)
        
        # Measure model loading time
        load_start = time.time()
        try:
            tracker.load_model()
            load_time = time.time() - load_start
            print(f"✅ Model loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
        
        # Get model size (approximate)
        model_size_mb = self.estimate_model_size(model_name)
        
        # Process video frames
        start_time = time.time()
        frame_count = 0
        total_detections = 0
        confidence_scores = []
        
        try:
            with VideoProcessor(self.video_path) as processor:
                for frame_number, frame, timestamp in processor.get_frame_generator():
                    if frame_count >= max_frames:
                        break
                    
                    # Run detection
                    detections = tracker.detect_and_track(frame, frame_number, timestamp)
                    total_detections += len(detections)
                    
                    # Collect confidence scores
                    for detection in detections:
                        confidence_scores.append(detection.confidence)
                    
                    frame_count += 1
                    
                    # Progress update
                    if frame_count % 20 == 0:
                        print(f"   📊 Processed {frame_count}/{max_frames} frames "
                              f"({frame_count/max_frames*100:.1f}%)")
        
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return None
        
        # Calculate metrics
        processing_time = time.time() - start_time
        fps = frame_count / processing_time if processing_time > 0 else 0
        detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Get tracking statistics
        stats = tracker.get_tracking_statistics()
        unique_persons = stats['unique_persons_tracked']
        tracking_quality = stats['tracking_quality']
        
        # Estimate memory usage
        memory_usage_mb = self.estimate_memory_usage(model_name)
        
        # Create result
        result = ModelResult(
            model_name=model_name,
            total_frames=frame_count,
            total_detections=total_detections,
            unique_persons=unique_persons,
            avg_confidence=avg_confidence,
            processing_time=processing_time,
            fps=fps,
            detections_per_frame=detections_per_frame,
            tracking_quality=tracking_quality,
            model_size_mb=model_size_mb,
            memory_usage_mb=memory_usage_mb
        )
        
        # Print results
        print(f"�� Results for {model_name}:")
        print(f"   - Frames Processed: {frame_count}")
        print(f"   - Total Detections: {total_detections}")
        print(f"   - Unique Persons: {unique_persons}")
        print(f"   - Avg Confidence: {avg_confidence:.3f}")
        print(f"   - Processing FPS: {fps:.1f}")
        print(f"   - Detections/Frame: {detections_per_frame:.2f}")
        print(f"   - Tracking Quality: {tracking_quality:.3f}")
        print(f"   - Model Size: {model_size_mb:.1f}MB")
        print(f"   - Memory Usage: {memory_usage_mb:.1f}MB")
        
        return result
    
    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in MB."""
        size_map = {
            "yolo11n.pt": 6.0,
            "yolo11s.pt": 22.0,
            "yolo11m.pt": 52.0,
            "yolo11l.pt": 87.0,
            "yolo11x.pt": 136.0,
            "yolov8n.pt": 6.0,
            "yolov8s.pt": 22.0,
            "yolov8m.pt": 52.0,
            "yolov8l.pt": 87.0,
            "yolov8x.pt": 136.0
        }
        return size_map.get(model_name, 50.0)
    
    def estimate_memory_usage(self, model_name: str) -> float:
        """Estimate memory usage in MB."""
        memory_map = {
            "yolo11n.pt": 150.0,
            "yolo11s.pt": 200.0,
            "yolo11m.pt": 300.0,
            "yolo11l.pt": 400.0,
            "yolo11x.pt": 500.0,
            "yolov8n.pt": 150.0,
            "yolov8s.pt": 200.0,
            "yolov8m.pt": 300.0,
            "yolov8l.pt": 400.0,
            "yolov8x.pt": 500.0
        }
        return memory_map.get(model_name, 250.0)
    
    def compare_models(self, models_to_test: List[str] = None, max_frames: int = 100):
        """Compare multiple YOLO models."""
        if models_to_test is None:
            models_to_test = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"]
        
        print(f"🔬 YOLO Model Comparison")
        print(f"�� Video: {self.video_path}")
        print(f"🎯 Testing {len(models_to_test)} models on {max_frames} frames each")
        print("=" * 80)
        
        for model_name in models_to_test:
            if model_name in self.models:
                result = self.test_model(model_name, max_frames)
                if result:
                    self.results.append(result)
            else:
                print(f"⚠️  Model {model_name} not found in available models")
        
        # Generate comparison report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        if not self.results:
            print("❌ No results to compare")
            return
        
        print(f"\n📊 MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Create comparison table
        df = pd.DataFrame([vars(result) for result in self.results])
        
        # Sort by different metrics
        print(f"\n🏆 RANKINGS:")
        
        # Speed ranking
        speed_ranking = df.sort_values('fps', ascending=False)
        print(f"\n�� SPEED RANKING (FPS):")
        for i, (_, row) in enumerate(speed_ranking.iterrows(), 1):
            print(f"   {i}. {row['model_name']}: {row['fps']:.1f} FPS")
        
        # Detection ranking
        detection_ranking = df.sort_values('total_detections', ascending=False)
        print(f"\n👥 DETECTION RANKING (Total Detections):")
        for i, (_, row) in enumerate(detection_ranking.iterrows(), 1):
            print(f"   {i}. {row['model_name']}: {row['total_detections']} detections")
        
        # Confidence ranking
        confidence_ranking = df.sort_values('avg_confidence', ascending=False)
        print(f"\n�� CONFIDENCE RANKING (Avg Confidence):")
        for i, (_, row) in enumerate(confidence_ranking.iterrows(), 1):
            print(f"   {i}. {row['model_name']}: {row['avg_confidence']:.3f}")
        
        # Efficiency ranking (detections per second)
        df['efficiency'] = df['total_detections'] / df['processing_time']
        efficiency_ranking = df.sort_values('efficiency', ascending=False)
        print(f"\n⚡ EFFICIENCY RANKING (Detections/Second):")
        for i, (_, row) in enumerate(efficiency_ranking.iterrows(), 1):
            print(f"   {i}. {row['model_name']}: {row['efficiency']:.1f} detections/sec")
        
        # Detailed comparison table
        print(f"\n📋 DETAILED COMPARISON:")
        print("-" * 80)
        print(f"{'Model':<15} {'FPS':<8} {'Detections':<12} {'Confidence':<12} {'Memory':<8} {'Size':<8}")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result.model_name:<15} {result.fps:<8.1f} {result.total_detections:<12} "
                  f"{result.avg_confidence:<12.3f} {result.memory_usage_mb:<8.1f} {result.model_size_mb:<8.1f}")
        
        # Save results to file
        self.save_results()
        
        # Generate recommendations
        self.generate_recommendations()
    
    def save_results(self):
        """Save results to JSON file."""
        results_data = []
        for result in self.results:
            results_data.append({
                'model_name': result.model_name,
                'model_description': self.models.get(result.model_name, 'Unknown'),
                'total_frames': result.total_frames,
                'total_detections': result.total_detections,
                'unique_persons': result.unique_persons,
                'avg_confidence': result.avg_confidence,
                'processing_time': result.processing_time,
                'fps': result.fps,
                'detections_per_frame': result.detections_per_frame,
                'tracking_quality': result.tracking_quality,
                'model_size_mb': result.model_size_mb,
                'memory_usage_mb': result.memory_usage_mb
            })
        
        output_file = "data/yolo_model_comparison_results.json"
        Path("data").mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def generate_recommendations(self):
        """Generate model recommendations based on results."""
        print(f"\n💡 RECOMMENDATIONS:")
        
        if not self.results:
            return
        
        # Find best models for different criteria
        fastest = max(self.results, key=lambda x: x.fps)
        most_detections = max(self.results, key=lambda x: x.total_detections)
        highest_confidence = max(self.results, key=lambda x: x.avg_confidence)
        smallest = min(self.results, key=lambda x: x.model_size_mb)
        
        print(f"��‍♂️  FASTEST: {fastest.model_name} ({fastest.fps:.1f} FPS)")
        print(f"   - Best for real-time applications")
        print(f"   - Good for live video processing")
        
        print(f"\n�� MOST DETECTIONS: {most_detections.model_name} ({most_detections.total_detections} detections)")
        print(f"   - Best for finding all possible persons")
        print(f"   - Good for surveillance applications")
        
        print(f"\n�� HIGHEST CONFIDENCE: {highest_confidence.model_name} ({highest_confidence.avg_confidence:.3f})")
        print(f"   - Most reliable detections")
        print(f"   - Good for critical applications")
        
        print(f"\n📦 SMALLEST MODEL: {smallest.model_name} ({smallest.model_size_mb:.1f}MB)")
        print(f"   - Best for resource-constrained environments")
        print(f"   - Good for edge devices")
        
        # Overall recommendation
        print(f"\n🏆 OVERALL RECOMMENDATION:")
        if fastest.fps > 10 and most_detections.total_detections > 0:
            print(f"   Use {fastest.model_name} for speed + {most_detections.model_name} for accuracy")
        elif fastest.fps > 15:
            print(f"   Use {fastest.model_name} for real-time processing")
        elif most_detections.total_detections > 0:
            print(f"   Use {most_detections.model_name} for maximum detection")
        else:
            print(f"   Consider using a different video or adjusting detection parameters")


def main():
    """Main function to run model comparison."""
    video_path = "videos/bank_sample.MOV"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Create comparator
    comparator = YOLOModelComparator(video_path)
    
    # Define models to test (you can modify this list)
    models_to_test = [
        "yolo11n.pt",  # Fastest
        "yolo11s.pt",  # Balanced
        "yolo11m.pt",  # Accurate
        "yolov8n.pt",  # Legacy fast
        "yolov8s.pt"   # Legacy balanced
    ]
    
    print("🔬 YOLO Model Comparison Tool")
    print("=" * 50)
    print(f"�� Video: {video_path}")
    print(f"🧪 Models to test: {len(models_to_test)}")
    print(f"🎯 Frames per test: 100")
    print("\nThis will test each model and compare:")
    print("   - Processing speed (FPS)")
    print("   - Detection accuracy")
    print("   - Confidence scores")
    print("   - Memory usage")
    print("   - Model size")
    print("\n" + "=" * 50)
    
    # Run comparison
    comparator.compare_models(models_to_test, max_frames=100)


if __name__ == "__main__":
    main()