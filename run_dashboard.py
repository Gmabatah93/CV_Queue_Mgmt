#!/usr/bin/env python3
"""
PREMISE CV Dashboard Launcher

This script provides an easy way to launch the Streamlit dashboard
with proper configuration and environment setup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Set up the environment for the dashboard."""
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    
    print("✅ Environment configured successfully")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'opencv-python',
        'numpy',
        'plotly',
        'pandas',
        'matplotlib',
        'psutil',
        'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Please install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def launch_dashboard(port=8501, host="localhost", debug=False):
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "premise_cv_platform" / "interface" / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    # Streamlit command
    cmd = [
        "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host
    ]
    
    if debug:
        cmd.extend(["--server.runOnSave", "true"])
        cmd.extend(["--logger.level", "debug"])
    
    print(f"🚀 Launching PREMISE CV Dashboard...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"🔧 Debug mode: {'enabled' if debug else 'disabled'}")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Launch PREMISE CV Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on (default: 8501)")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-check", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    print("🎬 PREMISE CV Dashboard Launcher")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies unless skipped
    if not args.no_check and not check_dependencies():
        sys.exit(1)
    
    # Launch dashboard
    success = launch_dashboard(port=args.port, host=args.host, debug=args.debug)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()