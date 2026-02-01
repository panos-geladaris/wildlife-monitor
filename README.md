# Wildlife Monitor

A Raspberry Pi-based wildlife monitoring system that detects and classifies animals passing by using computer vision.

## Overview

This system uses a PIR motion sensor and camera module connected to a Raspberry Pi to:
- Capture 1-2 second video clips when motion is detected
- Record scheduled hourly video samples
- Analyze footage using a TorchVision neural network to identify animals
- Provide a web UI to browse detections and view statistics

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raspberry Pi   │────▶│  Video Storage   │────▶│  Web UI (Flask) │
│  + PIR Sensor   │     │  + ML Analysis   │     │  View Results   │
│  + Camera       │     │  (TorchVision)   │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Hardware Requirements

- Raspberry Pi 4 (recommended for ML inference)
- Pi Camera Module 3 (12MP, autofocus, HDR support)
- PIR Motion Sensor (HC-SR501)

### PIR Sensor Wiring

| PIR Pin | Raspberry Pi |
|---------|--------------|
| VCC     | 5V (Pin 2)   |
| GND     | Ground (Pin 6) |
| OUT     | GPIO 17 (Pin 11) |

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `src/capture/` | ✅ Complete | Motion detection, camera control, scheduling |
| `src/storage/` | ✅ Complete | SQLite database, video file management |
| `src/analysis/` | 🔲 Planned | TorchVision animal classification |
| `src/web/` | 🔲 Planned | Flask web UI |

## Installation

```bash
# Clone or navigate to project
cd ~/projects/wildlife-monitor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Platform-Specific Dependencies

**On Raspberry Pi:**
```bash
pip install picamera2 RPi.GPIO
```

**For ML development (laptop/desktop):**
```bash
pip install torch torchvision
```

## Running on Raspberry Pi

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-pi.txt

# Run the capture service
python -m src.capture.capture_service
```

The service will start monitoring the PIR sensor and capturing video on motion detection.

## Running the Capture Module

### Simulation Mode (Laptop/Desktop)

Test the capture system without Pi hardware:

```bash
source venv/bin/activate

# Run the capture service demo
python -m src.capture.capture_service
```

This will:
- Start motion detection (simulated)
- Schedule hourly captures
- Create placeholder files in `data/videos/`

### On Raspberry Pi

```python
from pathlib import Path
from src.capture import CaptureService, CaptureServiceConfig

config = CaptureServiceConfig(
    video_output_dir=Path("data/videos"),
    gpio_pin=17,              # PIR sensor GPIO pin
    video_duration=2.0,       # Seconds per clip
    resolution=(1280, 720),
    cooldown_seconds=5.0,     # Delay between motion triggers
    hourly_capture_minute=0,  # Capture at :00 each hour
    simulation_mode=False     # Use real hardware
)

service = CaptureService(config)

# Optional: hook into captures for processing
def on_new_video(metadata):
    print(f"New video: {metadata.filepath}")
    # Send to ML analysis pipeline here

service.on_capture(on_new_video)
service.start()

# Keep running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    service.stop()
```

### Testing Individual Components

```bash
# Test motion detector
python -m src.capture.motion_detector

# Test camera
python -m src.capture.camera

# Test scheduler
python -m src.capture.scheduler
```

## Configuration

Configuration is loaded from `config.yaml` in the project root. You can also specify a custom config file:

```bash
python -m src.capture.capture_service --config /path/to/config.yaml
```

### config.yaml

```yaml
# Motion-triggered captures
motion:
  video_duration: 2.5        # Duration in seconds
  cooldown_seconds: 5.0      # Minimum time between triggers

# Hourly scheduled captures  
hourly:
  enabled: true
  minute: 0                  # Minute of each hour (0-59)
  video_duration: 15.0       # Duration in seconds

# 15-minute interval captures (with zoom comparison)
interval:
  enabled: true
  minutes: 15                # Interval in minutes
  video_duration: 7.0        # Duration in seconds
  zoom_levels: [1.0, 2.0]    # Capture at each zoom level

# Camera settings
camera:
  resolution: [1280, 720]    # Width x Height
  framerate: 30
  zoom_level: 1.0            # Default zoom (1.0-10.0)
  autofocus: true

# Hardware
hardware:
  gpio_pin: 17               # PIR sensor GPIO pin

# Output
output:
  video_dir: "data/videos"
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hardware.gpio_pin` | 17 | GPIO pin for PIR sensor |
| `motion.video_duration` | 2.5 | Motion-triggered recording length (seconds) |
| `motion.cooldown_seconds` | 5.0 | Minimum time between motion triggers |
| `hourly.video_duration` | 15.0 | Hourly scheduled recording length (seconds) |
| `hourly.minute` | 0 | Minute of hour for scheduled capture |
| `interval.video_duration` | 7.0 | Interval capture duration (seconds) |
| `interval.zoom_levels` | [1.0, 2.0] | Zoom levels for interval captures |
| `camera.resolution` | [1280, 720] | Video resolution |
| `camera.framerate` | 30 | Video framerate |
| `camera.zoom_level` | 1.0 | Default digital zoom (1.0-10.0x) |
| `camera.autofocus` | true | Enable continuous autofocus |

### Zoom and Focus Control

The Camera Module 3 supports digital zoom and autofocus. Configure in your service setup:

```python
config = CaptureServiceConfig(
    video_output_dir=Path("data/videos"),
    zoom_level=2.0,      # 2x digital zoom
    autofocus=True,      # Enable continuous autofocus
)
```

Or control at runtime:

```python
# Digital zoom (1.0 = no zoom, up to 10.0x)
service._camera.set_zoom(3.0)
service._camera.reset_zoom()

# Autofocus
service._camera.set_autofocus(True)   # Continuous autofocus
service._camera.set_autofocus(False)  # Disable autofocus

# Manual focus (distance in metres)
service._camera.set_manual_focus(5.0)   # Focus at 5 metres
service._camera.set_manual_focus(0.0)   # Focus at infinity
```

**Note:** Higher zoom levels crop the sensor image and upscale it. For best quality, keep zoom ≤ 2x when using 1080p output resolution.

## Storage Module

The storage module provides SQLite database for detection metadata and video file management.

### Database Setup

The database is automatically created on first use at `data/wildlife.db`. No manual setup required.

```python
from src.storage import Database, Detection, VideoStore

# Initialize database
db = Database()  # Uses default path: data/wildlife.db

# Add a detection record
detection = Detection(
    video_path="data/videos/motion_20240115_120000.mp4",
    trigger_type="motion",
)
detection_id = db.add_detection(detection)

# Update with analysis results
db.update_detection(detection_id, animal_class="bird", confidence=0.92, analyzed=True)

# Query detections
recent = db.get_detections(trigger_type="motion", limit=10)
unanalyzed = db.get_unanalyzed_detections()

# Get daily summary
summary = db.update_daily_summary()
print(f"Today: {summary.total_detections} detections, animals: {summary.animal_counts}")
```

### Video Store

```python
from src.storage import VideoStore

store = VideoStore()  # Uses default path: data/videos/

# List videos
videos = store.list_videos(trigger_type="motion", limit=20)

# Get storage usage
usage = store.get_storage_usage()
print(f"Storage: {usage['total_mb']} MB, {usage['video_count']} videos")

# Cleanup old videos (default: 30 days retention)
deleted = store.cleanup_old_videos(retention_days=30)
```

### Database on Raspberry Pi

SQLite works out of the box on Raspberry Pi OS. For better performance with SD cards:

```bash
# Ensure data directory exists with proper permissions
mkdir -p ~/projects/wildlife-monitor/data
chmod 755 ~/projects/wildlife-monitor/data

# Optional: Mount a USB drive for video storage (recommended for longevity)
# 1. Format USB drive as ext4
# 2. Mount it:
sudo mkdir -p /mnt/wildlife-data
sudo mount /dev/sda1 /mnt/wildlife-data

# 3. Update config.yaml to use USB storage:
# output:
#   video_dir: "/mnt/wildlife-data/videos"
```

**Tips for SD card longevity:**
- Store videos on USB drive instead of SD card
- Use `retention_days` to auto-delete old videos
- Consider using a high-endurance SD card

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_database.py
```

## Project Structure

```
wildlife-monitor/
├── src/
│   ├── capture/
│   │   ├── motion_detector.py  # PIR sensor monitoring
│   │   ├── camera.py           # Video recording
│   │   ├── scheduler.py        # Hourly captures
│   │   ├── config.py           # Configuration loader
│   │   └── capture_service.py  # Main orchestrator
│   ├── storage/
│   │   ├── database.py         # SQLite operations
│   │   └── video_store.py      # Video file management
│   ├── analysis/               # (planned) ML classification
│   └── web/                    # (planned) Flask UI
├── tests/
│   ├── test_database.py
│   └── test_video_store.py
├── data/
│   ├── videos/                 # Captured video clips
│   └── wildlife.db             # SQLite database
├── config.yaml                 # Configuration file
├── requirements.txt
└── README.md
```

## License

MIT
