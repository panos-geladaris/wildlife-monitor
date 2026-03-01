# Wildlife Monitor

A Raspberry Pi-based wildlife monitoring system that detects and classifies animals passing by using computer vision.

## Overview

This system uses a PIR motion sensor and camera module connected to a Raspberry Pi to:
- Capture video clips when motion is detected (configurable duration)
- Record scheduled hourly video samples
- Gate captures to daylight hours using sunrise/sunset data
- Classify animals using TorchVision MobileNetV3
- Detect and annotate animals with bounding boxes using SSDLite320
- Generate video thumbnails for the gallery view
- Automatically clean up empty detections (no animal recognised) after a configurable age
- Generate daily time-lapse videos from the day's detections
- Provide a web UI to browse detections, timelapses, view statistics, and trigger test captures

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
| `src/capture/` | ✅ Complete | Motion detection, camera control, scheduling, daylight gating |
| `src/storage/` | ✅ Complete | SQLite database, video file management, thumbnails, cleanup, timelapse generation |
| `src/analysis/` | ✅ Complete | TorchVision classification + SSDLite320 object detection with bounding boxes |
| `src/web/` | ✅ Complete | Flask web UI with dashboard, gallery, statistics, test capture, and delete support |

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

## Quick Start

### Running the Full System

```bash
# Activate virtual environment
source venv/bin/activate

# Run the complete wildlife monitor
python main.py

# The system will:
# - Start motion detection and scheduled captures
# - Analyze videos with ML classification
# - Store results in the database
# - Serve web UI at http://localhost:5001
```

Press `Ctrl+C` to stop.

### Run Modes

```bash
# Full system (capture + analysis + web)
python main.py

# Capture only (no web UI)
python main.py --capture-only

# Web UI only (view existing data)
python main.py --web-only

# Analyze a single video
python main.py --analyze data/videos/motion_20240115_120000.mp4

# Force simulation mode (development without Pi hardware)
python main.py --simulate

# Custom port and paths
python main.py --port 8080 --video-dir /mnt/usb/videos --db-path /mnt/usb/wildlife.db
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to config.yaml file |
| `--video-dir PATH` | Directory for video files (default: data/videos) |
| `--db-path PATH` | Path to SQLite database (default: data/wildlife.db) |
| `--port PORT` | Web UI port (default: 5001) |
| `--capture-only` | Run capture service without web UI |
| `--web-only` | Run web UI only (no capture) |
| `--no-analysis` | Disable ML analysis (just record videos) |
| `--no-detection` | Disable object detection (bounding boxes) |
| `--no-cleanup` | Disable automatic cleanup of empty detections |
| `--no-timelapse` | Disable daily timelapse generation |
| `--simulate` | Force simulation mode (no Pi hardware required) |
| `--analyze VIDEO` | Analyze a single video file and exit |
| `-v, --verbose` | Enable verbose logging |

## Running on Raspberry Pi

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-pi.txt

# Run the full system
python main.py
```

The system will auto-detect Pi hardware and run with real camera and PIR sensor.

### Deploying to the Pi

A systemd service and deploy script are provided for production use.

**First-time setup on the Pi:**

```bash
# Clone the repo and create a virtual environment
cd /home/pi
git clone <repo-url> wildlife-monitor
cd wildlife-monitor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-pi.txt
cp config.yaml.example config.yaml
# Edit config.yaml with your location, timezone, etc.

# Install and start the systemd service
sudo cp scripts/wildlife-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wildlife-monitor
sudo systemctl start wildlife-monitor
```

**Deploying updates from your dev machine:**

After merging a feature branch, run the deploy script to update the Pi:

```bash
# Deploy to a Pi reachable as "pi" (default)
./scripts/deploy.sh

# Deploy to a specific host
./scripts/deploy.sh mypi.local

# Custom user
PI_USER=admin ./scripts/deploy.sh mypi.local
```

The script SSHs into the Pi, pulls the latest code, installs dependencies, and restarts the service.

**Useful systemd commands on the Pi:**

```bash
sudo systemctl status wildlife-monitor   # Check status
sudo systemctl stop wildlife-monitor     # Stop the service
sudo systemctl restart wildlife-monitor  # Restart
journalctl -u wildlife-monitor -f        # Follow logs
```

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

Copy the example configuration and adjust for your location:

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your coordinates, timezone, etc.
```

Configuration is loaded from `config.yaml` in the project root. You can also specify a custom config file:

```bash
python -m src.capture.capture_service --config /path/to/config.yaml
```

### config.yaml

```yaml
# Motion-triggered captures
motion:
  video_duration: 5.0        # Duration in seconds
  cooldown_seconds: 3.0      # Minimum time between triggers

# Hourly scheduled captures
hourly:
  enabled: true
  minute: 0                  # Minute of each hour (0-59)
  video_duration: 5.0        # Duration in seconds

# 15-minute interval captures (with zoom comparison)
interval:
  enabled: false
  minutes: 15                # Interval in minutes
  video_duration: 5.0        # Duration in seconds
  zoom_levels: [1.0]         # Capture at each zoom level

# Camera settings
camera:
  resolution: [1280, 720]    # Width x Height
  framerate: 30
  zoom_level: 1.0            # Default zoom (1.0-10.0)
  autofocus: true

# Hardware
hardware:
  gpio_pin: 17               # PIR sensor GPIO pin

# Object detection settings
detection:
  enabled: true
  score_threshold: 0.3       # Minimum confidence for bounding boxes
  max_boxes_per_frame: 5
  num_frames: 8              # Number of key frames to analyze

# Automatic cleanup of empty detections
cleanup:
  enabled: true
  max_age_hours: 24          # Delete empty detections older than this
  interval_hours: 6          # How often the cleanup job runs

# Daylight-only capture
daylight:
  enabled: true
  lat: 51.5074               # Latitude (e.g. London)
  lng: -0.1278               # Longitude
  tzid: "Europe/London"      # Timezone identifier
  start_offset_minutes: 20   # Start capturing this many minutes before sunrise
  end_offset_minutes: 20     # Stop capturing this many minutes after sunset
  fallback: "allow"          # "allow" or "deny" when sunrise API is unavailable

# Output
output:
  video_dir: "data/videos"

# Daily time-lapse generation
timelapse:
  enabled: true
  generation_hour: 21          # Hour of day to generate (0-23)
  generation_minute: 0         # Minute of hour
  frame_duration: 0.5          # Seconds each frame is shown
  resolution: [1280, 720]
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hardware.gpio_pin` | 17 | GPIO pin for PIR sensor |
| `motion.video_duration` | 5.0 | Motion-triggered recording length (seconds) |
| `motion.cooldown_seconds` | 3.0 | Minimum time between motion triggers |
| `hourly.video_duration` | 5.0 | Hourly scheduled recording length (seconds) |
| `hourly.minute` | 0 | Minute of hour for scheduled capture |
| `interval.enabled` | false | Enable interval captures |
| `interval.video_duration` | 5.0 | Interval capture duration (seconds) |
| `interval.zoom_levels` | [1.0] | Zoom levels for interval captures |
| `camera.resolution` | [1280, 720] | Video resolution |
| `camera.framerate` | 30 | Video framerate |
| `camera.zoom_level` | 1.0 | Default digital zoom (1.0-10.0x) |
| `camera.autofocus` | true | Enable continuous autofocus |
| `detection.enabled` | true | Enable object detection (bounding boxes) |
| `detection.score_threshold` | 0.3 | Minimum confidence for detected objects |
| `detection.max_boxes_per_frame` | 5 | Maximum bounding boxes per frame |
| `detection.num_frames` | 8 | Number of key frames to analyze per video |
| `cleanup.enabled` | true | Enable automatic cleanup of empty detections |
| `cleanup.max_age_hours` | 24 | Minimum age (hours) before an empty detection is removed |
| `cleanup.interval_hours` | 6 | How often the cleanup job runs |
| `daylight.enabled` | true | Restrict captures to daylight hours only |
| `daylight.lat` | — | Latitude for sunrise/sunset calculation |
| `daylight.lng` | — | Longitude for sunrise/sunset calculation |
| `daylight.tzid` | — | Timezone identifier (e.g. `Europe/London`) |
| `daylight.start_offset_minutes` | 0 | Start capturing this many minutes before sunrise |
| `daylight.end_offset_minutes` | 0 | Stop capturing this many minutes after sunset |
| `daylight.fallback` | `allow` | Behavior when sunrise API is unavailable (`allow` or `deny`) |
| `timelapse.enabled` | true | Enable daily timelapse generation |
| `timelapse.generation_hour` | 21 | Hour of day to generate timelapse (0-23) |
| `timelapse.generation_minute` | 0 | Minute of hour to generate timelapse |
| `timelapse.frame_duration` | 0.5 | Seconds each frame is shown in timelapse |
| `timelapse.resolution` | [1280, 720] | Output timelapse video resolution |

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

## Analysis Module

The analysis module uses TorchVision's MobileNetV3 to classify animals in captured videos.

### Supported Animals

- Birds (various species)
- Cats
- Dogs
- Squirrels
- Foxes
- Rabbits
- Deer
- Hedgehogs
- Mice

### Usage

```python
from src.analysis import AnimalClassifier
from pathlib import Path

# Initialize classifier
classifier = AnimalClassifier(
    model_name="mobilenet_v3_small",  # Lightweight model for Pi
    confidence_threshold=0.3,
)

# Classify a video
result = classifier.classify_video(Path("data/videos/motion_20240115_120000.mp4"))

print(f"Detected: {result.animal_class}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Is animal: {result.is_animal}")

# Get all frame results
all_results = classifier.classify_video_all_frames(video_path, num_frames=5)
for r in all_results:
    print(f"Frame {r.frame_number}: {r.animal_class} ({r.confidence:.2%})")
```

### ML Dependencies on Raspberry Pi

PyTorch on Raspberry Pi requires special installation (standard pip packages cause "Illegal instruction" errors):

```bash
# Option 1: PyTorch CPU wheels (recommended for Pi 4 64-bit)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Option 2: Use piwheels (pre-built for Pi)
pip install torch torchvision --extra-index-url https://www.piwheels.org/simple

# Install OpenCV
pip install opencv-python-headless  # Headless version for Pi
```

**If PyTorch installation fails**, run without ML analysis:
```bash
python main.py --no-analysis
```

The system will still capture videos and serve the web UI - classification can be added later.

**Note:** Model inference on Pi 4 takes ~1-2 seconds per frame with MobileNetV3.

### Object Detection (Bounding Boxes)

In addition to whole-image classification, the system uses SSDLite320 (MobileNetV3-Large backbone) to locate animals in video frames with bounding boxes.

- COCO-pretrained: detects bird, cat, dog, horse, sheep, cow, bear, and more
- For non-COCO animals: crops detected regions and runs the MobileNetV3 classifier for refined identification
- Annotated key frames are saved as JPEGs in `data/annotated/<detection_id>/`
- Bounding box data is stored in the `frame_objects` database table

Object detection runs automatically after classification in the capture pipeline. Disable with:
```bash
python main.py --no-detection
```

## Daylight-Only Capture

When enabled in `config.yaml`, captures are restricted to daylight hours using the [Sunrise-Sunset API](https://sunrise-sunset.org/api). This prevents unnecessary recordings at night.

Configure your location coordinates and timezone in `config.yaml` under the `daylight` section. The `start_offset_minutes` and `end_offset_minutes` parameters allow you to extend the capture window before sunrise and after sunset.

If the API is unavailable, the `fallback` setting controls whether captures are allowed (`allow`) or denied (`deny`).

## Daily Timelapses

The system generates a daily time-lapse video at a configurable time (default 21:00) by extracting one representative frame from each detection and stitching them into an MP4. This provides a quick visual summary of the day's wildlife activity.

Timelapses are stored in `data/timelapses/` and browsable via the Timelapses page in the web UI. Each timelapse shows the video alongside a breakdown of animals detected that day.

Configure the generation time and playback speed in `config.yaml` under the `timelapse` section. Disable with `--no-timelapse` or by setting `timelapse.enabled: false` in the config.

## Automatic Cleanup of Empty Detections

The system automatically removes detections where neither the classifier nor the object detector recognised any animal, once they are older than a configurable threshold (default: 24 hours). This saves storage on the Pi by discarding videos triggered by wind, shadows, or passing cars.

A detection is considered "empty" when all of the following are true:
- Analysis has completed (`analyzed = true`)
- No animal was classified (`animal_class` is null or `unknown`)
- No objects were detected in any frame (no `frame_objects` rows)

When an empty detection is removed, its video file, thumbnail, annotated frames directory, and database record are all deleted. The cleanup job runs on a configurable interval (default: every 6 hours) and can be disabled with `--no-cleanup` or by setting `cleanup.enabled: false` in `config.yaml`.

## Highlights

Any detection can be marked as a highlight from its detail page using the ★ button, regardless of trigger type or whether an animal was recognised. Highlighted detections are shown with a star icon in the gallery and collected in a dedicated Highlights page (`/highlights`) for easy access. Highlights can also be deleted directly from that page.

## Web UI

The web UI provides a browser-based interface to view detections, timelapses, and statistics.

### Running the Web UI

```bash
# Activate virtual environment
source venv/bin/activate

# Start the web server
python -m src.web.app

# Server runs on http://localhost:5001
```

### Accessing the Web UI

**On the same machine:**
- Open http://localhost:5001

**From another device on the network:**
1. Find the Raspberry Pi's IP address:
   ```bash
   hostname -I
   # Example output: 192.168.1.50
   ```
2. Open http://192.168.1.50:5001 in your browser

**Pages:**

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | System status, today's summary, recent detections, test capture button |
| Gallery | `/gallery` | Browse all detections with filters and thumbnails |
| Detection | `/detection/:id` | View video, classification results, and annotated key frames |
| Highlights | `/highlights` | Browse and manage highlighted detections |
| Timelapses | `/timelapses` | Browse daily time-lapse videos with animal summaries |
| Timelapse | `/timelapse/:id` | View timelapse video and animal breakdown |
| Statistics | `/statistics` | Charts of detection trends and animal breakdowns |

### API Endpoints

The web UI also exposes a REST API:

```
GET    /api/status                    - System status (storage, counts)
GET    /api/detections                - List detections (with filters)
GET    /api/detections/:id            - Single detection details
GET    /api/detections/:id/objects    - Bounding box data for a detection
DELETE /api/detections/:id            - Delete a detection and its video
POST   /api/detections/bulk-delete    - Bulk delete detections (body: {"ids": [1,2,3]})
POST   /api/detections/:id/highlight   - Toggle highlight on a detection
POST   /api/test-capture              - Trigger on-demand test capture with analysis
GET    /api/highlights                 - List highlighted detections
GET    /api/videos                    - List video files
GET    /api/stats/daily               - Daily detection counts
GET    /api/stats/animals             - Animal type breakdown
GET    /api/stats/summary             - Dashboard summary
GET    /videos/:filename              - Serve video files
GET    /thumbnails/:filename          - Serve video thumbnails
GET    /api/timelapses                - List daily timelapses
GET    /api/timelapses/:id            - Single timelapse details
DELETE /api/timelapses/:id            - Delete a timelapse and its video
GET    /annotated/:detection_id/:file - Serve annotated frame images
GET    /timelapse-videos/:filename    - Serve timelapse video files
GET    /timelapse-thumbnails/:filename - Serve timelapse thumbnails
```

**Example API usage:**
```bash
# Get system status
curl http://localhost:5001/api/status

# Get recent motion detections
curl "http://localhost:5001/api/detections?trigger_type=motion&limit=10"

# Get last 7 days of stats
curl "http://localhost:5001/api/stats/daily?days=7"

# Trigger a test capture (requires full system mode, not --web-only)
curl -X POST http://localhost:5001/api/test-capture

# Delete a detection
curl -X DELETE http://localhost:5001/api/detections/42

# Get bounding box data for a detection
curl http://localhost:5001/api/detections/42/objects
```

### Running in Production

For production on Raspberry Pi, consider using Gunicorn:

```bash
pip install gunicorn

# Run with 2 workers
gunicorn -w 2 -b 0.0.0.0:5001 "src.web.app:create_app()"
```

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
pytest tests/test_analysis.py
pytest tests/test_detection.py
pytest tests/test_daylight.py
pytest tests/test_thumbnail.py
pytest tests/test_test_capture.py
pytest tests/test_timelapse.py
pytest tests/test_cleanup.py
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
│   │   ├── daylight.py         # Sunrise/sunset daylight gating
│   │   └── capture_service.py  # Main orchestrator
│   ├── storage/
│   │   ├── database.py         # SQLite operations
│   │   ├── cleanup.py          # Automatic empty detection cleanup
│   │   ├── thumbnail.py        # Video thumbnail generation
│   │   ├── timelapse.py        # Daily time-lapse generation
│   │   └── video_store.py      # Video file management
│   ├── analysis/
│   │   ├── model.py            # TorchVision model loading
│   │   ├── classifier.py       # Animal classification
│   │   ├── frame_extractor.py  # Video frame extraction
│   │   ├── detection_model.py  # SSDLite320 object detection model
│   │   ├── detector.py         # Object detection orchestrator
│   │   └── annotator.py        # Bounding box frame annotation
│   └── web/                    # Flask web UI
│       ├── app.py              # Flask application factory
│       ├── api.py              # REST API endpoints
│       ├── templates/          # HTML templates
│       └── static/             # CSS, JS assets
├── tests/
│   ├── test_database.py
│   ├── test_video_store.py
│   ├── test_analysis.py
│   ├── test_detection.py       # Object detection tests
│   ├── test_daylight.py        # Daylight gating tests
│   ├── test_thumbnail.py       # Thumbnail generation tests
│   ├── test_test_capture.py    # On-demand capture tests
│   ├── test_timelapse.py       # Timelapse generation tests
│   ├── test_web.py
│   └── test_integration.py
├── data/
│   ├── videos/                 # Captured video clips
│   ├── annotated/              # Annotated key-frame images
│   ├── timelapses/             # Generated daily time-lapse videos
│   └── wildlife.db             # SQLite database
├── main.py                     # Main entry point
├── config.yaml.example         # Example configuration (copy to config.yaml)
├── requirements.txt
├── requirements-pi.txt         # Raspberry Pi specific dependencies
└── README.md
```

## Future Ideas

### New Sensors & Hardware

- **Microphone / USB audio** — Record ambient sound alongside video; use an audio classification model (e.g., BirdNET) to identify species by call, especially at night or when animals are out of frame
- **BME280 / BME680 environmental sensor** — Log temperature, humidity, barometric pressure, and air quality per detection; correlate weather conditions with animal activity patterns
- **IR camera module / NoIR + IR LEDs** — Enable night vision captures with a dual-camera setup or a single NoIR camera with an IR illuminator ring
- **Ultrasonic range sensor (HC-SR04)** — Estimate animal distance and size; filter out detections that are too far away or too close
- **Light/lux sensor (BH1750)** — More precise daylight measurement than the API; adapt camera exposure settings automatically
- **Rain sensor** — Tag captures with weather conditions; optionally pause captures during heavy rain

### Software Features

- **Bird call identification** — Run BirdNET or a similar lightweight audio model to identify bird species by sound
- **Animal tracking across detections** — Use bounding boxes and timestamps to infer whether the same individual is returning
- **Push notifications** — Send alerts via Telegram, Pushover, or ntfy.sh when a specific animal is detected
- **Live MJPEG/HLS stream** — Add a `/live` page to the web UI showing the camera feed in real time
- **Multi-camera support** — Run multiple camera modules or USB cameras, tagged by location
- **Heatmap visualization** — Aggregate bounding box positions over time to show where animals most frequently appear
- **Export & sharing** — Export detection data as CSV; generate shareable daily/weekly summary reports
- **Custom model fine-tuning** — Collect labeled detections and fine-tune a classifier for the animals in your area
- **Seasonal analytics** — Long-term trends showing which species appear in which months

## License

MIT
