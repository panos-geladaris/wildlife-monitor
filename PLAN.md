# Wildlife Monitor - Project Plan

## Goal

Build a Raspberry Pi-based system to detect, record, and classify animals passing by a house window using computer vision.

## System Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raspberry Pi   │────▶│  Video Storage   │────▶│  Web UI (Flask) │
│  + PIR Sensor   │     │  + ML Analysis   │     │  View Results   │
│  + Camera       │     │  (TorchVision)   │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Computer | Raspberry Pi 4 | Main processing unit |
| Camera | Pi Camera Module 3 | Video capture (12MP, autofocus, HDR) |
| Sensor | HC-SR501 PIR | Motion detection |

## Modules

### 1. Capture Module ✅
**Status:** Complete

**Components:**
- `motion_detector.py` - PIR sensor monitoring via GPIO
- `camera.py` - Video recording with picamera2
- `scheduler.py` - Hourly capture scheduling (APScheduler)
- `capture_service.py` - Orchestrates all capture components

**Features:**
- Motion-triggered 1-2 second video capture
- Scheduled hourly video capture
- Configurable resolution, duration, cooldown
- Simulation mode for development without Pi hardware
- Callback system for downstream processing

---

### 2. Analysis Module ✅
**Status:** Complete

**Components:**
- `model.py` - TorchVision model loading and inference (MobileNetV3, ResNet18)
- `classifier.py` - Animal classification with confidence thresholds
- `frame_extractor.py` - Extract frames from video using OpenCV

**Features:**
- MobileNetV3 pretrained on ImageNet (lightweight for Pi)
- Maps 200+ ImageNet classes to simplified animal categories
- Configurable confidence threshold
- Multi-frame video analysis (best result selection)
- Context manager support for memory management

**Animal Classes Detected:**
- Birds (60+ species mapped)
- Cats (domestic and wild)
- Dogs (100+ breeds)
- Squirrels, Foxes, Rabbits
- Deer, Hedgehogs, Mice

**Output:**
- ClassificationResult with animal_class, confidence, timestamp
- Top-k predictions for debugging
- Frame number and timestamp for video analysis

---

### 3. Storage Module ✅
**Status:** Complete

**Components:**
- `database.py` - SQLite database operations (Detection, DailySummary)
- `video_store.py` - Video file management (VideoStore)

**Database Schema:**
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    video_path TEXT NOT NULL,
    trigger_type TEXT NOT NULL,  -- 'motion', 'scheduled', 'manual'
    animal_class TEXT,
    confidence REAL,
    analyzed BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE daily_summary (
    id INTEGER PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_detections INTEGER,
    animal_counts JSON,  -- {"bird": 5, "cat": 2, ...}
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Features:**
- Store detection metadata with CRUD operations
- Query detections by trigger type, date range, animal class
- Daily summary generation with animal counts
- Video file listing and metadata parsing
- Automatic cleanup of old videos (configurable retention)
- Storage usage statistics and disk space monitoring

---

### 4. Web UI Module ✅
**Status:** Complete

**Components:**
- `app.py` - Flask application factory
- `api.py` - REST API endpoints
- `templates/` - HTML templates (base, index, gallery, detection, statistics)
- `static/css/` - Custom styles
- `static/js/` - Utility functions and auto-refresh

**Pages:**
- **Dashboard** (`/`) - Live status, recent detections, today's summary
- **Gallery** (`/gallery`) - Browse all detections with filters and pagination
- **Detection Details** (`/detection/:id`) - View video, classification results
- **Statistics** (`/statistics`) - Charts showing detection trends over time

**Features:**
- Filter by animal type, trigger type
- Video playback with HTML5 player
- Daily detection charts (Chart.js)
- Animal breakdown charts
- Mobile-responsive design (Bootstrap 5)
- Auto-refresh dashboard (30s)
- Loading states and error handling

**API Endpoints:**
```
GET  /api/status              - System status (storage, disk, counts)
GET  /api/detections          - List detections (with filters)
GET  /api/detections/:id      - Get single detection
GET  /api/videos              - List video files
GET  /api/stats/daily         - Daily detection counts
GET  /api/stats/animals       - Animal type breakdown
GET  /api/stats/summary       - Dashboard summary
GET  /videos/:filename        - Serve video files
```

**Running the Web UI:**
```bash
# Development
python -m src.web.app

# Access at http://localhost:5001
# On network: http://<raspberry-pi-ip>:5001
```

---

## Implementation Order

1. ✅ **Capture Module** - Motion detection, camera, scheduling
2. ✅ **Storage Module** - Database and video management
3. ✅ **Analysis Module** - ML classification pipeline
4. ✅ **Web UI Module** - Flask app for viewing results
5. 🔲 **Integration** - Connect all modules, main.py entry point
6. 🔲 **Deployment** - Systemd service, auto-start on Pi

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML Model | MobileNetV3 | Lightweight, runs on Pi 4 |
| Database | SQLite | Simple, no server needed |
| Web Framework | Flask | Lightweight, sufficient for local UI |
| Scheduler | APScheduler | Python-native, easy integration |
| Video Format | MP4 (H264) | Good compression, broad support |

## File Structure

```
wildlife-monitor/
├── src/
│   ├── capture/           # ✅ Complete
│   │   ├── motion_detector.py
│   │   ├── camera.py
│   │   ├── scheduler.py
│   │   ├── config.py
│   │   └── capture_service.py
│   ├── storage/           # ✅ Complete
│   │   ├── database.py
│   │   └── video_store.py
│   ├── analysis/          # ✅ Complete
│   │   ├── model.py
│   │   ├── classifier.py
│   │   └── frame_extractor.py
│   ├── storage/           # ✅ Complete
│   │   ├── database.py
│   │   └── video_store.py
│   └── web/               # ✅ Complete
│       ├── app.py
│       ├── api.py
│       ├── templates/
│       │   ├── base.html
│       │   ├── index.html
│       │   ├── gallery.html
│       │   ├── detection.html
│       │   └── statistics.html
│       └── static/
│           ├── css/style.css
│           └── js/app.js
├── data/
│   ├── videos/
│   └── wildlife.db
├── models/
│   └── (pretrained weights)
├── main.py                # Entry point
├── requirements.txt
├── requirements-pi.txt
├── README.md
└── PLAN.md
```

## Future Enhancements

- [ ] Live streaming view in web UI
- [ ] Push notifications for specific animal detections
- [ ] Cloud backup for videos/detections
- [ ] Multi-camera support
- [ ] Night vision / IR camera support
- [ ] Species-level bird identification
- [ ] Weather correlation analysis
