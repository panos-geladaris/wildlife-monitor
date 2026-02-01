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

### 2. Analysis Module 🔲
**Status:** Planned

**Components:**
- `model.py` - TorchVision model loading and inference
- `classifier.py` - Animal classification logic
- `frame_extractor.py` - Extract frames from video for analysis

**Approach:**
- Use MobileNetV3 (lightweight, suitable for Pi)
- Pretrained on ImageNet, filter for animal classes
- Optional: Fine-tune on iNaturalist dataset for better wildlife detection

**Animal Classes to Detect:**
- Birds (various species)
- Cats
- Dogs
- Squirrels
- Foxes
- Rabbits
- Other small mammals

**Output:**
- Detected animal class
- Confidence score
- Bounding box (if using detection model)
- Timestamp

---

### 3. Storage Module 🔲
**Status:** Planned

**Components:**
- `database.py` - SQLite database operations
- `video_store.py` - Video file management

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
- Store detection metadata
- Link to video files
- Daily/weekly summaries
- Cleanup old videos (configurable retention)

---

### 4. Web UI Module 🔲
**Status:** Planned

**Components:**
- `app.py` - Flask application
- `templates/` - HTML templates
- `static/` - CSS, JS assets

**Pages:**
- **Dashboard** - Live status, recent detections, daily summary
- **Gallery** - Browse all videos with thumbnails
- **Detection Details** - View video, classification results
- **Statistics** - Charts showing detection trends over time

**Features:**
- Filter by animal type, date range, trigger type
- Video playback
- Daily/weekly detection charts
- Mobile-responsive design

**API Endpoints:**
```
GET  /api/detections          - List detections (with filters)
GET  /api/detections/:id      - Get single detection
GET  /api/videos/:filename    - Serve video file
GET  /api/stats/daily         - Daily detection counts
GET  /api/stats/animals       - Animal type breakdown
GET  /api/status              - System status
```

---

## Implementation Order

1. ✅ **Capture Module** - Motion detection, camera, scheduling
2. 🔲 **Storage Module** - Database and video management
3. 🔲 **Analysis Module** - ML classification pipeline
4. 🔲 **Web UI Module** - Flask app for viewing results
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
│   │   └── capture_service.py
│   ├── analysis/          # 🔲 Planned
│   │   ├── model.py
│   │   ├── classifier.py
│   │   └── frame_extractor.py
│   ├── storage/           # 🔲 Planned
│   │   ├── database.py
│   │   └── video_store.py
│   └── web/               # 🔲 Planned
│       ├── app.py
│       ├── templates/
│       └── static/
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
