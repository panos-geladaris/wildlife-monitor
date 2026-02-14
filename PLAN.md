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

### 5. Integration ✅
**Status:** Complete

**Components:**
- `main.py` - Main entry point that orchestrates all modules

**Features:**
- Full system integration (capture → analysis → storage → web)
- Multiple run modes:
  - Full system: `python main.py`
  - Capture only: `python main.py --capture-only`
  - Web only: `python main.py --web-only`
  - Analyze single video: `python main.py --analyze video.mp4`
- Auto-detection of simulation mode (Pi hardware vs development)
- Configurable via command-line arguments and config.yaml
- Graceful shutdown with Ctrl+C
- Background ML analysis of captured videos
- Automatic database updates on new captures

**Command-line Options:**
```
python main.py [OPTIONS]

Options:
  --config PATH       Path to config.yaml file
  --video-dir PATH    Directory for video files (default: data/videos)
  --db-path PATH      Path to SQLite database (default: data/wildlife.db)
  --port PORT         Web UI port (default: 5001)
  --capture-only      Run capture service without web UI
  --web-only          Run web UI only (no capture)
  --no-analysis       Disable ML analysis (just record videos)
  --simulate          Force simulation mode (no Pi hardware required)
  --analyze VIDEO     Analyze a single video file and exit
  -v, --verbose       Enable verbose logging
```

---

## Implementation Order

1. ✅ **Capture Module** - Motion detection, camera, scheduling
2. ✅ **Storage Module** - Database and video management
3. ✅ **Analysis Module** - ML classification pipeline
4. ✅ **Web UI Module** - Flask app for viewing results
5. ✅ **Integration** - Connect all modules, main.py entry point
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
│   │   ├── daylight.py
│   │   └── capture_service.py
│   ├── storage/           # ✅ Complete
│   │   ├── database.py
│   │   ├── video_store.py
│   │   ├── thumbnail.py
│   │   └── cleanup.py
│   ├── analysis/          # ✅ Complete
│   │   ├── model.py
│   │   ├── classifier.py
│   │   ├── frame_extractor.py
│   │   ├── detection_model.py
│   │   ├── detector.py
│   │   └── annotator.py
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
│   ├── annotated/
│   └── wildlife.db
├── main.py                # Main entry point (integration)
├── config.yaml.example    # Example configuration
├── requirements.txt
├── requirements-pi.txt
├── README.md
└── PLAN.md
```

### 6. Object Detection Module ✅
**Status:** Complete

**Goal:** Add bounding box object detection to locate and annotate animals in video frames, complementing the existing whole-image classification.

**Model Choice: SSDLite320 + MobileNetV3-Large**
- `torchvision.models.detection.ssdlite320_mobilenet_v3_large`
- Lightest detection model in TorchVision — suitable for Pi 4 CPU inference
- COCO-pretrained (covers bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
- For animals not in COCO (deer, fox, rabbit, squirrel, etc.): crop detected region → run existing MobileNetV3 classifier on the crop to map to `ANIMAL_CLASSES`

**Components (new files):**
- `src/analysis/detection_model.py` — Model loader for SSDLite320, preprocessing, inference
- `src/analysis/detector.py` — Orchestrates frame extraction + detection + optional crop-classification
- `src/analysis/annotator.py` — Draw bounding boxes and labels on frames, save annotated JPEGs

**Core Data Structure:**
```python
@dataclass
class DetectionBox:
    label: str              # COCO label or mapped animal class
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    img_width: int
    img_height: int
    frame_number: int
    frame_timestamp: float
    animal_class: Optional[str] = None        # from crop→classifier refinement
    animal_confidence: Optional[float] = None
```

**Database Schema (new table):**
```sql
CREATE TABLE IF NOT EXISTS frame_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    frame_timestamp REAL,
    label TEXT,
    score REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    img_width INTEGER,
    img_height INTEGER,
    animal_class TEXT,
    animal_confidence REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(detection_id) REFERENCES detections(id) ON DELETE CASCADE
);
```

**Annotated Output:**
- For each analyzed frame, save annotated JPEG: `data/annotated/<detection_id>/frame_<frame_number>.jpg`
- Bounding box rectangle with label + confidence drawn via OpenCV

**Performance Guardrails (Pi 4):**
- Analyze 3–5 key frames per clip (reuse `FrameExtractor`)
- Keep top-5 boxes per frame
- Use `torch.inference_mode()` + `torch.set_num_threads(4)`
- SSDLite expects 320px input (small and fast)

**Integration Pipeline:**
```
Capture → DB insert → Thumbnail → Classification → Detection → Annotate → Update DB → Daily Summary
```

**Web UI Changes:**
- Detection detail page: gallery of annotated key frames below the video player
- New API endpoints:
  - `GET /api/detections/<id>/objects` — bounding box data (JSON)
  - `GET /annotated/<detection_id>/frame_<frame_number>.jpg` — serve annotated images

**Coexistence Strategy:**
- Detection runs after classification in the pipeline
- Detector provides *localization* (where the animal is)
- Classifier provides *taxonomy* (what the animal is)
- For COCO-covered animals: use detector label directly
- For non-COCO animals: crop detected region → classify with existing MobileNetV3

---

#### Implementation Steps

**Step 1: Detection Model Loader** (`src/analysis/detection_model.py`)
- [x] Create `DetectionModelLoader` class mirroring `ModelLoader` pattern
- [x] Load `ssdlite320_mobilenet_v3_large` with COCO weights
- [x] Implement `predict(image) -> list[dict]` returning boxes, labels, scores
- [x] Add COCO label mapping (id → name) for the 91 COCO classes
- [x] Add filtering for animal-related COCO classes only

**Step 2: Object Detector** (`src/analysis/detector.py`)
- [x] Create `ObjectDetector` class with `detect_video(video_path, num_frames, score_threshold) -> dict[int, list[DetectionBox]]`
- [x] Reuse `FrameExtractor.extract_key_frames()` for frame selection
- [x] For each frame: run detection → filter by score threshold → keep top-N boxes
- [x] Optional crop-classify: for each box, crop region from frame and run `AnimalClassifier.classify_image()` to refine label
- [x] Return results keyed by frame number

**Step 3: Frame Annotator** (`src/analysis/annotator.py`)
- [x] Create `annotate_frame(image, boxes) -> Image` that draws rectangles + labels via OpenCV
- [x] Create `save_annotated_frames(detection_id, frames_with_boxes, output_dir)` to save JPEGs
- [x] Use color coding per animal class

**Step 4: Database Changes** (`src/storage/database.py`)
- [x] Add `frame_objects` table creation to `_init_db()`
- [x] Add `add_frame_objects(detection_id, boxes: list[DetectionBox])` method
- [x] Add `get_frame_objects(detection_id) -> list[DetectionBox]` method
- [x] Add `delete_frame_objects(detection_id)` for cascade cleanup
- [x] Update bulk delete to also clean up annotated frame files

**Step 5: Pipeline Integration** (`main.py`)
- [x] Add `_init_detector()` method to `WildlifeMonitor`
- [x] In `_on_video_captured()`, after classification: run detection → store boxes → save annotated frames
- [x] Add `--no-detection` CLI flag to disable object detection
- [x] Update `config.yaml` with detection settings (score threshold, max boxes, num frames)

**Step 6: Web UI Updates**
- [x] Add `GET /api/detections/<id>/objects` endpoint returning bounding box JSON
- [x] Serve annotated frame images at `/annotated/<detection_id>/<filename>`
- [x] Update detection detail template to show annotated key-frame gallery below video player
- [x] Update `__init__.py` exports for new classes

**Step 7: Tests**
- [x] Unit tests for `DetectionModelLoader` (model loading, dummy image inference)
- [x] Unit tests for `ObjectDetector` (video detection flow, crop-classify)
- [x] Unit tests for `annotate_frame` (output image dimensions, box drawing)
- [x] Integration tests for DB `frame_objects` CRUD
- [x] Integration tests for full pipeline (capture → detect → annotate → API)

---

### 7. On-Demand Test Capture & Analysis ✅
**Status:** Complete

**Goal:** Add a button in the web UI that triggers a 4-second video capture, runs the full analysis pipeline (classification + object detection), and redirects to the detection detail page — useful for testing camera angle, lighting, and positioning.

**Architecture Fit:**
- `Camera.capture_video(duration, reason=MANUAL)` already captures video
- `WildlifeMonitor._on_video_captured(metadata)` runs the full pipeline (DB insert → thumbnail → classification → detection → annotate → daily summary)
- The detection detail page (`/detection/<id>`) already shows video, classification, and annotated frames
- Gap: the web layer has no reference to the `WildlifeMonitor` or `CaptureService` instances, and there is no API endpoint to trigger a capture

**API Contract:**
```
POST /api/test-capture
Request body: (none)
Response 200:
{
  "detection_id": 42,
  "video_filename": "manual_20250208_143022.mp4",
  "animal_class": "bird",
  "confidence": 0.87,
  "message": "Test capture complete"
}
Response 503:
{
  "error": "Capture service not available (web-only mode)"
}
```

**UI Location:**
- New "Test Capture" card on the dashboard (`/`) between the status cards and the recent detections table
- Button triggers POST, shows spinner ("Capturing & Analyzing..."), then redirects to `/detection/<id>`
- Button disabled while capture is in progress to prevent double-triggers

**Design Decisions:**
- No new database tables (uses existing `detections` table with `trigger_type = "manual"`)
- No new templates (reuses the existing detection detail page)
- No new config options (4s duration hardcoded in the endpoint)
- No background task queue — request blocks for ~5-6s which is acceptable for a manual test action

---

#### Implementation Steps

**Step 1: Expose monitor instance to Flask** (`main.py`)
- [x] In `_start_web_server()`, store `self` on `app.config["MONITOR"]` after calling `create_app()`

**Step 2: Add duration parameter to manual capture** (`src/capture/capture_service.py`)
- [x] Add optional `duration: float = None` parameter to `trigger_manual_capture()`
- [x] Pass duration to `self._camera.capture_video(duration=duration, reason=CaptureReason.MANUAL)`

**Step 3: New API endpoint** (`src/web/api.py`)
- [x] Add `POST /api/test-capture` endpoint
- [x] Read `current_app.config["MONITOR"]` to access the monitor instance
- [x] Call `monitor._capture_service.trigger_manual_capture(duration=4.0)` to capture
- [x] Call `monitor._on_video_captured(metadata)` to run the full analysis pipeline
- [x] Query the database for the newly created detection to get classification results
- [x] Return `{"detection_id": id, "video_filename": ..., "animal_class": ..., "confidence": ...}`
- [x] Return 503 if capture service is unavailable (web-only mode)

**Step 4: Dashboard UI update** (`src/web/templates/index.html`)
- [x] Add "Test Capture" card with description and capture button
- [x] On click: disable button, show spinner text ("Capturing & Analyzing...")
- [x] POST to `/api/test-capture`
- [x] On success: redirect to `/detection/<detection_id>`
- [x] On error: show alert, re-enable button

**Step 5: Tests**
- [x] Unit test for `trigger_manual_capture(duration=4.0)` passes duration to camera
- [x] Integration test for `POST /api/test-capture` returns detection ID
- [x] Test 503 response when monitor has no capture service (web-only mode)

---

### 9. Daily Time-Lapse Generation 🔲
**Status:** Planned

**Goal:** Every day at a configurable time (default 21:00), generate a time-lapse video from all detections captured that day. Each detection contributes one representative frame. Time-lapses are browsable in a new "Timelapses" page with a gallery view, and each timelapse links to a detail page showing the video and a summary of animals detected that day.

**Data Flow:**
```
Daily APScheduler job (21:00)
  → Query detections for today
  → Extract 1 key frame per detection video (FrameExtractor)
  → Stitch frames into MP4 with OpenCV (cv2.VideoWriter)
  → Generate thumbnail from middle frame
  → Insert record into `timelapses` DB table
  → Store animal summary from constituent detections
```

**Database Schema (new table):**
```sql
CREATE TABLE IF NOT EXISTS timelapses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE NOT NULL,
    video_path TEXT NOT NULL,
    detection_count INTEGER DEFAULT 0,
    animal_counts TEXT,           -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Config:**
```yaml
timelapse:
  enabled: true
  generation_hour: 21          # Hour of day to generate (0-23)
  generation_minute: 0         # Minute of hour
  frame_duration: 0.5          # Seconds each frame is shown
  resolution: [1280, 720]
```

**Web UI:**
- New "Timelapses" link in navbar
- Gallery page (`/timelapses`) — grid of cards with thumbnails, date, detection count, animal badges
- Detail page (`/timelapse/<id>`) — video player + animal breakdown list + delete button

**API Endpoints:**
```
GET    /api/timelapses          — List timelapses (paginated, newest first)
GET    /api/timelapses/<id>     — Single timelapse detail (includes animal_counts)
DELETE /api/timelapses/<id>     — Delete timelapse and its video file
```

---

#### Implementation Steps

**Step 1: Database — new table + queries** (`src/storage/database.py`)
- [ ] Add `Timelapse` dataclass (id, date, video_path, detection_count, animal_counts, created_at)
- [ ] Add `timelapses` table creation to `_init_db()`
- [ ] Add `add_timelapse()` method
- [ ] Add `get_timelapse(timelapse_id)` method
- [ ] Add `get_timelapse_by_date(dt)` method
- [ ] Add `get_timelapses(limit, offset)` method
- [ ] Add `delete_timelapse(timelapse_id)` method

**Step 2: Time-lapse generator** (`src/storage/timelapse.py`)
- [ ] Create `generate_timelapse(db, video_dir, target_date)` function
- [ ] Query all detections for the target date ordered by timestamp
- [ ] Skip if no detections or timelapse already exists for that date
- [ ] Extract middle frame from each detection video using `FrameExtractor`
- [ ] Stitch frames into MP4 using `cv2.VideoWriter` (~2 fps)
- [ ] Save to `data/timelapses/timelapse_YYYYMMDD.mp4`
- [ ] Generate thumbnail using `generate_thumbnail()`
- [ ] Compute `animal_counts` from constituent detections
- [ ] Insert `Timelapse` record into DB
- [ ] Add `run_timelapse_job(db, video_dir, target_date)` scheduler wrapper

**Step 3: Config** (`config.yaml` + `src/capture/config.py`)
- [ ] Add `timelapse` section to `config.yaml.example`
- [ ] Read timelapse config from raw config dict in `main.py` (same pattern as cleanup)

**Step 4: Scheduler integration** (`main.py`)
- [ ] Add `_init_timelapse_scheduler()` using APScheduler `CronTrigger`
- [ ] Add `--no-timelapse` CLI flag
- [ ] Shut down timelapse scheduler in `stop()`

**Step 5: Web UI — API endpoints** (`src/web/api.py`)
- [ ] Add `GET /api/timelapses` endpoint (paginated, newest first)
- [ ] Add `GET /api/timelapses/<id>` endpoint with animal_counts
- [ ] Add `DELETE /api/timelapses/<id>` endpoint

**Step 6: Web UI — Routes and templates**
- [ ] Add `/timelapses` route → `timelapses.html` gallery page
- [ ] Add `/timelapse/<id>` route → `timelapse.html` detail page
- [ ] Add `/timelapse-videos/<filename>` and `/timelapse-thumbnails/<filename>` static routes
- [ ] Add "Timelapses" link to navbar in `base.html`
- [ ] Create `timelapses.html` template (gallery grid with thumbnails)
- [ ] Create `timelapse.html` template (video player + animal breakdown)

**Step 7: Housekeeping**
- [ ] Add `data/timelapses/` to `.gitignore`
- [ ] Update `src/storage/__init__.py` exports

**Step 8: Tests**
- [ ] Unit test: `generate_timelapse` creates an MP4 from mock detections
- [ ] Unit test: `generate_timelapse` skips when no detections exist
- [ ] Unit test: `generate_timelapse` skips when timelapse already exists for that date
- [ ] Unit test: DB CRUD for `timelapses` table
- [ ] Unit test: `GET /api/timelapses` returns correct data
- [ ] Unit test: `DELETE /api/timelapses/<id>` removes video and DB record
- [ ] Integration test: `run_timelapse_job` end-to-end

---

## Future Enhancements

- [ ] Live streaming view in web UI
- [ ] Push notifications for specific animal detections
- [ ] Cloud backup for videos/detections
- [ ] Multi-camera support
- [ ] Night vision / IR camera support
- [ ] Species-level bird identification
- [ ] Weather correlation analysis
- [ ] Annotated MP4 video generation (burn boxes into video file)
- [ ] Canvas-based bounding box overlay synced to video.currentTime in web UI
- [ ] Custom-trained lightweight detector for non-COCO animals (deer, fox, rabbit, squirrel)
