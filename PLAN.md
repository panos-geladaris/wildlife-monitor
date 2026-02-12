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
├── main.py                # Main entry point (integration)
├── requirements.txt
├── requirements-pi.txt
├── README.md
└── PLAN.md
```

### 6. Object Detection Module 🔲
**Status:** Planned

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
- [ ] Create `DetectionModelLoader` class mirroring `ModelLoader` pattern
- [ ] Load `ssdlite320_mobilenet_v3_large` with COCO weights
- [ ] Implement `predict(image) -> list[dict]` returning boxes, labels, scores
- [ ] Add COCO label mapping (id → name) for the 91 COCO classes
- [ ] Add filtering for animal-related COCO classes only

**Step 2: Object Detector** (`src/analysis/detector.py`)
- [ ] Create `ObjectDetector` class with `detect_video(video_path, num_frames, score_threshold) -> dict[int, list[DetectionBox]]`
- [ ] Reuse `FrameExtractor.extract_key_frames()` for frame selection
- [ ] For each frame: run detection → filter by score threshold → keep top-N boxes
- [ ] Optional crop-classify: for each box, crop region from frame and run `AnimalClassifier.classify_image()` to refine label
- [ ] Return results keyed by frame number

**Step 3: Frame Annotator** (`src/analysis/annotator.py`)
- [ ] Create `annotate_frame(image, boxes) -> Image` that draws rectangles + labels via OpenCV
- [ ] Create `save_annotated_frames(detection_id, frames_with_boxes, output_dir)` to save JPEGs
- [ ] Use color coding per animal class

**Step 4: Database Changes** (`src/storage/database.py`)
- [ ] Add `frame_objects` table creation to `_init_db()`
- [ ] Add `add_frame_objects(detection_id, boxes: list[DetectionBox])` method
- [ ] Add `get_frame_objects(detection_id) -> list[DetectionBox]` method
- [ ] Add `delete_frame_objects(detection_id)` for cascade cleanup
- [ ] Update bulk delete to also clean up annotated frame files

**Step 5: Pipeline Integration** (`main.py`)
- [ ] Add `_init_detector()` method to `WildlifeMonitor`
- [ ] In `_on_video_captured()`, after classification: run detection → store boxes → save annotated frames
- [ ] Add `--no-detection` CLI flag to disable object detection
- [ ] Update `config.yaml` with detection settings (score threshold, max boxes, num frames)

**Step 6: Web UI Updates**
- [ ] Add `GET /api/detections/<id>/objects` endpoint returning bounding box JSON
- [ ] Serve annotated frame images at `/annotated/<detection_id>/<filename>`
- [ ] Update detection detail template to show annotated key-frame gallery below video player
- [ ] Update `__init__.py` exports for new classes

**Step 7: Tests**
- [ ] Unit tests for `DetectionModelLoader` (model loading, dummy image inference)
- [ ] Unit tests for `ObjectDetector` (video detection flow, crop-classify)
- [ ] Unit tests for `annotate_frame` (output image dimensions, box drawing)
- [ ] Integration tests for DB `frame_objects` CRUD
- [ ] Integration tests for full pipeline (capture → detect → annotate → API)

---

### 7. On-Demand Test Capture & Analysis 🔲
**Status:** Planned

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
- [ ] In `_start_web_server()`, store `self` on `app.config["MONITOR"]` after calling `create_app()`

**Step 2: Add duration parameter to manual capture** (`src/capture/capture_service.py`)
- [ ] Add optional `duration: float = None` parameter to `trigger_manual_capture()`
- [ ] Pass duration to `self._camera.capture_video(duration=duration, reason=CaptureReason.MANUAL)`

**Step 3: New API endpoint** (`src/web/api.py`)
- [ ] Add `POST /api/test-capture` endpoint
- [ ] Read `current_app.config["MONITOR"]` to access the monitor instance
- [ ] Call `monitor._capture_service.trigger_manual_capture(duration=4.0)` to capture
- [ ] Call `monitor._on_video_captured(metadata)` to run the full analysis pipeline
- [ ] Query the database for the newly created detection to get classification results
- [ ] Return `{"detection_id": id, "video_filename": ..., "animal_class": ..., "confidence": ...}`
- [ ] Return 503 if capture service is unavailable (web-only mode)

**Step 4: Dashboard UI update** (`src/web/templates/index.html`)
- [ ] Add "Test Capture" card with description and capture button
- [ ] On click: disable button, show spinner text ("Capturing & Analyzing...")
- [ ] POST to `/api/test-capture`
- [ ] On success: redirect to `/detection/<detection_id>`
- [ ] On error: show alert, re-enable button

**Step 5: Tests**
- [ ] Unit test for `trigger_manual_capture(duration=4.0)` passes duration to camera
- [ ] Integration test for `POST /api/test-capture` returns detection ID
- [ ] Test 503 response when monitor has no capture service (web-only mode)

---

### 8. Automatic Cleanup of Empty Detections 🔲
**Status:** Planned

**Goal:** Automatically remove detections older than a configurable age where neither the classifier nor the object detector recognised any animal. This saves storage on the Pi by discarding videos that captured motion (e.g. wind, shadows, cars) but contained nothing of interest.

**Definition of "empty":**
A detection is empty when **all** of the following are true:
- `analyzed = TRUE` (analysis has completed — never delete unprocessed videos)
- `animal_class IS NULL OR animal_class = 'unknown'` (classifier found nothing)
- No rows exist in `frame_objects` for that `detection_id` (detector found nothing)

**What gets deleted per detection:**
1. Database row in `detections` (cascade deletes `frame_objects` rows)
2. Video file at `detection.video_path`
3. Thumbnail at `data/videos/thumbnails/<video_stem>.jpg`
4. Annotated frames directory at `data/annotated/<detection_id>/`

**Config:**
```yaml
# Automatic cleanup of empty detections
cleanup:
  enabled: true
  max_age_hours: 24          # Delete empty detections older than this
  interval_hours: 6          # How often the cleanup job runs
```

**Architecture Fit:**
- Reuses the existing `_cleanup_detection_artifacts()` helper in `api.py` — extract it to `src/storage/cleanup.py` so both the API delete endpoints and the scheduled cleanup can share the same logic
- Runs on an APScheduler interval job inside `WildlifeMonitor`, alongside the existing capture scheduler
- Daily summary is recalculated after cleanup to keep counts accurate

---

#### Implementation Steps

**Step 1: Database query** (`src/storage/database.py`)
- [ ] Add `get_empty_detections(before: datetime) -> list[Detection]` method
- [ ] Query: `SELECT * FROM detections WHERE analyzed = 1 AND (animal_class IS NULL OR animal_class = 'unknown') AND timestamp < ? AND id NOT IN (SELECT DISTINCT detection_id FROM frame_objects)`

**Step 2: Shared cleanup helper** (`src/storage/cleanup.py`)
- [ ] Extract `_cleanup_detection_artifacts()` from `src/web/api.py` into a standalone function `cleanup_detection(db, detection, video_dir)` that deletes: video file, thumbnail, annotated frames dir, and DB record
- [ ] Update `src/web/api.py` delete endpoints to call the shared helper instead of the inline version

**Step 3: Cleanup service** (`src/storage/cleanup.py`)
- [ ] Add `run_cleanup(db, video_dir, max_age_hours)` function that:
  1. Calls `db.get_empty_detections(cutoff)` to find candidates
  2. Calls `cleanup_detection()` for each
  3. Calls `db.update_daily_summary()` to refresh counts
  4. Returns count of deleted detections
  5. Logs a summary line

**Step 4: Config** (`src/capture/config.py` + `config.yaml`)
- [ ] Add `cleanup_enabled`, `cleanup_max_age_hours`, `cleanup_interval_hours` to `CaptureServiceConfig`
- [ ] Load from `cleanup` section in `config.yaml`
- [ ] Add defaults: `enabled=true`, `max_age_hours=24`, `interval_hours=6`

**Step 5: Integration** (`main.py`)
- [ ] Add `_init_cleanup_scheduler()` method that schedules `run_cleanup` on an APScheduler `IntervalTrigger`
- [ ] Call it from `start()` when cleanup is enabled
- [ ] Shut it down in `stop()`
- [ ] Add `--no-cleanup` CLI flag

**Step 6: Tests**
- [ ] Unit test for `get_empty_detections` — returns only analyzed detections with no animal and no frame_objects
- [ ] Unit test for `get_empty_detections` — does not return detections that have frame_objects
- [ ] Unit test for `get_empty_detections` — does not return unanalyzed detections
- [ ] Unit test for `run_cleanup` — deletes video, thumbnail, annotated dir, and DB record
- [ ] Unit test for `run_cleanup` — skips detections newer than max_age_hours
- [ ] Integration test verifying API delete endpoints still work after extracting the shared helper

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
