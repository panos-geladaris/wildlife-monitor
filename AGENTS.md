# AGENTS.md

## Project Overview

Raspberry Pi wildlife monitoring system that detects motion via PIR sensor, captures video, classifies animals using TorchVision (MobileNetV3 + SSDLite320), stores results in SQLite, and serves a Flask web UI.

## Architecture

Four modules under `src/`, orchestrated by `main.py` (`WildlifeMonitor` class):

- **`src/capture/`** — PIR motion detection, camera recording (picamera2), scheduled captures (APScheduler), daylight gating
- **`src/analysis/`** — MobileNetV3 classification, SSDLite320 object detection, frame extraction (OpenCV), bounding box annotation
- **`src/storage/`** — SQLite database (Detection, Timelapse, DailySummary), video file management, thumbnail generation, timelapse generation, empty detection cleanup
- **`src/web/`** — Flask app factory (`app.py`), REST API (`api.py`), Jinja2 templates, static assets (Bootstrap 5, Chart.js)

Data flow: Capture → Analysis → Storage → Web UI.

## Development Environment

- **Python 3** with `venv` (virtual environment at `venv/` or `.venv/`)
- **Dependencies**: `pip install -r requirements.txt` (core); `requirements-pi.txt` adds Pi-specific packages (`picamera2`, `RPi.GPIO`)
- **ML deps** (torch, torchvision) are optional — the system degrades gracefully without them
- **Simulation mode**: auto-detected when `RPi.GPIO` is unavailable, or forced with `--simulate`. Uses simulated camera and motion detector.
- **Config**: `config.yaml` (gitignored) — copy from `config.yaml.example`

## Running

```bash
python main.py              # Full system (capture + analysis + web)
python main.py --web-only   # Web UI only (port 5001)
python main.py --simulate   # Force simulation mode
```

## Testing

```bash
pytest tests/               # All tests
pytest tests/ -v            # Verbose
pytest tests/test_database.py  # Single test file
```

Tests use standard `pytest` with no special fixtures or plugins. Test files mirror module names (e.g., `test_analysis.py`, `test_cleanup.py`).

## Code Conventions

- **Style**: Standard Python conventions; no formatter or linter is configured
- **Type hints**: Used on function signatures (especially public APIs); `Optional`, `Path`, `list`, `dict` are common
- **Imports**: Standard library → third-party → local (`from src.module import ...`)
- **Dataclasses**: Used for data models (`Detection`, `Timelapse`, `DailySummary`, `CaptureServiceConfig`, `ClassificationResult`)
- **Logging**: Use `logging.getLogger(__name__)` — no print statements in library code
- **Config loading**: YAML config read in `WildlifeMonitor._load_config_data()`; capture config via `src/capture/config.py:load_config()`
- **Error handling**: Modules degrade gracefully when optional dependencies (torch, picamera2, RPi.GPIO) are missing — catch `ImportError` and log a warning
- **Flask patterns**: App factory in `app.py:create_app()`, blueprints for API routes in `api.py`
- **Database**: Raw SQLite via `sqlite3` module — no ORM. Queries are inline strings in `database.py` methods.
- **File paths**: Use `pathlib.Path` throughout, not string concatenation

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, `WildlifeMonitor` orchestrator class, CLI argument parsing |
| `src/capture/capture_service.py` | Main capture orchestrator (motion + scheduled + manual) |
| `src/capture/config.py` | `CaptureServiceConfig` dataclass and YAML loader |
| `src/storage/database.py` | All SQLite schema and CRUD operations |
| `src/analysis/classifier.py` | Animal classification pipeline |
| `src/analysis/detector.py` | Object detection orchestrator |
| `src/web/app.py` | Flask app factory |
| `src/web/api.py` | All REST API endpoints |
| `config.yaml.example` | Reference configuration with all options documented |

## Data Directories

- `data/videos/` — Captured MP4 clips
- `data/videos/thumbnails/` — Video thumbnails (JPEG)
- `data/annotated/<detection_id>/` — Annotated key-frame images with bounding boxes
- `data/timelapses/` — Generated daily timelapse MP4s
- `data/wildlife.db` — SQLite database

All `data/` contents are gitignored except `.gitkeep` files.

## Important Notes

- **Dual-platform**: Code must work on both Raspberry Pi (ARM, real hardware) and macOS/Linux dev machines (simulation mode). Never add hard dependencies on Pi-specific packages.
- **Web UI port**: Default is 5001.
- **Database migrations**: There is no migration system — schema changes go directly in `Database._init_db()` using `CREATE TABLE IF NOT EXISTS`.
- **Background processing**: Video analysis runs in daemon threads spawned from `_on_video_captured()`. Scheduled jobs (cleanup, timelapse) use APScheduler `BackgroundScheduler`.

## Development practices

- Always use Red/Green Test-driven development for the implementation
- Before starting with the implementation, provide the plan for review and ask the user if they want to proceed
