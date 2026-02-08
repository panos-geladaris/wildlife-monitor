"""Tests for the on-demand test capture feature."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.capture.camera import CaptureReason, VideoMetadata

try:
    from src.capture.capture_service import CaptureService
    HAS_CAPTURE_SERVICE = True
except ImportError:
    HAS_CAPTURE_SERVICE = False

try:
    from src.web.app import create_app
    from src.storage.database import Database, Detection
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


@pytest.mark.skipif(not HAS_CAPTURE_SERVICE, reason="apscheduler not installed")
class TestTriggerManualCaptureDuration:
    """Step 2: trigger_manual_capture passes duration to Camera."""

    def test_duration_forwarded_to_camera(self):
        camera = MagicMock()
        camera.capture_video.return_value = VideoMetadata(
            filepath=Path("/tmp/test.mp4"),
            timestamp=datetime.now(),
            duration_seconds=4.0,
            reason=CaptureReason.MANUAL,
            resolution=(1280, 720),
        )

        with patch.object(CaptureService, "__init__", lambda self, *a, **kw: None):
            svc = CaptureService.__new__(CaptureService)
            svc._camera = camera
            svc._on_capture_callback = None

            svc.trigger_manual_capture(duration=4.0)

        camera.capture_video.assert_called_once_with(
            duration=4.0, reason=CaptureReason.MANUAL
        )

    def test_default_duration_is_none(self):
        camera = MagicMock()
        camera.capture_video.return_value = VideoMetadata(
            filepath=Path("/tmp/test.mp4"),
            timestamp=datetime.now(),
            duration_seconds=2.0,
            reason=CaptureReason.MANUAL,
            resolution=(1280, 720),
        )

        with patch.object(CaptureService, "__init__", lambda self, *a, **kw: None):
            svc = CaptureService.__new__(CaptureService)
            svc._camera = camera
            svc._on_capture_callback = None

            svc.trigger_manual_capture()

        camera.capture_video.assert_called_once_with(
            duration=None, reason=CaptureReason.MANUAL
        )


@pytest.mark.skipif(not HAS_FLASK, reason="flask not installed")
class TestTestCaptureEndpoint:
    """Step 3: POST /api/test-capture endpoint tests."""

    @pytest.fixture()
    def app_with_monitor(self, tmp_path):
        db_path = tmp_path / "test.db"
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        app = create_app(video_dir=video_dir, db_path=db_path)

        video_file = video_dir / "manual_20250208_143022.mp4"
        video_file.write_text("fake video")

        monitor = MagicMock()
        db = Database(db_path)

        metadata = VideoMetadata(
            filepath=video_file,
            timestamp=datetime.now(),
            duration_seconds=4.0,
            reason=CaptureReason.MANUAL,
            resolution=(1280, 720),
        )

        def fake_trigger(duration=None):
            detection = Detection(
                timestamp=metadata.timestamp,
                video_path=str(metadata.filepath),
                trigger_type=metadata.reason.value,
                animal_class="bird",
                confidence=0.87,
                analyzed=True,
            )
            db.add_detection(detection)
            return metadata

        monitor._capture_service.trigger_manual_capture.side_effect = fake_trigger

        app.config["MONITOR"] = monitor

        return app, monitor, db

    def test_returns_detection_id(self, app_with_monitor):
        app, monitor, db = app_with_monitor
        with app.test_client() as client:
            resp = client.post("/api/test-capture")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "detection_id" in data
            assert data["detection_id"] == 1
            assert data["animal_class"] == "bird"
            assert data["confidence"] == 0.87
            assert data["message"] == "Test capture complete"

    def test_calls_capture_with_4s(self, app_with_monitor):
        app, monitor, db = app_with_monitor
        with app.test_client() as client:
            client.post("/api/test-capture")
        monitor._capture_service.trigger_manual_capture.assert_called_once_with(
            duration=4.0
        )

    def test_does_not_double_call_on_video_captured(self, app_with_monitor):
        app, monitor, db = app_with_monitor
        with app.test_client() as client:
            client.post("/api/test-capture")
        monitor._on_video_captured.assert_not_called()

    def test_503_when_no_monitor(self, tmp_path):
        app = create_app(
            video_dir=tmp_path / "videos",
            db_path=tmp_path / "test.db",
        )
        with app.test_client() as client:
            resp = client.post("/api/test-capture")
            assert resp.status_code == 503
            data = resp.get_json()
            assert "web-only" in data["error"]

    def test_503_when_no_capture_service(self, tmp_path):
        app = create_app(
            video_dir=tmp_path / "videos",
            db_path=tmp_path / "test.db",
        )
        monitor = MagicMock()
        monitor._capture_service = None
        app.config["MONITOR"] = monitor

        with app.test_client() as client:
            resp = client.post("/api/test-capture")
            assert resp.status_code == 503

    def test_500_on_capture_exception(self, tmp_path):
        app = create_app(
            video_dir=tmp_path / "videos",
            db_path=tmp_path / "test.db",
        )
        monitor = MagicMock()
        monitor._capture_service.trigger_manual_capture.side_effect = RuntimeError(
            "Camera busy"
        )
        app.config["MONITOR"] = monitor

        with app.test_client() as client:
            resp = client.post("/api/test-capture")
            assert resp.status_code == 500
            assert "Camera busy" in resp.get_json()["error"]
