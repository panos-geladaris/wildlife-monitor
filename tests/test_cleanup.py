"""Tests for automatic cleanup of empty detections."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.storage.database import Database, Detection
from src.storage.cleanup import cleanup_detection, run_cleanup


@pytest.fixture
def temp_env():
    """Create a temporary environment with DB, video dir, and artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_dir = tmpdir / "videos"
        video_dir.mkdir()
        db_path = tmpdir / "test.db"
        db = Database(db_path)
        yield db, video_dir, tmpdir


_detection_counter = 0


def _add_detection(db, video_dir, age_hours=48, animal_class=None, analyzed=True, with_wav=False):
    """Helper to create a detection with optional video, thumbnail, and annotated dir."""
    global _detection_counter
    _detection_counter += 1
    ts = datetime.now() - timedelta(hours=age_hours, seconds=_detection_counter)
    video_path = video_dir / f"motion_{ts.strftime('%Y%m%d_%H%M%S')}_{_detection_counter}.mp4"
    video_path.touch()

    if with_wav:
        video_path.with_suffix(".wav").touch()

    detection = Detection(
        timestamp=ts,
        video_path=str(video_path),
        trigger_type="motion",
        animal_class=animal_class,
        analyzed=analyzed,
    )
    detection_id = db.add_detection(detection)

    thumb_dir = video_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    thumb_path = thumb_dir / (video_path.stem + ".jpg")
    thumb_path.touch()

    annotated_dir = video_dir.parent / "annotated" / str(detection_id)
    annotated_dir.mkdir(parents=True)
    (annotated_dir / "frame_0.jpg").touch()

    return detection_id, video_path


class TestGetEmptyDetections:
    def test_returns_analyzed_empty_detections(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, _ = _add_detection(db, video_dir, age_hours=48, analyzed=True)

        cutoff = datetime.now() - timedelta(hours=24)
        results = db.get_empty_detections(before=cutoff)
        assert len(results) == 1
        assert results[0].id == det_id

    def test_excludes_detections_with_animal_class(self, temp_env):
        db, video_dir, _ = temp_env
        _add_detection(db, video_dir, age_hours=48, animal_class="bird", analyzed=True)

        cutoff = datetime.now() - timedelta(hours=24)
        results = db.get_empty_detections(before=cutoff)
        assert len(results) == 0

    def test_excludes_unanalyzed_detections(self, temp_env):
        db, video_dir, _ = temp_env
        _add_detection(db, video_dir, age_hours=48, analyzed=False)

        cutoff = datetime.now() - timedelta(hours=24)
        results = db.get_empty_detections(before=cutoff)
        assert len(results) == 0

    def test_excludes_detections_newer_than_cutoff(self, temp_env):
        db, video_dir, _ = temp_env
        _add_detection(db, video_dir, age_hours=1, analyzed=True)

        cutoff = datetime.now() - timedelta(hours=24)
        results = db.get_empty_detections(before=cutoff)
        assert len(results) == 0

    def test_excludes_detections_with_frame_objects(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, _ = _add_detection(db, video_dir, age_hours=48, analyzed=True)

        db.add_frame_objects(det_id, [{
            "frame_number": 0,
            "frame_timestamp": 0.0,
            "label": "bird",
            "score": 0.9,
            "x1": 0, "y1": 0, "x2": 100, "y2": 100,
            "img_width": 640, "img_height": 480,
            "animal_class": "bird",
            "animal_confidence": 0.9,
        }])

        cutoff = datetime.now() - timedelta(hours=24)
        results = db.get_empty_detections(before=cutoff)
        assert len(results) == 0

    def test_includes_unknown_animal_class(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, _ = _add_detection(
            db, video_dir, age_hours=48, animal_class="unknown", analyzed=True
        )

        cutoff = datetime.now() - timedelta(hours=24)
        results = db.get_empty_detections(before=cutoff)
        assert len(results) == 1
        assert results[0].id == det_id


class TestCleanupDetection:
    def test_deletes_all_artifacts(self, temp_env):
        db, video_dir, tmpdir = temp_env
        det_id, video_path = _add_detection(db, video_dir, age_hours=48, analyzed=True)

        detection = db.get_detection(det_id)
        cleanup_detection(db, detection, video_dir)

        assert not video_path.exists()
        assert not (video_dir / "thumbnails" / (video_path.stem + ".jpg")).exists()
        assert not (tmpdir / "annotated" / str(det_id)).exists()
        assert db.get_detection(det_id) is None

    def test_deletes_companion_wav(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, video_path = _add_detection(db, video_dir, age_hours=48, analyzed=True, with_wav=True)
        wav_path = video_path.with_suffix(".wav")
        assert wav_path.exists()

        detection = db.get_detection(det_id)
        cleanup_detection(db, detection, video_dir)

        assert not wav_path.exists()
        assert db.get_detection(det_id) is None

    def test_handles_missing_wav(self, temp_env):
        """Should not raise when no WAV file exists alongside the video."""
        db, video_dir, _ = temp_env
        det_id, video_path = _add_detection(db, video_dir, age_hours=48, analyzed=True)
        assert not video_path.with_suffix(".wav").exists()

        detection = db.get_detection(det_id)
        cleanup_detection(db, detection, video_dir)

        assert db.get_detection(det_id) is None

    def test_handles_missing_video(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, video_path = _add_detection(db, video_dir, age_hours=48, analyzed=True)
        video_path.unlink()

        detection = db.get_detection(det_id)
        cleanup_detection(db, detection, video_dir)

        assert db.get_detection(det_id) is None


class TestRunCleanup:
    def test_removes_empty_old_detections(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, video_path = _add_detection(db, video_dir, age_hours=48, analyzed=True)

        count = run_cleanup(db, video_dir, max_age_hours=24)
        assert count == 1
        assert db.get_detection(det_id) is None
        assert not video_path.exists()

    def test_skips_recent_detections(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, video_path = _add_detection(db, video_dir, age_hours=1, analyzed=True)

        count = run_cleanup(db, video_dir, max_age_hours=24)
        assert count == 0
        assert db.get_detection(det_id) is not None
        assert video_path.exists()

    def test_skips_detections_with_animals(self, temp_env):
        db, video_dir, _ = temp_env
        det_id, _ = _add_detection(
            db, video_dir, age_hours=48, animal_class="cat", analyzed=True
        )

        count = run_cleanup(db, video_dir, max_age_hours=24)
        assert count == 0
        assert db.get_detection(det_id) is not None

    def test_mixed_detections(self, temp_env):
        db, video_dir, _ = temp_env
        empty_id, empty_video = _add_detection(
            db, video_dir, age_hours=48, analyzed=True
        )
        animal_id, animal_video = _add_detection(
            db, video_dir, age_hours=48, animal_class="bird", analyzed=True
        )
        recent_id, recent_video = _add_detection(
            db, video_dir, age_hours=1, analyzed=True
        )

        count = run_cleanup(db, video_dir, max_age_hours=24)
        assert count == 1

        assert db.get_detection(empty_id) is None
        assert not empty_video.exists()

        assert db.get_detection(animal_id) is not None
        assert animal_video.exists()

        assert db.get_detection(recent_id) is not None
        assert recent_video.exists()

    def test_returns_zero_when_nothing_to_clean(self, temp_env):
        db, video_dir, _ = temp_env
        count = run_cleanup(db, video_dir, max_age_hours=24)
        assert count == 0
