"""Tests for timelapse generation and API."""

import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.storage.database import Database, Detection, Timelapse


@pytest.fixture
def db():
    """Create a temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        database = Database(Path(f.name))
    yield database
    database.close()


@pytest.fixture
def video_dir(tmp_path):
    """Create a temporary video directory."""
    vdir = tmp_path / "videos"
    vdir.mkdir()
    return vdir


# --- Database CRUD Tests ---

class TestTimelapseCRUD:

    def test_add_and_get_timelapse(self, db):
        tl = Timelapse(
            date=date(2026, 2, 14),
            video_path="/data/timelapses/timelapse_20260214.mp4",
            detection_count=5,
            animal_counts={"bird": 3, "cat": 2},
        )
        tl_id = db.add_timelapse(tl)
        assert tl_id is not None

        retrieved = db.get_timelapse(tl_id)
        assert retrieved is not None
        assert retrieved.date == date(2026, 2, 14)
        assert retrieved.detection_count == 5
        assert retrieved.animal_counts == {"bird": 3, "cat": 2}

    def test_get_timelapse_not_found(self, db):
        assert db.get_timelapse(999) is None

    def test_get_timelapse_by_date(self, db):
        tl = Timelapse(
            date=date(2026, 1, 1),
            video_path="/path/tl.mp4",
            detection_count=3,
        )
        db.add_timelapse(tl)

        found = db.get_timelapse_by_date(date(2026, 1, 1))
        assert found is not None
        assert found.detection_count == 3

        assert db.get_timelapse_by_date(date(2026, 1, 2)) is None

    def test_get_timelapses_ordered(self, db):
        for d in [date(2026, 1, 1), date(2026, 1, 3), date(2026, 1, 2)]:
            db.add_timelapse(Timelapse(date=d, video_path=f"/tl_{d}.mp4"))

        results = db.get_timelapses()
        dates = [t.date for t in results]
        assert dates == [date(2026, 1, 3), date(2026, 1, 2), date(2026, 1, 1)]

    def test_get_timelapses_pagination(self, db):
        for i in range(5):
            db.add_timelapse(Timelapse(
                date=date(2026, 1, i + 1),
                video_path=f"/tl_{i}.mp4",
            ))

        page1 = db.get_timelapses(limit=2, offset=0)
        page2 = db.get_timelapses(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2

    def test_delete_timelapse(self, db):
        tl_id = db.add_timelapse(Timelapse(
            date=date(2026, 2, 1),
            video_path="/tl.mp4",
        ))
        assert db.delete_timelapse(tl_id) is True
        assert db.get_timelapse(tl_id) is None

    def test_delete_timelapse_not_found(self, db):
        assert db.delete_timelapse(999) is False


# --- Timelapse Generation Tests ---

class TestGenerateTimelapse:

    def _add_detection(self, db, video_dir, timestamp, animal_class=None):
        """Helper to add a detection with a dummy video file."""
        video_path = video_dir / f"motion_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path.write_bytes(b"fake video content")

        det = Detection(
            timestamp=timestamp,
            video_path=str(video_path),
            trigger_type="motion",
            animal_class=animal_class,
            analyzed=True,
        )
        return db.add_detection(det)

    @patch("src.storage.timelapse.cv2")
    @patch("src.storage.timelapse.CV2_AVAILABLE", True)
    def test_generates_timelapse(self, mock_cv2, db, video_dir):
        """Timelapse is created when detections exist."""
        from src.storage.timelapse import generate_timelapse

        target = date(2026, 2, 14)
        self._add_detection(db, video_dir, datetime(2026, 2, 14, 10, 0), "bird")
        self._add_detection(db, video_dir, datetime(2026, 2, 14, 14, 0), "cat")

        mock_frame = MagicMock()
        mock_frame.image = MagicMock()
        mock_frame.image.convert.return_value.resize.return_value = MagicMock()

        mock_extractor = MagicMock()
        mock_extractor.__enter__ = MagicMock(return_value=mock_extractor)
        mock_extractor.__exit__ = MagicMock(return_value=False)
        mock_extractor.extract_middle_frame.return_value = mock_frame

        mock_writer = MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0

        timelapse_dir = video_dir.parent / "timelapses"

        with patch("src.analysis.frame_extractor.FrameExtractor", return_value=mock_extractor):
            with patch("src.storage.thumbnail.generate_thumbnail"):
                # Create the output file so the existence check passes
                timelapse_dir.mkdir(parents=True, exist_ok=True)
                expected_path = timelapse_dir / "timelapse_20260214.mp4"
                expected_path.write_bytes(b"fake mp4")

                tl_id = generate_timelapse(db, video_dir, target)

        assert tl_id is not None
        tl = db.get_timelapse(tl_id)
        assert tl.date == target
        assert tl.detection_count == 2
        assert tl.animal_counts == {"bird": 1, "cat": 1}
        assert mock_writer.write.call_count == 2
        mock_writer.release.assert_called_once()

    def test_skips_when_no_detections(self, db, video_dir):
        from src.storage.timelapse import generate_timelapse

        result = generate_timelapse(db, video_dir, date(2026, 2, 14))
        assert result is None

    @patch("src.storage.timelapse.CV2_AVAILABLE", True)
    def test_skips_when_timelapse_exists(self, db, video_dir):
        from src.storage.timelapse import generate_timelapse

        target = date(2026, 2, 14)
        db.add_timelapse(Timelapse(date=target, video_path="/existing.mp4"))
        self._add_detection(db, video_dir, datetime(2026, 2, 14, 10, 0))

        result = generate_timelapse(db, video_dir, target)
        assert result is None

    def test_skips_when_cv2_unavailable(self, db, video_dir):
        from src.storage.timelapse import generate_timelapse

        with patch("src.storage.timelapse.CV2_AVAILABLE", False):
            result = generate_timelapse(db, video_dir, date(2026, 2, 14))
        assert result is None


# --- run_timelapse_job Tests ---

class TestRunTimelapseJob:

    @patch("src.storage.timelapse.generate_timelapse")
    def test_defaults_to_today(self, mock_gen, db, video_dir):
        from src.storage.timelapse import run_timelapse_job

        run_timelapse_job(db, video_dir)
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args
        assert call_args[0][2] == date.today()

    @patch("src.storage.timelapse.generate_timelapse")
    def test_uses_specified_date(self, mock_gen, db, video_dir):
        from src.storage.timelapse import run_timelapse_job

        target = date(2026, 3, 1)
        run_timelapse_job(db, video_dir, target_date=target)
        assert mock_gen.call_args[0][2] == target

    @patch("src.storage.timelapse.generate_timelapse", side_effect=Exception("boom"))
    def test_handles_exception(self, mock_gen, db, video_dir):
        from src.storage.timelapse import run_timelapse_job

        # Should not raise
        run_timelapse_job(db, video_dir)


# --- API Tests ---

class TestTimelapseAPI:

    @pytest.fixture
    def app(self, db, video_dir):
        from src.web.app import create_app
        app = create_app(video_dir=video_dir, db_path=db.db_path)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def test_list_timelapses_empty(self, client):
        resp = client.get("/api/timelapses")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["timelapses"] == []
        assert data["count"] == 0

    def test_list_timelapses(self, client, db):
        db.add_timelapse(Timelapse(
            date=date(2026, 2, 14),
            video_path="/tl.mp4",
            detection_count=5,
            animal_counts={"bird": 3},
        ))
        resp = client.get("/api/timelapses")
        data = resp.get_json()
        assert data["count"] == 1
        assert data["timelapses"][0]["date"] == "2026-02-14"
        assert data["timelapses"][0]["animal_counts"] == {"bird": 3}

    def test_get_timelapse(self, client, db):
        tl_id = db.add_timelapse(Timelapse(
            date=date(2026, 2, 14),
            video_path="/tl.mp4",
            detection_count=3,
            animal_counts={"cat": 1},
        ))
        resp = client.get(f"/api/timelapses/{tl_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["detection_count"] == 3

    def test_get_timelapse_not_found(self, client):
        resp = client.get("/api/timelapses/999")
        assert resp.status_code == 404

    def test_delete_timelapse(self, client, db, video_dir):
        tl_dir = video_dir.parent / "timelapses"
        tl_dir.mkdir(parents=True, exist_ok=True)
        tl_path = tl_dir / "timelapse_20260214.mp4"
        tl_path.write_bytes(b"fake video")

        tl_id = db.add_timelapse(Timelapse(
            date=date(2026, 2, 14),
            video_path=str(tl_path),
            detection_count=5,
        ))

        resp = client.delete(f"/api/timelapses/{tl_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["video_deleted"] is True
        assert not tl_path.exists()
        assert db.get_timelapse(tl_id) is None

    def test_delete_timelapse_not_found(self, client):
        resp = client.delete("/api/timelapses/999")
        assert resp.status_code == 404
