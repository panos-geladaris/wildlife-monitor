"""Tests for the web module (Flask app and API endpoints)."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.web.app import create_app
from src.storage.database import Database, Detection


@pytest.fixture
def app():
    """Create a Flask app with temporary database and video directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_dir = Path(tmpdir) / "videos"
        video_dir.mkdir()
        db_path = Path(tmpdir) / "test.db"
        
        app = create_app(
            video_dir=video_dir,
            db_path=db_path,
        )
        app.config["TESTING"] = True
        yield app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


@pytest.fixture
def db(app):
    """Get database instance for test data setup."""
    return Database(app.config["DB_PATH"])


@pytest.fixture
def sample_detections(db):
    """Create sample detections in the database."""
    detections = []
    for i in range(3):
        detection = Detection(
            timestamp=datetime.now(),
            video_path=f"/data/videos/test_{i}.mp4",
            trigger_type="motion" if i < 2 else "scheduled",
            animal_class="bird" if i == 0 else None,
            confidence=0.85 if i == 0 else None,
            analyzed=i == 0,
        )
        detection_id = db.add_detection(detection)
        detection.id = detection_id
        detections.append(detection)
    return detections


class TestCreateApp:
    """Tests for create_app function."""
    
    def test_create_app_returns_flask_app(self, app):
        """Test that create_app returns a Flask application."""
        from flask import Flask
        assert isinstance(app, Flask)
    
    def test_app_has_correct_config(self, app):
        """Test that app has expected configuration."""
        assert app.config["TESTING"] is True
        assert "VIDEO_DIR" in app.config
        assert "DB_PATH" in app.config
        assert isinstance(app.config["VIDEO_DIR"], Path)
        assert isinstance(app.config["DB_PATH"], Path)
    
    def test_blueprints_are_registered(self, app):
        """Test that API blueprint is registered."""
        blueprint_names = list(app.blueprints.keys())
        assert "api" in blueprint_names


class TestPageRoutes:
    """Tests for page routes."""
    
    def test_index_returns_200(self, client):
        """Test GET / returns 200."""
        with patch("src.web.app.render_template") as mock_render:
            mock_render.return_value = "<html>Dashboard</html>"
            response = client.get("/")
            assert response.status_code == 200
            mock_render.assert_called_once_with("index.html")
    
    def test_gallery_returns_200(self, client):
        """Test GET /gallery returns 200."""
        with patch("src.web.app.render_template") as mock_render:
            mock_render.return_value = "<html>Gallery</html>"
            response = client.get("/gallery")
            assert response.status_code == 200
            mock_render.assert_called_once_with("gallery.html")
    
    def test_detection_detail_returns_200(self, client):
        """Test GET /detection/1 returns 200."""
        with patch("src.web.app.render_template") as mock_render:
            mock_render.return_value = "<html>Detection</html>"
            response = client.get("/detection/1")
            assert response.status_code == 200
            mock_render.assert_called_once_with("detection.html", detection_id=1)
    
    def test_statistics_returns_200(self, client):
        """Test GET /statistics returns 200."""
        with patch("src.web.app.render_template") as mock_render:
            mock_render.return_value = "<html>Statistics</html>"
            response = client.get("/statistics")
            assert response.status_code == 200
            mock_render.assert_called_once_with("statistics.html")


class TestServeVideo:
    """Tests for /videos/<filename> route."""

    def test_serve_video_returns_file(self, app, client):
        """Test GET /videos/<filename> serves an existing video file."""
        video_dir = app.config["VIDEO_DIR"]
        (video_dir / "test_clip.mp4").write_bytes(b"fake video data")

        response = client.get("/videos/test_clip.mp4")
        assert response.status_code == 200
        assert response.content_type == "video/mp4"
        assert response.data == b"fake video data"

    def test_serve_video_not_found(self, client):
        """Test GET /videos/<filename> returns 404 for missing file."""
        response = client.get("/videos/nonexistent.mp4")
        assert response.status_code == 404


class TestApiStatus:
    """Tests for /api/status endpoint."""
    
    def test_status_returns_json(self, client):
        """Test GET /api/status returns JSON with expected fields."""
        response = client.get("/api/status")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        
        data = response.get_json()
        assert "status" in data
        assert "timestamp" in data
        assert "storage" in data
        assert "detections" in data
        
        assert data["status"] == "running"
        assert "videos_mb" in data["storage"]
        assert "video_count" in data["storage"]
        assert "disk_free_gb" in data["storage"]
        assert "total" in data["detections"]
        assert "last_hour" in data["detections"]


class TestApiDetections:
    """Tests for /api/detections endpoints."""
    
    def test_list_detections_returns_json_array(self, client):
        """Test GET /api/detections returns JSON array."""
        response = client.get("/api/detections")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        
        data = response.get_json()
        assert "detections" in data
        assert isinstance(data["detections"], list)
        assert "count" in data
        assert "limit" in data
        assert "offset" in data
    
    def test_list_detections_with_data(self, client, sample_detections):
        """Test GET /api/detections returns detection data."""
        response = client.get("/api/detections")
        data = response.get_json()
        
        assert data["count"] == 3
        assert len(data["detections"]) == 3
        
        detection = data["detections"][0]
        assert "id" in detection
        assert "timestamp" in detection
        assert "video_path" in detection
        assert "trigger_type" in detection
        assert "animal_class" in detection
        assert "confidence" in detection
        assert "analyzed" in detection
    
    def test_list_detections_with_trigger_type_filter(self, client, sample_detections):
        """Test GET /api/detections with trigger_type filter."""
        response = client.get("/api/detections?trigger_type=motion")
        data = response.get_json()
        
        assert data["count"] == 2
        for detection in data["detections"]:
            assert detection["trigger_type"] == "motion"
    
    def test_list_detections_with_analyzed_filter(self, client, sample_detections):
        """Test GET /api/detections with analyzed filter."""
        response = client.get("/api/detections?analyzed=true")
        data = response.get_json()
        
        assert data["count"] == 1
        assert data["detections"][0]["analyzed"] is True
    
    def test_list_detections_with_limit(self, client, sample_detections):
        """Test GET /api/detections with limit parameter."""
        response = client.get("/api/detections?limit=2")
        data = response.get_json()
        
        assert len(data["detections"]) == 2
        assert data["limit"] == 2
    
    def test_list_detections_with_offset(self, client, sample_detections):
        """Test GET /api/detections with offset parameter."""
        response = client.get("/api/detections?offset=1")
        data = response.get_json()
        
        assert data["offset"] == 1
        assert len(data["detections"]) == 2
    
    def test_get_detection_not_found(self, client):
        """Test GET /api/detections/:id returns 404 for non-existent."""
        response = client.get("/api/detections/9999")
        assert response.status_code == 404
        
        data = response.get_json()
        assert "error" in data
        assert "not found" in data["error"].lower()
    
    def test_get_detection_by_id(self, client, sample_detections):
        """Test GET /api/detections/:id returns detection data."""
        detection_id = sample_detections[0].id
        response = client.get(f"/api/detections/{detection_id}")
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data["id"] == detection_id
        assert "timestamp" in data
        assert "video_path" in data
        assert "trigger_type" in data
        assert "created_at" in data
    
    def test_get_detection_includes_video_filename(self, client, sample_detections):
        """Test GET /api/detections/:id returns video_filename extracted from video_path."""
        detection_id = sample_detections[0].id
        response = client.get(f"/api/detections/{detection_id}")
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "video_filename" in data
        assert data["video_filename"] == Path(data["video_path"]).name
        assert "/" not in data["video_filename"]


class TestApiStats:
    """Tests for /api/stats endpoints."""
    
    def test_daily_stats_returns_data(self, client):
        """Test GET /api/stats/daily returns data."""
        response = client.get("/api/stats/daily")
        assert response.status_code == 200
        
        data = response.get_json()
        assert "stats" in data
        assert "days" in data
        assert isinstance(data["stats"], list)
        assert data["days"] == 7
    
    def test_daily_stats_with_days_param(self, client):
        """Test GET /api/stats/daily with days parameter."""
        response = client.get("/api/stats/daily?days=3")
        data = response.get_json()
        
        assert data["days"] == 3
        assert len(data["stats"]) == 3
        
        stat = data["stats"][0]
        assert "date" in stat
        assert "total" in stat
        assert "motion" in stat
        assert "scheduled" in stat
    
    def test_animal_stats_returns_data(self, client):
        """Test GET /api/stats/animals returns data."""
        response = client.get("/api/stats/animals")
        assert response.status_code == 200
        
        data = response.get_json()
        assert "animals" in data
        assert "total_analyzed" in data
        assert "days" in data
        assert isinstance(data["animals"], list)
    
    def test_animal_stats_with_data(self, client, sample_detections):
        """Test GET /api/stats/animals with detection data."""
        response = client.get("/api/stats/animals")
        data = response.get_json()
        
        assert data["total_analyzed"] == 1
        assert len(data["animals"]) == 1
        assert data["animals"][0]["animal_class"] == "bird"
        assert data["animals"][0]["count"] == 1
    
    def test_summary_returns_data(self, client):
        """Test GET /api/stats/summary returns data."""
        response = client.get("/api/stats/summary")
        assert response.status_code == 200
        
        data = response.get_json()
        assert "today" in data
        assert "recent_detections" in data
        
        assert "total" in data["today"]
        assert "motion" in data["today"]
        assert "animals" in data["today"]
        assert isinstance(data["recent_detections"], list)
    
    def test_summary_with_data(self, client, sample_detections):
        """Test GET /api/stats/summary includes recent detections."""
        response = client.get("/api/stats/summary")
        data = response.get_json()
        
        assert data["today"]["total"] == 3
        assert len(data["recent_detections"]) == 3
        
        recent = data["recent_detections"][0]
        assert "id" in recent
        assert "timestamp" in recent
        assert "trigger_type" in recent
        assert "animal_class" in recent
        assert "confidence" in recent


class TestHighlightsApi:
    """Tests for highlights API endpoints."""

    def test_highlights_page_returns_200(self, client):
        """Test GET /highlights returns 200."""
        response = client.get("/highlights")
        assert response.status_code == 200

    def test_toggle_highlight_on(self, client, sample_detections):
        """Test POST /api/detections/<id>/highlight toggles on."""
        det_id = sample_detections[0].id
        response = client.post(f"/api/detections/{det_id}/highlight")
        data = response.get_json()

        assert response.status_code == 200
        assert data["highlighted"] is True

    def test_toggle_highlight_off(self, client, sample_detections):
        """Test toggling highlight off after it was on."""
        det_id = sample_detections[0].id
        client.post(f"/api/detections/{det_id}/highlight")
        response = client.post(f"/api/detections/{det_id}/highlight")
        data = response.get_json()

        assert response.status_code == 200
        assert data["highlighted"] is False

    def test_toggle_highlight_not_found(self, client):
        """Test toggling highlight on non-existent detection."""
        response = client.post("/api/detections/9999/highlight")
        assert response.status_code == 404

    def test_list_highlights_empty(self, client):
        """Test GET /api/highlights with no highlighted detections."""
        response = client.get("/api/highlights")
        data = response.get_json()

        assert response.status_code == 200
        assert data["detections"] == []

    def test_list_highlights_with_data(self, client, sample_detections):
        """Test GET /api/highlights returns only highlighted detections."""
        client.post(f"/api/detections/{sample_detections[0].id}/highlight")
        client.post(f"/api/detections/{sample_detections[2].id}/highlight")

        response = client.get("/api/highlights")
        data = response.get_json()

        assert response.status_code == 200
        assert data["count"] == 2
        ids = {d["id"] for d in data["detections"]}
        assert sample_detections[0].id in ids
        assert sample_detections[2].id in ids

    def test_detection_includes_highlighted_field(self, client, sample_detections):
        """Test that detection API responses include the highlighted field."""
        det_id = sample_detections[0].id
        response = client.get(f"/api/detections/{det_id}")
        data = response.get_json()

        assert "highlighted" in data
        assert data["highlighted"] is False

    def test_detections_list_includes_highlighted(self, client, sample_detections):
        """Test that detections list includes highlighted field."""
        response = client.get("/api/detections")
        data = response.get_json()

        for d in data["detections"]:
            assert "highlighted" in d


class TestSoundClassificationApi:
    """Tests for sound classification fields in API responses."""

    def test_detection_includes_sound_fields(self, client, db):
        """Detection detail should include sound classification fields."""
        det_id = db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/test.mp4",
            trigger_type="motion",
        ))
        db.update_detection(
            det_id,
            sound_class="bird",
            sound_species="Turdus merula_Eurasian Blackbird",
            sound_confidence=0.91,
        )

        response = client.get(f"/api/detections/{det_id}")
        data = response.get_json()

        assert data["sound_class"] == "bird"
        assert data["sound_species"] == "Turdus merula_Eurasian Blackbird"
        assert data["sound_confidence"] == 0.91

    def test_detection_sound_fields_null_by_default(self, client, sample_detections):
        """Sound fields should be null when no audio analysis was done."""
        det_id = sample_detections[0].id
        response = client.get(f"/api/detections/{det_id}")
        data = response.get_json()

        assert data["sound_class"] is None
        assert data["sound_species"] is None
        assert data["sound_confidence"] is None

    def test_detections_list_includes_sound_fields(self, client, sample_detections):
        """Detection list should include sound classification fields."""
        response = client.get("/api/detections")
        data = response.get_json()

        for d in data["detections"]:
            assert "sound_class" in d
            assert "sound_species" in d
            assert "sound_confidence" in d

    def test_highlights_include_sound_fields(self, client, sample_detections):
        """Highlighted detections should include sound fields."""
        det_id = sample_detections[0].id
        client.post(f"/api/detections/{det_id}/highlight")

        response = client.get("/api/highlights")
        data = response.get_json()

        for d in data["detections"]:
            assert "sound_class" in d
            assert "sound_species" in d
            assert "sound_confidence" in d

    def test_summary_recent_includes_sound_fields(self, client, sample_detections):
        """Summary recent detections should include sound fields."""
        response = client.get("/api/stats/summary")
        data = response.get_json()

        for d in data["recent_detections"]:
            assert "sound_class" in d
            assert "sound_species" in d
            assert "sound_confidence" in d


class TestReclassifyApi:
    """Tests for POST /api/detections/<id>/reclassify."""

    @pytest.fixture(autouse=True)
    def mock_analysis_module(self):
        """
        Inject mock ML classes directly into src.analysis.__dict__ to avoid
        triggering the lazy __getattr__ imports that need torch/cv2.
        """
        import src.analysis as mod

        # Build a mock AnimalClassifier whose instance returns a bird result
        mock_result = MagicMock()
        mock_result.animal_class = "bird"
        mock_result.confidence = 0.85
        mock_result.bird_species = None

        mock_instance = MagicMock()
        mock_instance.classify_video.return_value = mock_result

        mock_cls = MagicMock(return_value=mock_instance)
        self._mock_classifier_cls = mock_cls
        self._mock_classifier_instance = mock_instance

        # ObjectDetector raises ImportError so the reclassify code skips it cleanly
        def _raise_import(*a, **kw):
            raise ImportError("mocked")

        mock_detector_cls = MagicMock(side_effect=_raise_import)

        mod.__dict__["AnimalClassifier"] = mock_cls
        mod.__dict__["ObjectDetector"] = mock_detector_cls

        yield

        mod.__dict__.pop("AnimalClassifier", None)
        mod.__dict__.pop("ObjectDetector", None)

    @pytest.fixture
    def detection_with_video(self, app, db):
        """Create a detection whose video file actually exists on disk."""
        video_dir = app.config["VIDEO_DIR"]
        video_path = video_dir / "motion_20260404_120000.mp4"
        video_path.write_bytes(b"fake video data")

        det_id = db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path=str(video_path),
            trigger_type="motion",
            analyzed=False,
        ))
        return det_id, video_path

    def test_reclassify_not_found(self, client):
        response = client.post("/api/detections/9999/reclassify")
        assert response.status_code == 404

    def test_reclassify_runs_audio_when_wav_present(self, client, app, db, detection_with_video):
        """When a WAV file exists alongside the video, audio classification runs."""
        det_id, video_path = detection_with_video
        wav_path = video_path.with_suffix(".wav")
        wav_path.write_bytes(b"fake wav data")

        mock_audio_result = MagicMock()
        mock_audio_result.has_sound = True
        mock_audio_result.sound_class = "bird"
        mock_audio_result.sound_species = "Turdus merula_Eurasian Blackbird"
        mock_audio_result.sound_confidence = 0.91

        with patch("src.analysis.audio_classifier.AudioClassifier.classify_audio",
                   return_value=mock_audio_result):
            response = client.post(f"/api/detections/{det_id}/reclassify")

        assert response.status_code == 200
        data = response.get_json()
        assert data["sound_class"] == "bird"
        assert data["sound_species"] == "Turdus merula_Eurasian Blackbird"
        assert data["sound_confidence"] == 0.91

    def test_reclassify_skips_audio_when_no_wav(self, client, app, db, detection_with_video):
        """When no WAV file exists, audio classification is skipped gracefully."""
        det_id, video_path = detection_with_video
        assert not video_path.with_suffix(".wav").exists()

        response = client.post(f"/api/detections/{det_id}/reclassify")

        assert response.status_code == 200
        data = response.get_json()
        assert data["sound_class"] is None
        assert data["sound_species"] is None
        assert data["sound_confidence"] is None

    def test_reclassify_audio_no_sound_detected(self, client, app, db, detection_with_video):
        """When audio classification finds nothing, sound fields remain null."""
        det_id, video_path = detection_with_video
        wav_path = video_path.with_suffix(".wav")
        wav_path.write_bytes(b"fake wav data")

        mock_audio_result = MagicMock()
        mock_audio_result.has_sound = False

        with patch("src.analysis.audio_classifier.AudioClassifier.classify_audio",
                   return_value=mock_audio_result):
            response = client.post(f"/api/detections/{det_id}/reclassify")

        assert response.status_code == 200
        data = response.get_json()
        assert data["sound_class"] is None

    def test_reclassify_response_includes_sound_fields(self, client, app, db, detection_with_video):
        """Response always includes sound fields regardless of outcome."""
        det_id, _ = detection_with_video

        response = client.post(f"/api/detections/{det_id}/reclassify")

        assert response.status_code == 200
        data = response.get_json()
        assert "sound_class" in data
        assert "sound_species" in data
        assert "sound_confidence" in data
        assert data["message"] == "Re-classification complete"
