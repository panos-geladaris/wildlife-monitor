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
