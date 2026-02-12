"""Tests for the main integration module."""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from main import WildlifeMonitor, setup_logging


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_dir = tmpdir / "videos"
        video_dir.mkdir()
        db_path = tmpdir / "test.db"
        yield video_dir, db_path


@pytest.fixture
def monitor(temp_dirs):
    """Create a WildlifeMonitor instance for testing."""
    video_dir, db_path = temp_dirs
    m = WildlifeMonitor(
        video_dir=video_dir,
        db_path=db_path,
        simulation_mode=True,
        enable_analysis=False,
    )
    yield m
    if m.is_running:
        m.stop()


class TestWildlifeMonitorInit:
    """Tests for WildlifeMonitor initialization."""
    
    def test_init_creates_directories(self, temp_dirs):
        """Test that init creates required directories."""
        video_dir, db_path = temp_dirs
        new_video_dir = video_dir.parent / "new_videos"
        
        monitor = WildlifeMonitor(
            video_dir=new_video_dir,
            db_path=db_path,
            simulation_mode=True,
        )
        
        assert new_video_dir.exists()
    
    def test_init_default_values(self):
        """Test default configuration values."""
        monitor = WildlifeMonitor(simulation_mode=True)
        
        assert monitor.video_dir == Path("data/videos")
        assert monitor.db_path == Path("data/wildlife.db")
        assert monitor.web_port == 5001
        assert monitor.enable_analysis is True
    
    def test_init_custom_values(self, temp_dirs):
        """Test custom configuration values."""
        video_dir, db_path = temp_dirs
        
        monitor = WildlifeMonitor(
            video_dir=video_dir,
            db_path=db_path,
            web_port=8080,
            simulation_mode=True,
            enable_analysis=False,
        )
        
        assert monitor.video_dir == video_dir
        assert monitor.db_path == db_path
        assert monitor.web_port == 8080
        assert monitor.enable_analysis is False


class TestSimulationModeDetection:
    """Tests for simulation mode auto-detection."""
    
    def test_explicit_simulation_mode(self, temp_dirs):
        """Test explicit simulation mode setting."""
        video_dir, db_path = temp_dirs
        
        monitor = WildlifeMonitor(
            video_dir=video_dir,
            db_path=db_path,
            simulation_mode=True,
        )
        
        assert monitor._detect_simulation_mode() is True
    
    def test_explicit_non_simulation_mode(self, temp_dirs):
        """Test explicit non-simulation mode setting."""
        video_dir, db_path = temp_dirs
        
        monitor = WildlifeMonitor(
            video_dir=video_dir,
            db_path=db_path,
            simulation_mode=False,
        )
        
        assert monitor._detect_simulation_mode() is False
    
    def test_env_var_simulation_mode(self, temp_dirs, monkeypatch):
        """Test SIMULATE environment variable."""
        video_dir, db_path = temp_dirs
        monkeypatch.setenv("SIMULATE", "1")
        
        monitor = WildlifeMonitor(
            video_dir=video_dir,
            db_path=db_path,
        )
        
        assert monitor._detect_simulation_mode() is True
    
    def test_env_var_non_simulation_mode(self, temp_dirs, monkeypatch):
        """Test SIMULATE=0 environment variable."""
        video_dir, db_path = temp_dirs
        monkeypatch.setenv("SIMULATE", "0")
        
        monitor = WildlifeMonitor(
            video_dir=video_dir,
            db_path=db_path,
        )
        
        assert monitor._detect_simulation_mode() is False


class TestDatabaseInit:
    """Tests for database initialization."""
    
    def test_database_initialized(self, monitor):
        """Test that database is created on start."""
        monitor._init_database()
        
        assert monitor._database is not None
        assert monitor.db_path.exists()
    
    def test_database_can_add_detection(self, monitor):
        """Test adding a detection to the database."""
        from src.storage import Detection
        
        monitor._init_database()
        
        detection = Detection(
            timestamp=datetime.now(),
            video_path="/test/video.mp4",
            trigger_type="motion",
        )
        detection_id = monitor._database.add_detection(detection)
        
        assert detection_id is not None
        assert detection_id > 0


class TestCaptureService:
    """Tests for capture service integration."""
    
    def test_capture_service_initialized(self, monitor):
        """Test that capture service is created."""
        monitor._init_database()
        monitor._init_capture_service()
        
        assert monitor._capture_service is not None
    
    def test_capture_service_starts(self, monitor):
        """Test that capture service can start."""
        monitor.start(capture=True, web=False)
        
        assert monitor._capture_service is not None
        assert monitor._capture_service.is_running
        
        monitor.stop()
    
    def test_on_video_captured_stores_detection(self, monitor):
        """Test that captured videos are stored in database."""
        from src.capture.camera import VideoMetadata, CaptureReason
        
        monitor._init_database()
        
        metadata = VideoMetadata(
            filepath=monitor.video_dir / "test_motion.mp4",
            timestamp=datetime.now(),
            duration_seconds=2.0,
            reason=CaptureReason.MOTION,
            resolution=(1280, 720),
        )
        
        (monitor.video_dir / "test_motion.mp4").touch()
        
        monitor._process_capture(metadata)
        
        detections = monitor._database.get_detections()
        assert len(detections) == 1
        assert detections[0].trigger_type == "motion"


class TestWebServer:
    """Tests for web server integration."""
    
    def test_web_server_starts(self, monitor):
        """Test that web server starts in background."""
        monitor.start(capture=False, web=True)
        
        time.sleep(0.5)
        
        assert monitor._web_thread is not None
        assert monitor._web_thread.is_alive()
        
        monitor.stop()
    
    def test_web_only_mode(self, monitor):
        """Test running in web-only mode."""
        monitor.start(capture=False, web=True)
        
        assert monitor._capture_service is None
        assert monitor._database is not None
        assert monitor._web_thread is not None
        
        monitor.stop()


class TestFullIntegration:
    """Tests for full system integration."""
    
    def test_start_and_stop(self, monitor):
        """Test starting and stopping the full system."""
        monitor.start(capture=True, web=True)
        
        assert monitor.is_running
        assert monitor._database is not None
        assert monitor._capture_service is not None
        
        monitor.stop()
        
        assert not monitor.is_running
    
    def test_double_start_warning(self, monitor, caplog):
        """Test that double start logs a warning."""
        monitor.start(capture=True, web=False)
        monitor.start(capture=True, web=False)
        
        assert "already running" in caplog.text.lower()
        
        monitor.stop()
    
    def test_stop_when_not_running(self, monitor):
        """Test that stop is safe when not running."""
        monitor.stop()


class TestVideoAnalysis:
    """Tests for video analysis functionality."""
    
    def test_analyze_video_without_classifier(self, monitor):
        """Test analyze_video when classifier unavailable."""
        with patch.object(monitor, '_init_classifier') as mock_init:
            mock_init.return_value = None
            monitor._classifier = None
            
            result = monitor.analyze_video(Path("test.mp4"))
            
            assert "error" in result
    
    @patch('main.WildlifeMonitor._init_classifier')
    def test_analyze_video_returns_dict(self, mock_init, monitor, temp_dirs):
        """Test that analyze_video returns a dictionary."""
        video_dir, _ = temp_dirs
        video_path = video_dir / "test.mp4"
        video_path.touch()
        
        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "animal_class": "bird",
            "confidence": 0.85,
            "is_animal": True,
        }
        
        mock_classifier = Mock()
        mock_classifier.classify_video.return_value = mock_result
        monitor._classifier = mock_classifier
        
        result = monitor.analyze_video(video_path)
        
        assert result["animal_class"] == "bird"
        assert result["confidence"] == 0.85


class TestCaptureOnlyMode:
    """Tests for capture-only mode."""
    
    def test_capture_only_no_web(self, monitor):
        """Test running in capture-only mode."""
        monitor.start(capture=True, web=False)
        
        assert monitor._capture_service is not None
        assert monitor._web_thread is None
        
        monitor.stop()


class TestSetupLogging:
    """Tests for logging setup."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging(verbose=False)
    
    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        setup_logging(verbose=True)


class TestAnalysisIntegration:
    """Tests for analysis pipeline integration."""
    
    def test_on_video_captured_with_analysis(self, temp_dirs):
        """Test that analysis runs on captured videos."""
        video_dir, db_path = temp_dirs
        
        monitor = WildlifeMonitor(
            video_dir=video_dir,
            db_path=db_path,
            simulation_mode=True,
            enable_analysis=True,
        )
        
        monitor._init_database()
        
        mock_classifier = Mock()
        mock_result = Mock()
        mock_result.animal_class = "cat"
        mock_result.confidence = 0.9
        mock_result.is_animal = True
        mock_classifier.classify_video.return_value = mock_result
        monitor._classifier = mock_classifier
        
        from src.capture.camera import VideoMetadata, CaptureReason
        
        video_path = video_dir / "test_motion.mp4"
        video_path.touch()
        
        metadata = VideoMetadata(
            filepath=video_path,
            timestamp=datetime.now(),
            duration_seconds=2.0,
            reason=CaptureReason.MOTION,
            resolution=(1280, 720),
        )
        
        monitor._process_capture(metadata)
        
        detections = monitor._database.get_detections()
        assert len(detections) == 1
        assert detections[0].animal_class == "cat"
        assert detections[0].confidence == 0.9
        assert detections[0].analyzed is True
        
        monitor.stop()
