"""Tests for the video store module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.storage.video_store import VideoStore, VideoInfo


@pytest.fixture
def video_store():
    """Create a temporary video store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VideoStore(video_dir=Path(tmpdir), retention_days=30)
        yield store


@pytest.fixture
def video_store_with_files(video_store):
    """Create a video store with some test video files."""
    # Create test video files
    files = [
        ("motion_20240115_120000.mp4", "motion"),
        ("motion_20240115_130000.mp4", "motion"),
        ("scheduled_20240115_140000.mp4", "scheduled"),
        ("manual_20240115_150000.mp4", "manual"),
    ]
    
    for filename, _ in files:
        path = video_store.video_dir / filename
        path.write_text(f"fake video content for {filename}")
    
    return video_store


class TestVideoStore:
    """Tests for VideoStore class."""
    
    def test_init_creates_directory(self):
        """Test that initialization creates the video directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_dir = Path(tmpdir) / "videos" / "subdir"
            store = VideoStore(video_dir=video_dir)
            
            assert video_dir.exists()
    
    def test_get_video_path(self, video_store):
        """Test getting full path for a filename."""
        path = video_store.get_video_path("test.mp4")
        
        assert path == video_store.video_dir / "test.mp4"
    
    def test_video_exists_true(self, video_store):
        """Test checking if existing video exists."""
        # Create a test file
        test_file = video_store.video_dir / "test.mp4"
        test_file.write_text("test content")
        
        assert video_store.video_exists("test.mp4") is True
    
    def test_video_exists_false(self, video_store):
        """Test checking if non-existent video exists."""
        assert video_store.video_exists("nonexistent.mp4") is False
    
    def test_get_video_info(self, video_store):
        """Test getting video info."""
        # Create a test file
        filename = "motion_20240115_120000.mp4"
        test_file = video_store.video_dir / filename
        test_file.write_text("test video content")
        
        info = video_store.get_video_info(filename)
        
        assert info is not None
        assert info.filename == filename
        assert info.trigger_type == "motion"
        assert info.size_bytes > 0
    
    def test_get_video_info_not_found(self, video_store):
        """Test getting info for non-existent video."""
        info = video_store.get_video_info("nonexistent.mp4")
        
        assert info is None
    
    def test_list_videos_empty(self, video_store):
        """Test listing videos in empty directory."""
        videos = video_store.list_videos()
        
        assert videos == []
    
    def test_list_videos(self, video_store_with_files):
        """Test listing all videos."""
        videos = video_store_with_files.list_videos()
        
        assert len(videos) == 4
    
    def test_list_videos_filter_by_trigger_type(self, video_store_with_files):
        """Test filtering videos by trigger type."""
        motion_videos = video_store_with_files.list_videos(trigger_type="motion")
        scheduled_videos = video_store_with_files.list_videos(trigger_type="scheduled")
        
        assert len(motion_videos) == 2
        assert len(scheduled_videos) == 1
        assert all(v.trigger_type == "motion" for v in motion_videos)
    
    def test_list_videos_with_limit(self, video_store_with_files):
        """Test limiting video results."""
        videos = video_store_with_files.list_videos(limit=2)
        
        assert len(videos) == 2
    
    def test_delete_video(self, video_store):
        """Test deleting a video."""
        # Create a test file
        filename = "test.mp4"
        test_file = video_store.video_dir / filename
        test_file.write_text("test content")
        
        success = video_store.delete_video(filename)
        
        assert success is True
        assert not test_file.exists()
    
    def test_delete_video_not_found(self, video_store):
        """Test deleting non-existent video."""
        success = video_store.delete_video("nonexistent.mp4")
        
        assert success is False
    
    def test_cleanup_old_videos(self, video_store):
        """Test cleaning up old videos."""
        import os
        
        # Create an "old" file by setting its modification time
        old_filename = "motion_20230101_120000.mp4"
        old_file = video_store.video_dir / old_filename
        old_file.write_text("old video")
        
        # Set modification time to 60 days ago
        old_time = datetime.now() - timedelta(days=60)
        os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
        
        # Create a "new" file with current timestamp in filename
        now = datetime.now()
        new_filename = f"motion_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        new_file = video_store.video_dir / new_filename
        new_file.write_text("new video")
        
        # Cleanup with 30-day retention
        deleted_count = video_store.cleanup_old_videos(retention_days=30)
        
        assert deleted_count == 1
        assert not old_file.exists()
        assert new_file.exists()
    
    def test_get_storage_usage(self, video_store_with_files):
        """Test getting storage usage statistics."""
        usage = video_store_with_files.get_storage_usage()
        
        assert "total_bytes" in usage
        assert "video_count" in usage
        assert usage["video_count"] == 4
        assert usage["total_bytes"] > 0
        assert "by_trigger_type" in usage
        assert "motion" in usage["by_trigger_type"]
    
    def test_get_disk_free_space(self, video_store):
        """Test getting disk free space."""
        disk = video_store.get_disk_free_space()
        
        assert "total_bytes" in disk
        assert "free_bytes" in disk
        assert "free_gb" in disk
        assert "used_percent" in disk
        assert disk["free_bytes"] > 0


class TestVideoInfoParsing:
    """Tests for filename parsing functionality."""
    
    def test_parse_trigger_type_motion(self, video_store):
        """Test parsing motion trigger type."""
        trigger = video_store._parse_trigger_type("motion_20240115_120000.mp4")
        assert trigger == "motion"
    
    def test_parse_trigger_type_scheduled(self, video_store):
        """Test parsing scheduled trigger type."""
        trigger = video_store._parse_trigger_type("scheduled_20240115_120000.mp4")
        assert trigger == "scheduled"
    
    def test_parse_trigger_type_manual(self, video_store):
        """Test parsing manual trigger type."""
        trigger = video_store._parse_trigger_type("manual_20240115_120000.mp4")
        assert trigger == "manual"
    
    def test_parse_trigger_type_unknown(self, video_store):
        """Test parsing unknown trigger type."""
        trigger = video_store._parse_trigger_type("random_video.mp4")
        assert trigger == "unknown"
    
    def test_parse_timestamp(self, video_store):
        """Test parsing timestamp from filename."""
        timestamp = video_store._parse_timestamp("motion_20240115_120000.mp4")
        
        assert timestamp is not None
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15
        assert timestamp.hour == 12
        assert timestamp.minute == 0
        assert timestamp.second == 0
    
    def test_parse_timestamp_invalid(self, video_store):
        """Test parsing invalid timestamp."""
        timestamp = video_store._parse_timestamp("invalid_filename.mp4")
        
        assert timestamp is None
