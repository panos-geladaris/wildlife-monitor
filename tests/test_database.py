"""Tests for the storage database module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, date, timedelta

from src.storage.database import Database, Detection, DailySummary


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    database = Database(db_path)
    yield database
    
    # Cleanup
    database.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_detection():
    """Create a sample detection for testing."""
    return Detection(
        timestamp=datetime.now(),
        video_path="/data/videos/motion_20240115_120000.mp4",
        trigger_type="motion",
    )


class TestDatabase:
    """Tests for Database class."""
    
    def test_init_creates_tables(self, db):
        """Test that database initialization creates required tables."""
        with db._get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row["name"] for row in tables}
        
        assert "detections" in table_names
        assert "daily_summary" in table_names
    
    def test_add_detection(self, db, sample_detection):
        """Test adding a detection record."""
        detection_id = db.add_detection(sample_detection)
        
        assert detection_id is not None
        assert detection_id > 0
    
    def test_get_detection(self, db, sample_detection):
        """Test retrieving a detection by ID."""
        detection_id = db.add_detection(sample_detection)
        
        retrieved = db.get_detection(detection_id)
        
        assert retrieved is not None
        assert retrieved.id == detection_id
        assert retrieved.video_path == sample_detection.video_path
        assert retrieved.trigger_type == sample_detection.trigger_type
    
    def test_get_detection_not_found(self, db):
        """Test retrieving a non-existent detection."""
        retrieved = db.get_detection(9999)
        
        assert retrieved is None
    
    def test_get_detections_empty(self, db):
        """Test getting detections from empty database."""
        detections = db.get_detections()
        
        assert detections == []
    
    def test_get_detections_with_data(self, db):
        """Test getting all detections."""
        # Add multiple detections
        for i in range(5):
            detection = Detection(
                timestamp=datetime.now(),
                video_path=f"/data/videos/test_{i}.mp4",
                trigger_type="motion",
            )
            db.add_detection(detection)
        
        detections = db.get_detections()
        
        assert len(detections) == 5
    
    def test_get_detections_filter_by_trigger_type(self, db):
        """Test filtering detections by trigger type."""
        # Add motion detection
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/motion_1.mp4",
            trigger_type="motion",
        ))
        
        # Add scheduled detection
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/scheduled_1.mp4",
            trigger_type="scheduled",
        ))
        
        motion_detections = db.get_detections(trigger_type="motion")
        scheduled_detections = db.get_detections(trigger_type="scheduled")
        
        assert len(motion_detections) == 1
        assert len(scheduled_detections) == 1
        assert motion_detections[0].trigger_type == "motion"
    
    def test_get_detections_filter_by_date(self, db):
        """Test filtering detections by date range."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        # Add detection from yesterday
        db.add_detection(Detection(
            timestamp=yesterday,
            video_path="/data/videos/old.mp4",
            trigger_type="motion",
        ))
        
        # Add detection from today
        db.add_detection(Detection(
            timestamp=now,
            video_path="/data/videos/new.mp4",
            trigger_type="motion",
        ))
        
        # Filter for today only
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        detections = db.get_detections(start_date=today_start)
        
        assert len(detections) == 1
        assert "new.mp4" in detections[0].video_path
    
    def test_update_detection(self, db, sample_detection):
        """Test updating a detection with analysis results."""
        detection_id = db.add_detection(sample_detection)
        
        success = db.update_detection(
            detection_id,
            animal_class="bird",
            confidence=0.92,
            analyzed=True,
        )
        
        assert success is True
        
        updated = db.get_detection(detection_id)
        assert updated.animal_class == "bird"
        assert updated.confidence == 0.92
        assert updated.analyzed is True
    
    def test_update_detection_not_found(self, db):
        """Test updating a non-existent detection."""
        success = db.update_detection(9999, animal_class="cat")
        
        assert success is False
    
    def test_delete_detection(self, db, sample_detection):
        """Test deleting a detection."""
        detection_id = db.add_detection(sample_detection)
        
        success = db.delete_detection(detection_id)
        
        assert success is True
        assert db.get_detection(detection_id) is None
    
    def test_delete_detection_not_found(self, db):
        """Test deleting a non-existent detection."""
        success = db.delete_detection(9999)
        
        assert success is False
    
    def test_delete_old_detections(self, db):
        """Test deleting old detections."""
        now = datetime.now()
        old_date = now - timedelta(days=60)
        
        # Add old detection
        db.add_detection(Detection(
            timestamp=old_date,
            video_path="/data/videos/old.mp4",
            trigger_type="motion",
        ))
        
        # Add recent detection
        db.add_detection(Detection(
            timestamp=now,
            video_path="/data/videos/recent.mp4",
            trigger_type="motion",
        ))
        
        # Delete detections older than 30 days
        cutoff = now - timedelta(days=30)
        deleted_count = db.delete_old_detections(cutoff)
        
        assert deleted_count == 1
        
        remaining = db.get_detections()
        assert len(remaining) == 1
        assert "recent.mp4" in remaining[0].video_path
    
    def test_get_unanalyzed_detections(self, db):
        """Test getting unanalyzed detections."""
        # Add unanalyzed detection
        unanalyzed_id = db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/unanalyzed.mp4",
            trigger_type="motion",
            analyzed=False,
        ))
        
        # Add analyzed detection
        analyzed_id = db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/analyzed.mp4",
            trigger_type="motion",
            analyzed=True,
        ))
        
        unanalyzed = db.get_unanalyzed_detections()
        
        assert len(unanalyzed) == 1
        assert unanalyzed[0].id == unanalyzed_id
    
    def test_get_detection_count(self, db):
        """Test getting detection count."""
        # Add 3 detections
        for i in range(3):
            db.add_detection(Detection(
                timestamp=datetime.now(),
                video_path=f"/data/videos/test_{i}.mp4",
                trigger_type="motion",
            ))
        
        count = db.get_detection_count()
        
        assert count == 3
    
    def test_get_detection_count_filtered(self, db):
        """Test getting filtered detection count."""
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/motion.mp4",
            trigger_type="motion",
        ))
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/scheduled.mp4",
            trigger_type="scheduled",
        ))
        
        motion_count = db.get_detection_count(trigger_type="motion")
        
        assert motion_count == 1


class TestBulkDelete:
    """Tests for bulk delete functionality."""
    
    def test_bulk_delete_multiple(self, db):
        """Test deleting multiple detections at once."""
        ids = []
        for i in range(5):
            detection_id = db.add_detection(Detection(
                timestamp=datetime.now(),
                video_path=f"/data/videos/test_{i}.mp4",
                trigger_type="motion",
            ))
            ids.append(detection_id)
        
        deleted = db.delete_detections_bulk(ids[:3])
        
        assert deleted == 3
        assert db.get_detection_count() == 2
    
    def test_bulk_delete_empty_list(self, db):
        """Test bulk delete with empty list."""
        deleted = db.delete_detections_bulk([])
        
        assert deleted == 0
    
    def test_bulk_delete_nonexistent_ids(self, db):
        """Test bulk delete with IDs that don't exist."""
        deleted = db.delete_detections_bulk([9998, 9999])
        
        assert deleted == 0
    
    def test_bulk_delete_mixed_ids(self, db):
        """Test bulk delete with mix of existing and non-existing IDs."""
        detection_id = db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/test.mp4",
            trigger_type="motion",
        ))
        
        deleted = db.delete_detections_bulk([detection_id, 9999])
        
        assert deleted == 1
        assert db.get_detection(detection_id) is None


class TestDailySummary:
    """Tests for daily summary functionality."""
    
    def test_update_daily_summary(self, db):
        """Test creating/updating daily summary."""
        # Add some detections
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/test1.mp4",
            trigger_type="motion",
            animal_class="bird",
            analyzed=True,
        ))
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/test2.mp4",
            trigger_type="motion",
            animal_class="bird",
            analyzed=True,
        ))
        db.add_detection(Detection(
            timestamp=datetime.now(),
            video_path="/data/videos/test3.mp4",
            trigger_type="scheduled",
            animal_class="cat",
            analyzed=True,
        ))
        
        summary = db.update_daily_summary()
        
        assert summary.date == date.today()
        assert summary.total_detections == 3
        assert summary.animal_counts.get("bird") == 2
        assert summary.animal_counts.get("cat") == 1
    
    def test_get_daily_summary(self, db):
        """Test retrieving daily summary."""
        # Create summary
        db.update_daily_summary()
        
        summary = db.get_daily_summary(date.today())
        
        assert summary is not None
        assert summary.date == date.today()
    
    def test_get_daily_summary_not_found(self, db):
        """Test retrieving non-existent summary."""
        old_date = date.today() - timedelta(days=365)
        
        summary = db.get_daily_summary(old_date)
        
        assert summary is None
    
    def test_get_daily_summaries(self, db):
        """Test getting multiple daily summaries."""
        # Create summaries for multiple days
        today = date.today()
        for i in range(3):
            day = today - timedelta(days=i)
            db.update_daily_summary(day)
        
        summaries = db.get_daily_summaries(limit=10)
        
        assert len(summaries) == 3
