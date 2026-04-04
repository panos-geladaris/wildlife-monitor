"""
SQLite database for storing detection metadata.
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "wildlife.db"


@dataclass
class Detection:
    """Represents a single video detection record."""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    video_path: str = ""
    trigger_type: str = "manual"  # 'motion', 'scheduled', 'manual'
    animal_class: Optional[str] = None
    confidence: Optional[float] = None
    bird_species: Optional[str] = None
    analyzed: bool = False
    highlighted: bool = False
    sound_class: Optional[str] = None
    sound_species: Optional[str] = None
    sound_confidence: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DailySummary:
    """Daily summary of detections."""
    id: Optional[int] = None
    date: date = field(default_factory=date.today)
    total_detections: int = 0
    animal_counts: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Timelapse:
    """Represents a daily time-lapse video."""
    id: Optional[int] = None
    date: date = field(default_factory=date.today)
    video_path: str = ""
    detection_count: int = 0
    animal_counts: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class Database:
    """
    SQLite database manager for wildlife detections.
    
    Provides CRUD operations for detections and daily summaries.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    video_path TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    animal_class TEXT,
                    confidence REAL,
                    analyzed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    total_detections INTEGER DEFAULT 0,
                    animal_counts TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
                ON detections(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_detections_trigger_type 
                ON detections(trigger_type);
                
                CREATE INDEX IF NOT EXISTS idx_detections_animal_class 
                ON detections(animal_class);

                CREATE TABLE IF NOT EXISTS frame_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL,
                    frame_timestamp REAL,
                    label TEXT,
                    score REAL,
                    x1 REAL,
                    y1 REAL,
                    x2 REAL,
                    y2 REAL,
                    img_width INTEGER,
                    img_height INTEGER,
                    animal_class TEXT,
                    animal_confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(detection_id) REFERENCES detections(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_frame_objects_detection
                ON frame_objects(detection_id);

                CREATE INDEX IF NOT EXISTS idx_frame_objects_frame
                ON frame_objects(detection_id, frame_number);

                CREATE TABLE IF NOT EXISTS timelapses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    video_path TEXT NOT NULL,
                    detection_count INTEGER DEFAULT 0,
                    animal_counts TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_timelapses_date
                ON timelapses(date);
            """)

            # Add highlighted column if it doesn't exist yet
            columns = [
                row["name"]
                for row in conn.execute("PRAGMA table_info(detections)").fetchall()
            ]
            if "highlighted" not in columns:
                conn.execute(
                    "ALTER TABLE detections ADD COLUMN highlighted BOOLEAN DEFAULT FALSE"
                )

            # Add bird_species column to detections if it doesn't exist yet
            if "bird_species" not in columns:
                conn.execute(
                    "ALTER TABLE detections ADD COLUMN bird_species TEXT"
                )

            # Add sound classification columns if they don't exist yet
            if "sound_class" not in columns:
                conn.execute(
                    "ALTER TABLE detections ADD COLUMN sound_class TEXT"
                )
            if "sound_species" not in columns:
                conn.execute(
                    "ALTER TABLE detections ADD COLUMN sound_species TEXT"
                )
            if "sound_confidence" not in columns:
                conn.execute(
                    "ALTER TABLE detections ADD COLUMN sound_confidence REAL"
                )

            # Add bird_species column to frame_objects if it doesn't exist yet
            fo_columns = [
                row["name"]
                for row in conn.execute("PRAGMA table_info(frame_objects)").fetchall()
            ]
            if "bird_species" not in fo_columns:
                conn.execute(
                    "ALTER TABLE frame_objects ADD COLUMN bird_species TEXT"
                )
        logger.info(f"Database initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def add_detection(self, detection: Detection) -> int:
        """
        Add a new detection record.
        
        Returns:
            The ID of the inserted record.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detections 
                (timestamp, video_path, trigger_type, animal_class, confidence, bird_species, analyzed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection.timestamp.isoformat(),
                    detection.video_path,
                    detection.trigger_type,
                    detection.animal_class,
                    detection.confidence,
                    detection.bird_species,
                    detection.analyzed,
                )
            )
            detection_id = cursor.lastrowid
            logger.debug(f"Added detection {detection_id}: {detection.video_path}")
            return detection_id
    
    def get_detection(self, detection_id: int) -> Optional[Detection]:
        """Get a detection by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM detections WHERE id = ?", (detection_id,)
            ).fetchone()
            
            if row:
                return self._row_to_detection(row)
            return None
    
    def get_detections(
        self,
        trigger_type: Optional[str] = None,
        animal_class: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        analyzed: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Detection]:
        """
        Get detections with optional filters.
        
        Args:
            trigger_type: Filter by trigger type ('motion', 'scheduled', 'manual')
            animal_class: Filter by detected animal class
            start_date: Filter detections after this date
            end_date: Filter detections before this date
            analyzed: Filter by analyzed status
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of Detection objects
        """
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        
        if trigger_type:
            query += " AND trigger_type = ?"
            params.append(trigger_type)
        
        if animal_class:
            query += " AND animal_class = ?"
            params.append(animal_class)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if analyzed is not None:
            query += " AND analyzed = ?"
            params.append(analyzed)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_detection(row) for row in rows]
    
    def get_unanalyzed_detections(self, limit: int = 50) -> list[Detection]:
        """Get detections that haven't been analyzed yet."""
        return self.get_detections(analyzed=False, limit=limit)
    
    def update_detection(
        self,
        detection_id: int,
        animal_class: Optional[str] = None,
        confidence: Optional[float] = None,
        analyzed: Optional[bool] = None,
        bird_species: Optional[str] = None,
        sound_class: Optional[str] = None,
        sound_species: Optional[str] = None,
        sound_confidence: Optional[float] = None,
    ) -> bool:
        """
        Update a detection record.
        
        Returns:
            True if record was updated, False if not found.
        """
        updates = []
        params = []
        
        if animal_class is not None:
            updates.append("animal_class = ?")
            params.append(animal_class)
        
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        
        if analyzed is not None:
            updates.append("analyzed = ?")
            params.append(analyzed)
        
        if bird_species is not None:
            updates.append("bird_species = ?")
            params.append(bird_species)
        
        if sound_class is not None:
            updates.append("sound_class = ?")
            params.append(sound_class)
        
        if sound_species is not None:
            updates.append("sound_species = ?")
            params.append(sound_species)
        
        if sound_confidence is not None:
            updates.append("sound_confidence = ?")
            params.append(sound_confidence)
        
        if not updates:
            return False
        
        params.append(detection_id)
        query = f"UPDATE detections SET {', '.join(updates)} WHERE id = ?"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0
    
    def delete_detection(self, detection_id: int) -> bool:
        """Delete a detection record."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM detections WHERE id = ?", (detection_id,)
            )
            return cursor.rowcount > 0

    def delete_detections_bulk(self, detection_ids: list[int]) -> int:
        """Delete multiple detection records.

        Returns:
            Number of records deleted.
        """
        if not detection_ids:
            return 0

        placeholders = ",".join("?" for _ in detection_ids)
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM detections WHERE id IN ({placeholders})",
                detection_ids,
            )
            count = cursor.rowcount
            logger.info(f"Bulk deleted {count} detections")
            return count

    def delete_old_detections(self, before_date: datetime) -> int:
        """
        Delete detections older than specified date.
        
        Returns:
            Number of records deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM detections WHERE timestamp < ?",
                (before_date.isoformat(),)
            )
            count = cursor.rowcount
            logger.info(f"Deleted {count} old detections before {before_date}")
            return count

    def add_frame_objects(self, detection_id: int, boxes: list[dict]) -> int:
        """
        Store bounding box detections for a video's analyzed frames.
        
        Args:
            detection_id: The parent detection ID
            boxes: List of dicts with keys: frame_number, frame_timestamp, label, score,
                   x1, y1, x2, y2, img_width, img_height, animal_class, animal_confidence
        
        Returns:
            Number of rows inserted.
        """
        if not boxes:
            return 0
        
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO frame_objects
                (detection_id, frame_number, frame_timestamp, label, score,
                 x1, y1, x2, y2, img_width, img_height, animal_class, animal_confidence, bird_species)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        detection_id,
                        b["frame_number"],
                        b.get("frame_timestamp"),
                        b.get("label"),
                        b.get("score"),
                        b.get("x1"),
                        b.get("y1"),
                        b.get("x2"),
                        b.get("y2"),
                        b.get("img_width"),
                        b.get("img_height"),
                        b.get("animal_class"),
                        b.get("animal_confidence"),
                        b.get("bird_species"),
                    )
                    for b in boxes
                ],
            )
            count = len(boxes)
            logger.debug(f"Added {count} frame objects for detection {detection_id}")
            return count

    def get_frame_objects(self, detection_id: int) -> list[dict]:
        """
        Get all bounding box detections for a detection.
        
        Returns:
            List of dicts with bounding box data, ordered by frame_number.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM frame_objects 
                WHERE detection_id = ? 
                ORDER BY frame_number, score DESC
                """,
                (detection_id,),
            ).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "detection_id": row["detection_id"],
                    "frame_number": row["frame_number"],
                    "frame_timestamp": row["frame_timestamp"],
                    "label": row["label"],
                    "score": row["score"],
                    "x1": row["x1"],
                    "y1": row["y1"],
                    "x2": row["x2"],
                    "y2": row["y2"],
                    "img_width": row["img_width"],
                    "img_height": row["img_height"],
                    "animal_class": row["animal_class"],
                    "animal_confidence": row["animal_confidence"],
                }
                for row in rows
            ]

    def delete_frame_objects(self, detection_id: int) -> int:
        """Delete all frame objects for a detection.
        
        Returns:
            Number of records deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM frame_objects WHERE detection_id = ?",
                (detection_id,),
            )
            count = cursor.rowcount
            if count > 0:
                logger.debug(f"Deleted {count} frame objects for detection {detection_id}")
            return count

    def get_detection_count(
        self,
        trigger_type: Optional[str] = None,
        animal_class: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        analyzed: Optional[bool] = None,
    ) -> int:
        """Get count of detections with optional filters."""
        query = "SELECT COUNT(*) FROM detections WHERE 1=1"
        params = []

        if trigger_type:
            query += " AND trigger_type = ?"
            params.append(trigger_type)

        if animal_class:
            query += " AND animal_class = ?"
            params.append(animal_class)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if analyzed is not None:
            query += " AND analyzed = ?"
            params.append(analyzed)

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row[0]

    def get_empty_detections(self, before: datetime) -> list["Detection"]:
        """
        Get analyzed detections with no recognised animal that are older than *before*.

        A detection is "empty" when:
        - analyzed = TRUE
        - animal_class IS NULL OR animal_class = 'unknown'
        - No rows exist in frame_objects for that detection_id
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM detections
                WHERE analyzed = 1
                  AND (animal_class IS NULL OR animal_class = 'unknown')
                  AND timestamp < ?
                  AND id NOT IN (
                      SELECT DISTINCT detection_id FROM frame_objects
                  )
                ORDER BY timestamp ASC
                """,
                (before.isoformat(),),
            ).fetchall()
            return [self._row_to_detection(row) for row in rows]

    def set_highlighted(self, detection_id: int, highlighted: bool) -> bool:
        """Set the highlighted flag on a detection. Returns True if updated."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE detections SET highlighted = ? WHERE id = ?",
                (highlighted, detection_id),
            )
            return cursor.rowcount > 0

    def get_highlighted_detections(
        self, limit: int = 100, offset: int = 0
    ) -> list["Detection"]:
        """Get all highlighted detections, newest first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE highlighted = 1 ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_detection(row) for row in rows]

    def _row_to_detection(self, row: sqlite3.Row) -> Detection:
        """Convert a database row to Detection object."""
        keys = row.keys()
        return Detection(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            video_path=row["video_path"],
            trigger_type=row["trigger_type"],
            animal_class=row["animal_class"],
            confidence=row["confidence"],
            bird_species=row["bird_species"] if "bird_species" in keys else None,
            analyzed=bool(row["analyzed"]),
            highlighted=bool(row["highlighted"]) if row["highlighted"] is not None else False,
            sound_class=row["sound_class"] if "sound_class" in keys else None,
            sound_species=row["sound_species"] if "sound_species" in keys else None,
            sound_confidence=row["sound_confidence"] if "sound_confidence" in keys else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
        )
    
    # Daily Summary Methods
    
    def update_daily_summary(self, summary_date: Optional[date] = None) -> DailySummary:
        """
        Update or create daily summary for the given date.
        
        Calculates summary from detections table.
        """
        summary_date = summary_date or date.today()
        start = datetime.combine(summary_date, datetime.min.time())
        end = datetime.combine(summary_date, datetime.max.time())
        
        with self._get_connection() as conn:
            # Get total detections
            total = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE timestamp >= ? AND timestamp <= ?",
                (start.isoformat(), end.isoformat())
            ).fetchone()[0]
            
            # Get animal counts
            rows = conn.execute(
                """
                SELECT animal_class, COUNT(*) as count 
                FROM detections 
                WHERE timestamp >= ? AND timestamp <= ? AND animal_class IS NOT NULL
                GROUP BY animal_class
                """,
                (start.isoformat(), end.isoformat())
            ).fetchall()
            
            animal_counts = {row["animal_class"]: row["count"] for row in rows}
            
            # Upsert summary
            conn.execute(
                """
                INSERT INTO daily_summary (date, total_detections, animal_counts)
                VALUES (?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_detections = excluded.total_detections,
                    animal_counts = excluded.animal_counts
                """,
                (summary_date.isoformat(), total, json.dumps(animal_counts))
            )
            
            return DailySummary(
                date=summary_date,
                total_detections=total,
                animal_counts=animal_counts,
            )
    
    def get_daily_summary(self, summary_date: date) -> Optional[DailySummary]:
        """Get daily summary for a specific date."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_summary WHERE date = ?",
                (summary_date.isoformat(),)
            ).fetchone()
            
            if row:
                return DailySummary(
                    id=row["id"],
                    date=date.fromisoformat(row["date"]),
                    total_detections=row["total_detections"],
                    animal_counts=json.loads(row["animal_counts"]) if row["animal_counts"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                )
            return None
    
    def get_daily_summaries(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 30,
    ) -> list[DailySummary]:
        """Get daily summaries for a date range."""
        query = "SELECT * FROM daily_summary WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                DailySummary(
                    id=row["id"],
                    date=date.fromisoformat(row["date"]),
                    total_detections=row["total_detections"],
                    animal_counts=json.loads(row["animal_counts"]) if row["animal_counts"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                )
                for row in rows
            ]
    
    # Timelapse Methods

    def add_timelapse(self, timelapse: Timelapse) -> int:
        """Add a new timelapse record. Returns the ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO timelapses
                (date, video_path, detection_count, animal_counts)
                VALUES (?, ?, ?, ?)
                """,
                (
                    timelapse.date.isoformat(),
                    timelapse.video_path,
                    timelapse.detection_count,
                    json.dumps(timelapse.animal_counts),
                ),
            )
            timelapse_id = cursor.lastrowid
            logger.debug(f"Added timelapse {timelapse_id} for {timelapse.date}")
            return timelapse_id

    def get_timelapse(self, timelapse_id: int) -> Optional[Timelapse]:
        """Get a timelapse by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM timelapses WHERE id = ?", (timelapse_id,)
            ).fetchone()
            if row:
                return self._row_to_timelapse(row)
            return None

    def get_timelapse_by_date(self, dt: date) -> Optional[Timelapse]:
        """Get a timelapse for a specific date."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM timelapses WHERE date = ?", (dt.isoformat(),)
            ).fetchone()
            if row:
                return self._row_to_timelapse(row)
            return None

    def get_timelapses(self, limit: int = 50, offset: int = 0) -> list[Timelapse]:
        """Get timelapses ordered by date descending."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM timelapses ORDER BY date DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_timelapse(row) for row in rows]

    def delete_timelapse(self, timelapse_id: int) -> bool:
        """Delete a timelapse record. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM timelapses WHERE id = ?", (timelapse_id,)
            )
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted timelapse {timelapse_id}")
            return deleted

    def _row_to_timelapse(self, row: sqlite3.Row) -> Timelapse:
        """Convert a database row to Timelapse object."""
        return Timelapse(
            id=row["id"],
            date=date.fromisoformat(row["date"]),
            video_path=row["video_path"],
            detection_count=row["detection_count"],
            animal_counts=json.loads(row["animal_counts"]) if row["animal_counts"] else {},
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
        )

    def close(self) -> None:
        """Close database (no-op for SQLite with context manager pattern)."""
        logger.info("Database closed")


if __name__ == "__main__":
    import tempfile
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Database(Path(f.name))
    
    # Add a detection
    detection = Detection(
        timestamp=datetime.now(),
        video_path="/data/videos/motion_20240115_120000.mp4",
        trigger_type="motion",
    )
    detection_id = db.add_detection(detection)
    print(f"Added detection: {detection_id}")
    
    # Retrieve it
    retrieved = db.get_detection(detection_id)
    print(f"Retrieved: {retrieved}")
    
    # Update with analysis results
    db.update_detection(detection_id, animal_class="bird", confidence=0.85, analyzed=True)
    
    # Get updated
    updated = db.get_detection(detection_id)
    print(f"Updated: {updated}")
    
    # Get daily summary
    summary = db.update_daily_summary()
    print(f"Daily summary: {summary}")
    
    db.close()
