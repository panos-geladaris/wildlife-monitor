"""
Automatic cleanup of empty detections.

Removes analyzed detections where neither the classifier nor the object
detector recognised any animal, along with all associated artifacts
(video file, thumbnail, annotated frames).
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .database import Database, Detection

logger = logging.getLogger(__name__)


def cleanup_detection(db: Database, detection: Detection, video_dir: Path) -> None:
    """Delete a single detection and all its artifacts.

    Removes:
    - The video file
    - The thumbnail image
    - The annotated frames directory
    - The database record (cascades to frame_objects)
    """
    video_path = Path(detection.video_path)
    if video_path.exists():
        video_path.unlink()

    thumb_path = video_dir / "thumbnails" / (video_path.stem + ".jpg")
    if thumb_path.exists():
        thumb_path.unlink()

    annotated_dir = video_dir.parent / "annotated" / str(detection.id)
    if annotated_dir.exists():
        shutil.rmtree(annotated_dir)

    db.delete_detection(detection.id)


def run_cleanup(
    db: Database,
    video_dir: Path,
    max_age_hours: float = 24,
) -> int:
    """Remove empty detections older than *max_age_hours*.

    Returns:
        Number of detections deleted.
    """
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    detections = db.get_empty_detections(before=cutoff)

    if not detections:
        logger.debug("Cleanup: no empty detections to remove")
        return 0

    for detection in detections:
        try:
            cleanup_detection(db, detection, video_dir)
        except Exception as e:
            logger.error(f"Cleanup failed for detection {detection.id}: {e}")

    db.update_daily_summary()

    logger.info(f"Cleanup: removed {len(detections)} empty detections older than {max_age_hours}h")
    return len(detections)
