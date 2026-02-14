"""
Daily time-lapse video generation from detection frames.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - timelapse generation disabled")

DEFAULT_FPS = 2.0
DEFAULT_RESOLUTION = (1280, 720)


def generate_timelapse(
    db,
    video_dir: Path,
    target_date: date,
    fps: float = DEFAULT_FPS,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> Optional[int]:
    """
    Generate a time-lapse video from all detections on a given date.

    Extracts one representative frame per detection video and stitches
    them into an MP4.  Skips generation if there are no detections or a
    timelapse already exists for the date.

    Args:
        db: Database instance
        video_dir: Base video directory (e.g. data/videos)
        target_date: The day to generate a timelapse for
        fps: Playback frame rate (lower = slower)
        resolution: Output video resolution (width, height)

    Returns:
        Timelapse ID on success, None if skipped or failed.
    """
    if not CV2_AVAILABLE:
        logger.warning("Cannot generate timelapse: OpenCV not available")
        return None

    existing = db.get_timelapse_by_date(target_date)
    if existing is not None:
        logger.info(f"Timelapse already exists for {target_date.isoformat()}, skipping")
        return None

    start = datetime.combine(target_date, datetime.min.time())
    end = datetime.combine(target_date, datetime.max.time())
    detections = db.get_detections(start_date=start, end_date=end, limit=10000)

    if not detections:
        logger.info(f"No detections for {target_date.isoformat()}, skipping timelapse")
        return None

    from src.analysis.frame_extractor import FrameExtractor

    frames = []
    for det in detections:
        video_path = Path(det.video_path)
        if not video_path.exists():
            logger.debug(f"Video not found, skipping: {video_path}")
            continue
        try:
            with FrameExtractor(video_path) as extractor:
                frame = extractor.extract_middle_frame()
                if frame is not None:
                    frames.append(frame.image)
        except Exception as e:
            logger.warning(f"Failed to extract frame from {video_path}: {e}")

    if not frames:
        logger.info(f"No frames extracted for {target_date.isoformat()}, skipping timelapse")
        return None

    timelapse_dir = video_dir.parent / "timelapses"
    timelapse_dir.mkdir(parents=True, exist_ok=True)
    filename = f"timelapse_{target_date.strftime('%Y%m%d')}.mp4"
    output_path = timelapse_dir / filename

    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        import numpy as np

        for pil_image in frames:
            img = pil_image.convert("RGB").resize((width, height))
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.error(f"Timelapse video creation failed: {output_path}")
        return None

    logger.info(
        f"Timelapse created: {output_path.name} "
        f"({len(frames)} frames, {len(frames) / fps:.1f}s)"
    )

    from src.storage.thumbnail import generate_thumbnail as gen_thumb
    try:
        gen_thumb(output_path)
    except Exception as e:
        logger.warning(f"Timelapse thumbnail generation failed: {e}")

    animal_counts: dict[str, int] = {}
    for det in detections:
        if det.animal_class and det.animal_class != "unknown":
            animal_counts[det.animal_class] = animal_counts.get(det.animal_class, 0) + 1

    from src.storage.database import Timelapse

    timelapse = Timelapse(
        date=target_date,
        video_path=str(output_path),
        detection_count=len(detections),
        animal_counts=animal_counts,
    )
    timelapse_id = db.add_timelapse(timelapse)
    logger.info(f"Timelapse record saved: id={timelapse_id}, date={target_date.isoformat()}")
    return timelapse_id


def run_timelapse_job(
    db,
    video_dir: Path,
    target_date: Optional[date] = None,
    fps: float = DEFAULT_FPS,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> None:
    """
    Scheduler-friendly wrapper for generate_timelapse.

    Defaults to today's date when target_date is None.
    """
    target_date = target_date or date.today()
    logger.info(f"Running timelapse job for {target_date.isoformat()}")
    try:
        generate_timelapse(db, video_dir, target_date, fps=fps, resolution=resolution)
    except Exception as e:
        logger.error(f"Timelapse job failed for {target_date.isoformat()}: {e}")
