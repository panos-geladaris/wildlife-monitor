"""
Thumbnail generation for captured videos using OpenCV.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - thumbnail generation disabled")

DEFAULT_THUMBNAIL_WIDTH = 320
DEFAULT_THUMBNAIL_HEIGHT = 180


def generate_thumbnail(
    video_path: Path,
    output_path: Optional[Path] = None,
    width: int = DEFAULT_THUMBNAIL_WIDTH,
    height: int = DEFAULT_THUMBNAIL_HEIGHT,
    time_seconds: float = 0.5,
) -> Optional[Path]:
    """
    Generate a JPEG thumbnail from a video file.

    Extracts a frame at ``time_seconds`` (default 0.5 s) and writes it as a
    JPEG image.  When *output_path* is ``None`` the thumbnail is placed next
    to the video with a ``.jpg`` extension inside a ``thumbnails/``
    subdirectory relative to the video's parent directory.

    Returns the path to the thumbnail, or ``None`` if generation failed.
    """
    if not CV2_AVAILABLE:
        logger.warning("Cannot generate thumbnail: OpenCV not available")
        return None

    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning(f"Video not found: {video_path}")
        return None

    if output_path is None:
        thumb_dir = video_path.parent / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumb_dir / (video_path.stem + ".jpg")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame = int(time_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from {video_path}")
                return None

        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), resized)
        logger.debug(f"Thumbnail saved: {output_path}")
        return output_path
    finally:
        cap.release()
