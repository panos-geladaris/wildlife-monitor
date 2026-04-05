"""
Mux audio into an existing video file using FFmpeg.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


def mux_audio(video_path: Path, audio_path: Path) -> bool:
    """
    Merge an audio track into an MP4 video.

    The video stream is copied without re-encoding.  Audio is encoded
    as AAC for browser compatibility.  ``-movflags +faststart`` is used
    so the resulting file can be streamed in the web UI.

    On success the original video is atomically replaced and the WAV
    file is deleted.

    Args:
        video_path: Path to the ``.mp4`` video.
        audio_path: Path to the ``.wav`` audio.

    Returns:
        ``True`` if muxing succeeded, ``False`` otherwise.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not video_path.exists() or not audio_path.exists():
        logger.warning(f"Mux skipped – missing file: video={video_path.exists()}, audio={audio_path.exists()}")
        return False

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=video_path.parent)
    tmp_path = Path(tmp_path)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-shortest",
        str(tmp_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:200]
            logger.warning(f"FFmpeg mux failed (rc={result.returncode}): {stderr}")
            tmp_path.unlink(missing_ok=True)
            return False

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            logger.warning("FFmpeg mux produced empty file")
            tmp_path.unlink(missing_ok=True)
            return False

        original_size = video_path.stat().st_size
        output_size = tmp_path.stat().st_size
        if output_size < original_size * 0.8:
            logger.warning(
                f"FFmpeg output too small ({output_size} B < 80% of {original_size} B), "
                "refusing to replace original"
            )
            tmp_path.unlink(missing_ok=True)
            return False

        # Atomically replace original video
        tmp_path.replace(video_path)
        audio_path.unlink(missing_ok=True)
        logger.info(f"Audio muxed into {video_path.name}")
        return True

    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg mux timed out")
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        logger.error(f"Audio mux error: {e}")
        tmp_path.unlink(missing_ok=True)
        return False
