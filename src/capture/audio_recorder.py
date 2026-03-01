"""
Audio recording from USB microphone using FFmpeg.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"


class AudioRecorder:
    """
    Records audio from an ALSA device via FFmpeg.

    Designed to run concurrently with video capture.  If the microphone
    or FFmpeg is unavailable the recorder degrades gracefully — it will
    never block or break video capture.
    """

    DEFAULT_DEVICE = "plughw:1,0"
    DEFAULT_SAMPLE_RATE = 44100

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self._process: Optional[subprocess.Popen] = None
        self._output_path: Optional[Path] = None

    def start(self, output_path: Path, duration: float) -> Optional[Path]:
        """
        Begin recording audio in the background.

        Args:
            output_path: Destination ``.wav`` file.
            duration: Maximum recording length in seconds.

        Returns:
            The output path on success, or ``None`` if recording could
            not be started.
        """
        if self._process is not None:
            logger.warning("Audio recording already in progress")
            return None

        self._output_path = Path(output_path)

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-f", "alsa",
            "-i", self.device,
            "-t", str(duration),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            str(self._output_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.debug(f"Audio recording started: {self._output_path.name}")
            return self._output_path
        except Exception as e:
            logger.warning(f"Failed to start audio recording: {e}")
            self._process = None
            return None

    def wait(self) -> Optional[Path]:
        """
        Block until the recording finishes.

        Returns:
            Path to the recorded file, or ``None`` if recording failed.
        """
        if self._process is None:
            return None

        try:
            self._process.wait(timeout=30)
            if self._process.returncode != 0:
                stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
                logger.warning(f"Audio recording failed (rc={self._process.returncode}): {stderr[:200]}")
                self._cleanup()
                return None
        except subprocess.TimeoutExpired:
            logger.warning("Audio recording timed out, killing process")
            self.stop()
            return None
        finally:
            self._process = None

        if self._output_path and self._output_path.exists() and self._output_path.stat().st_size > 0:
            logger.debug(f"Audio recorded: {self._output_path.name}")
            return self._output_path

        self._cleanup()
        return None

    def stop(self) -> None:
        """Terminate a running recording early."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None

    def _cleanup(self) -> None:
        """Remove a failed/empty output file."""
        if self._output_path and self._output_path.exists():
            try:
                self._output_path.unlink()
            except OSError:
                pass
        self._output_path = None
