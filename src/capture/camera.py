"""
Camera module for capturing video clips on Raspberry Pi.
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from enum import Enum

try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logging.warning("picamera2 not available - running in simulation mode")

logger = logging.getLogger(__name__)


class CaptureReason(Enum):
    MOTION = "motion"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class VideoMetadata:
    filepath: Path
    timestamp: datetime
    duration_seconds: float
    reason: CaptureReason
    resolution: tuple[int, int]


class Camera:
    """
    Pi Camera controller for video capture.
    
    Supports both picamera2 (Pi Camera Module) and simulation mode for development.
    """
    
    DEFAULT_RESOLUTION = (1280, 720)
    DEFAULT_FRAMERATE = 30
    DEFAULT_DURATION = 2.0
    
    def __init__(
        self,
        output_dir: Path,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        framerate: int = DEFAULT_FRAMERATE,
        default_duration: float = DEFAULT_DURATION,
        simulation_mode: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resolution = resolution
        self.framerate = framerate
        self.default_duration = default_duration
        self.simulation_mode = simulation_mode or not PICAMERA_AVAILABLE
        
        self._camera: Optional[Picamera2] = None
        self._is_recording = False
        
        if not self.simulation_mode:
            self._init_camera()
        else:
            logger.info("Camera running in simulation mode")
    
    def _init_camera(self) -> None:
        self._camera = Picamera2()
        video_config = self._camera.create_video_configuration(
            main={"size": self.resolution},
            controls={"FrameRate": self.framerate}
        )
        self._camera.configure(video_config)
        logger.info(f"Camera initialized: {self.resolution} @ {self.framerate}fps")
    
    def _generate_filename(self, reason: CaptureReason) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{reason.value}_{timestamp}.mp4"
        return self.output_dir / filename
    
    def capture_video(
        self,
        duration: Optional[float] = None,
        reason: CaptureReason = CaptureReason.MANUAL
    ) -> VideoMetadata:
        """
        Capture a video clip.
        
        Args:
            duration: Recording duration in seconds (uses default if None)
            reason: Why this capture was triggered
            
        Returns:
            VideoMetadata with capture details
        """
        if self._is_recording:
            raise RuntimeError("Recording already in progress")
        
        duration = duration or self.default_duration
        filepath = self._generate_filename(reason)
        start_time = datetime.now()
        
        logger.info(f"Starting {duration}s video capture: {filepath.name}")
        self._is_recording = True
        
        try:
            if self.simulation_mode:
                self._simulate_recording(filepath, duration)
            else:
                self._record_video(filepath, duration)
        finally:
            self._is_recording = False
        
        metadata = VideoMetadata(
            filepath=filepath,
            timestamp=start_time,
            duration_seconds=duration,
            reason=reason,
            resolution=self.resolution
        )
        
        logger.info(f"Video saved: {filepath.name}")
        return metadata
    
    def _record_video(self, filepath: Path, duration: float) -> None:
        encoder = H264Encoder()
        output = FfmpegOutput(str(filepath))
        
        self._camera.start()
        self._camera.start_encoder(encoder, output)
        time.sleep(duration)
        self._camera.stop_encoder()
        self._camera.stop()
    
    def _simulate_recording(self, filepath: Path, duration: float) -> None:
        """Create a placeholder file for simulation mode."""
        time.sleep(min(duration, 0.1))  # Brief delay to simulate
        filepath.write_text(f"SIMULATED VIDEO - Duration: {duration}s")
    
    def capture_on_motion(self) -> VideoMetadata:
        """Convenience method for motion-triggered captures."""
        return self.capture_video(reason=CaptureReason.MOTION)
    
    def capture_scheduled(self) -> VideoMetadata:
        """Convenience method for scheduled captures."""
        return self.capture_video(reason=CaptureReason.SCHEDULED)
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    def close(self) -> None:
        """Release camera resources."""
        if self._camera:
            self._camera.close()
            self._camera = None
        logger.info("Camera closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    output_path = Path(__file__).parent.parent.parent / "data" / "videos"
    camera = Camera(output_dir=output_path, simulation_mode=True)
    
    # Test captures
    meta = camera.capture_video(duration=1.5, reason=CaptureReason.MANUAL)
    print(f"Captured: {meta.filepath}")
    
    camera.close()
