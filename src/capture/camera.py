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
    audio_path: Optional[Path] = None


class Camera:
    """
    Pi Camera controller for video capture.
    
    Supports both picamera2 (Pi Camera Module) and simulation mode for development.
    Includes digital zoom and autofocus controls for Camera Module 3.
    """
    
    DEFAULT_RESOLUTION = (1280, 720)
    DEFAULT_FRAMERATE = 30
    DEFAULT_DURATION = 2.0
    MIN_ZOOM = 1.0
    MAX_ZOOM = 10.0
    
    def __init__(
        self,
        output_dir: Path,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        framerate: int = DEFAULT_FRAMERATE,
        default_duration: float = DEFAULT_DURATION,
        simulation_mode: bool = False,
        audio_recorder=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resolution = resolution
        self.framerate = framerate
        self.default_duration = default_duration
        self.simulation_mode = simulation_mode or not PICAMERA_AVAILABLE
        self._audio_recorder = audio_recorder
        
        self._camera: Optional[Picamera2] = None
        self._is_recording = False
        self._zoom_level: float = 1.0
        self._sensor_size: Optional[tuple[int, int]] = None
        
        if not self.simulation_mode:
            self._init_camera()
        else:
            logger.info("Camera running in simulation mode")
    
    def _init_camera(self) -> None:
        self._camera = Picamera2()
        self._sensor_size = self._camera.camera_properties.get('PixelArraySize', (4608, 2592))
        video_config = self._camera.create_video_configuration(
            main={"size": self.resolution},
            controls={"FrameRate": self.framerate}
        )
        self._camera.configure(video_config)
        logger.info(f"Camera initialized: {self.resolution} @ {self.framerate}fps")
        logger.info(f"Sensor size: {self._sensor_size}")
    
    def set_zoom(self, zoom_level: float) -> None:
        """
        Set digital zoom level (1.0 = no zoom, 2.0 = 2x zoom, etc.).
        
        Args:
            zoom_level: Zoom factor between 1.0 and 10.0
        """
        zoom_level = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom_level))
        self._zoom_level = zoom_level
        
        if self.simulation_mode or not self._camera or not self._sensor_size:
            logger.info(f"Zoom set to {zoom_level}x (simulated)")
            return
        
        sensor_w, sensor_h = self._sensor_size
        crop_w = int(sensor_w / zoom_level)
        crop_h = int(sensor_h / zoom_level)
        crop_x = (sensor_w - crop_w) // 2
        crop_y = (sensor_h - crop_h) // 2
        
        self._camera.set_controls({"ScalerCrop": (crop_x, crop_y, crop_w, crop_h)})
        logger.info(f"Zoom set to {zoom_level}x (crop: {crop_w}x{crop_h})")
    
    def get_zoom(self) -> float:
        """Get current zoom level."""
        return self._zoom_level
    
    def reset_zoom(self) -> None:
        """Reset zoom to 1.0 (no zoom)."""
        self.set_zoom(1.0)
    
    def set_autofocus(self, enabled: bool = True) -> None:
        """
        Enable or disable continuous autofocus.
        
        Args:
            enabled: True for continuous AF, False for manual focus
        """
        if self.simulation_mode or not self._camera:
            logger.info(f"Autofocus {'enabled' if enabled else 'disabled'} (simulated)")
            return
        
        af_mode = 2 if enabled else 0  # 2 = Continuous, 0 = Manual
        self._camera.set_controls({"AfMode": af_mode})
        logger.info(f"Autofocus {'enabled' if enabled else 'disabled'}")
    
    def set_manual_focus(self, distance: float) -> None:
        """
        Set manual focus distance.
        
        Args:
            distance: Focus distance in metres (0.0 = infinity, higher = closer)
                      Typical range: 0.0 (infinity) to 10.0 (very close)
        """
        if self.simulation_mode or not self._camera:
            logger.info(f"Manual focus set to {distance}m (simulated)")
            return
        
        lens_position = 1.0 / distance if distance > 0 else 0.0
        self._camera.set_controls({"AfMode": 0, "LensPosition": lens_position})
        logger.info(f"Manual focus set to {distance}m (lens position: {lens_position})")
    
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
        
        audio_path = None
        if self._audio_recorder and not self.simulation_mode:
            wav_path = filepath.with_suffix(".wav")
            audio_path = self._audio_recorder.start(wav_path, duration)
        
        try:
            if self.simulation_mode:
                self._simulate_recording(filepath, duration)
            else:
                self._record_video(filepath, duration)
        finally:
            self._is_recording = False
        
        if self._audio_recorder and audio_path:
            audio_path = self._audio_recorder.wait()
        
        metadata = VideoMetadata(
            filepath=filepath,
            timestamp=start_time,
            duration_seconds=duration,
            reason=reason,
            resolution=self.resolution,
            audio_path=audio_path,
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
