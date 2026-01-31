"""
Main capture service that orchestrates motion detection, camera, and scheduling.
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

from .motion_detector import MotionDetector, MotionEvent
from .camera import Camera, CaptureReason, VideoMetadata
from .scheduler import CaptureScheduler

logger = logging.getLogger(__name__)


@dataclass
class CaptureServiceConfig:
    """Configuration for the capture service."""
    video_output_dir: Path
    gpio_pin: int = 17
    video_duration: float = 2.0
    resolution: tuple[int, int] = (1280, 720)
    framerate: int = 30
    cooldown_seconds: float = 5.0
    hourly_capture_minute: int = 0
    simulation_mode: bool = False


class CaptureService:
    """
    Orchestrates all capture components:
    - Motion detection triggers immediate video capture
    - Scheduler triggers hourly video capture
    - All captures are logged and can trigger callbacks
    """
    
    def __init__(self, config: CaptureServiceConfig):
        self.config = config
        
        self._camera = Camera(
            output_dir=config.video_output_dir,
            resolution=config.resolution,
            framerate=config.framerate,
            default_duration=config.video_duration,
            simulation_mode=config.simulation_mode
        )
        
        self._motion_detector = MotionDetector(
            gpio_pin=config.gpio_pin,
            cooldown_seconds=config.cooldown_seconds,
            simulation_mode=config.simulation_mode
        )
        
        self._scheduler = CaptureScheduler()
        
        self._on_capture_callback: Optional[Callable[[VideoMetadata], None]] = None
        self._motion_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        self._motion_detector.on_motion(self._handle_motion)
        self._scheduler.set_capture_callback(self._handle_scheduled_capture)
    
    def on_capture(self, callback: Callable[[VideoMetadata], None]) -> None:
        """
        Register callback for all video captures.
        
        The callback receives VideoMetadata for each capture, allowing
        downstream processing (e.g., ML analysis, database storage).
        """
        self._on_capture_callback = callback
    
    def _handle_motion(self, event: MotionEvent) -> None:
        """Handle motion detection by capturing video."""
        logger.info(f"Motion triggered capture at {event.timestamp}")
        try:
            metadata = self._camera.capture_on_motion()
            self._notify_capture(metadata)
        except Exception as e:
            logger.error(f"Motion capture failed: {e}")
    
    def _handle_scheduled_capture(self) -> None:
        """Handle scheduled capture."""
        try:
            metadata = self._camera.capture_scheduled()
            self._notify_capture(metadata)
        except Exception as e:
            logger.error(f"Scheduled capture failed: {e}")
    
    def _notify_capture(self, metadata: VideoMetadata) -> None:
        """Notify callback of new capture."""
        if self._on_capture_callback:
            try:
                self._on_capture_callback(metadata)
            except Exception as e:
                logger.error(f"Capture callback failed: {e}")
    
    def start(self) -> None:
        """Start the capture service."""
        if self._running:
            logger.warning("Capture service already running")
            return
        
        self._running = True
        
        # Start scheduled captures
        self._scheduler.schedule_hourly(minute=self.config.hourly_capture_minute)
        self._scheduler.start()
        
        # Start motion detection in background thread
        self._motion_thread = threading.Thread(
            target=self._motion_detector.start_monitoring,
            daemon=True
        )
        self._motion_thread.start()
        
        logger.info("Capture service started")
        logger.info(f"  - Motion detection: GPIO {self.config.gpio_pin}")
        logger.info(f"  - Hourly captures: minute {self.config.hourly_capture_minute}")
        logger.info(f"  - Video duration: {self.config.video_duration}s")
        logger.info(f"  - Output: {self.config.video_output_dir}")
    
    def stop(self) -> None:
        """Stop the capture service."""
        self._running = False
        self._scheduler.stop()
        self._motion_detector.stop()
        self._camera.close()
        
        if self._motion_thread:
            self._motion_thread.join(timeout=2.0)
        
        logger.info("Capture service stopped")
    
    def trigger_manual_capture(self) -> VideoMetadata:
        """Manually trigger a video capture."""
        metadata = self._camera.capture_video(reason=CaptureReason.MANUAL)
        self._notify_capture(metadata)
        return metadata
    
    def simulate_motion(self) -> None:
        """Simulate motion event for testing."""
        if self.config.simulation_mode:
            self._motion_detector.simulate_motion()
        else:
            logger.warning("Simulation only available in simulation mode")
    
    @property
    def is_running(self) -> bool:
        return self._running


if __name__ == "__main__":
    import time
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    config = CaptureServiceConfig(
        video_output_dir=Path(__file__).parent.parent.parent / "data" / "videos",
        video_duration=1.5,
        simulation_mode=True
    )
    
    service = CaptureService(config)
    
    def on_new_capture(metadata: VideoMetadata):
        print(f"\n>>> New capture: {metadata.filepath.name}")
        print(f"    Reason: {metadata.reason.value}")
        print(f"    Duration: {metadata.duration_seconds}s\n")
    
    service.on_capture(on_new_capture)
    service.start()
    
    try:
        # Test manual capture
        service.trigger_manual_capture()
        
        # Test simulated motion
        time.sleep(1)
        service.simulate_motion()
        
        # Keep running
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
