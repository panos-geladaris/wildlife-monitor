"""
Main capture service that orchestrates motion detection, camera, and scheduling.
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from .motion_detector import MotionDetector, MotionEvent
from .camera import Camera, CaptureReason, VideoMetadata
from .scheduler import CaptureScheduler
from .config import CaptureServiceConfig, load_config
from .daylight import DaylightGate, SunriseSunsetClient

logger = logging.getLogger(__name__)


class CaptureService:
    """
    Orchestrates all capture components:
    - Motion detection triggers immediate video capture
    - Scheduler triggers hourly video capture
    - All captures are logged and can trigger callbacks
    """
    
    def __init__(self, config: CaptureServiceConfig, audio_recorder=None):
        self.config = config
        
        self._camera = Camera(
            output_dir=config.video_output_dir,
            resolution=config.resolution,
            framerate=config.framerate,
            default_duration=config.video_duration,
            simulation_mode=config.simulation_mode,
            audio_recorder=audio_recorder,
        )
        
        if config.zoom_level != 1.0:
            self._camera.set_zoom(config.zoom_level)
        if config.autofocus:
            self._camera.set_autofocus(True)
        
        self._motion_detector = MotionDetector(
            gpio_pin=config.gpio_pin,
            cooldown_seconds=config.cooldown_seconds,
            simulation_mode=config.simulation_mode
        )
        
        self._scheduler = CaptureScheduler()
        self._interval_scheduler = CaptureScheduler()
        
        self._on_capture_callback: Optional[Callable[[VideoMetadata], None]] = None
        self._motion_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._daylight_gate: Optional[DaylightGate] = None
        self._daylight_timer: Optional[threading.Timer] = None
        if config.daylight_enabled:
            client = SunriseSunsetClient()
            self._daylight_gate = DaylightGate(
                client=client,
                lat=config.daylight_lat,
                lng=config.daylight_lng,
                tzid=config.daylight_tzid,
                start_offset_minutes=config.daylight_start_offset_minutes,
                end_offset_minutes=config.daylight_end_offset_minutes,
                fallback=config.daylight_fallback,
            )
        
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        self._motion_detector.on_motion(self._handle_motion)
        self._scheduler.set_capture_callback(self._handle_scheduled_capture)
        self._interval_scheduler.set_capture_callback(self._handle_interval_capture)
    
    def on_capture(self, callback: Callable[[VideoMetadata], None]) -> None:
        """
        Register callback for all video captures.
        
        The callback receives VideoMetadata for each capture, allowing
        downstream processing (e.g., ML analysis, database storage).
        """
        self._on_capture_callback = callback
    
    def _capture_allowed(self) -> bool:
        if self._daylight_gate is None:
            return True
        return self._daylight_gate.is_capture_allowed()

    def _handle_motion(self, event: MotionEvent) -> None:
        """Handle motion detection by capturing video."""
        if not self._capture_allowed():
            logger.info("Motion ignored (outside daylight window)")
            return
        logger.info(f"Motion triggered capture at {event.timestamp}")
        try:
            metadata = self._camera.capture_on_motion()
            self._notify_capture(metadata)
        except Exception as e:
            logger.error(f"Motion capture failed: {e}")
    
    def _handle_scheduled_capture(self) -> None:
        """Handle hourly scheduled capture."""
        if not self._capture_allowed():
            logger.info("Scheduled capture skipped (outside daylight window)")
            return
        try:
            metadata = self._camera.capture_video(
                duration=self.config.scheduled_video_duration,
                reason=CaptureReason.SCHEDULED
            )
            self._notify_capture(metadata)
        except Exception as e:
            logger.error(f"Scheduled capture failed: {e}")
    
    def _handle_interval_capture(self) -> None:
        """Handle 15-minute interval captures with multiple zoom levels."""
        if not self._capture_allowed():
            logger.info("Interval capture skipped (outside daylight window)")
            return
        original_zoom = self._camera.get_zoom()
        
        for zoom_level in self.config.interval_capture_zoom_levels:
            try:
                self._camera.set_zoom(zoom_level)
                logger.info(f"Interval capture at {zoom_level}x zoom")
                metadata = self._camera.capture_video(
                    duration=self.config.interval_capture_duration,
                    reason=CaptureReason.SCHEDULED
                )
                self._notify_capture(metadata)
            except Exception as e:
                logger.error(f"Interval capture at {zoom_level}x failed: {e}")
        
        self._camera.set_zoom(original_zoom)
    
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
        
        # Start hourly scheduled captures
        self._scheduler.schedule_hourly(minute=self.config.hourly_capture_minute)
        self._scheduler.start()
        
        # Start 15-minute interval captures with zoom comparison
        if self.config.interval_capture_enabled:
            self._interval_scheduler.schedule_interval(minutes=self.config.interval_capture_minutes)
            self._interval_scheduler.start()
        
        # Start motion detection in background thread
        self._motion_thread = threading.Thread(
            target=self._motion_detector.start_monitoring,
            daemon=True
        )
        self._motion_thread.start()
        
        if self._daylight_gate is not None:
            self._daylight_gate.refresh_if_needed(self._daylight_gate._now_fn())
            allowed = self._daylight_gate.is_capture_allowed()
            sun = self._daylight_gate.sun_times
            if sun:
                logger.info(
                    "Daylight capture: sunrise %s — sunset %s (currently %s)",
                    sun.sunrise.strftime("%H:%M"),
                    sun.sunset.strftime("%H:%M"),
                    "active" if allowed else "paused",
                )
            self._schedule_daylight_transition()
        
        logger.info("Capture service started")
        logger.info(f"  - Simulation mode: {self.config.simulation_mode}")
        logger.info(f"  - Motion detection: GPIO {self.config.gpio_pin}")
        logger.info(f"  - Hourly captures: minute {self.config.hourly_capture_minute}")
        logger.info(f"  - Video duration: {self.config.video_duration}s")
        logger.info(f"  - Zoom level: {self.config.zoom_level}x")
        logger.info(f"  - Autofocus: {self.config.autofocus}")
        logger.info(f"  - Daylight gating: {self.config.daylight_enabled}")
        logger.info(f"  - Output: {self.config.video_output_dir}")
    
    def _schedule_daylight_transition(self) -> None:
        if self._daylight_gate is None:
            return
        transition = self._daylight_gate.next_transition()
        if transition is None:
            return
        transition_time, will_be_allowed = transition
        now = self._daylight_gate._now_fn()
        delay = max(0, (transition_time - now).total_seconds())
        label = "sunrise" if will_be_allowed else "sunset"
        logger.info("Next daylight transition (%s) in %.0f minutes", label, delay / 60)

        def _on_transition():
            if not self._running:
                return
            state = "active" if will_be_allowed else "paused"
            logger.info("Daylight transition: capture is now %s", state)
            self._schedule_daylight_transition()

        self._daylight_timer = threading.Timer(delay, _on_transition)
        self._daylight_timer.daemon = True
        self._daylight_timer.start()

    def stop(self) -> None:
        """Stop the capture service."""
        self._running = False
        if self._daylight_timer is not None:
            self._daylight_timer.cancel()
            self._daylight_timer = None
        self._scheduler.stop()
        self._motion_detector.stop()
        self._camera.close()
        
        if self._motion_thread:
            self._motion_thread.join(timeout=2.0)
        
        logger.info("Capture service stopped")
    
    def trigger_manual_capture(self, duration: float = None) -> VideoMetadata:
        """Manually trigger a video capture."""
        metadata = self._camera.capture_video(
            duration=duration, reason=CaptureReason.MANUAL
        )
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


def _wait_for_exit():
    """Wait for Ctrl+X or Ctrl+C to exit."""
    import sys
    import time
    import signal
    
    stop_flag = False
    
    def handle_signal(signum, frame):
        nonlocal stop_flag
        stop_flag = True
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    print("Capture service running. Press Ctrl+C to stop.")
    
    try:
        import termios
        import tty
        import select
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())  # cbreak instead of raw - allows output
            while not stop_flag:
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x18':  # Ctrl+X
                        print("\nShutting down...")
                        return
                time.sleep(0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except (ImportError, termios.error):
        while not stop_flag:
            time.sleep(0.5)
    
    print("\nShutting down...")


if __name__ == "__main__":
    import os
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load config from file (or use --config path)
    config_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        config_path = Path(sys.argv[2])
    
    config = load_config(config_path)
    
    # Auto-detect simulation mode unless on Pi (or set SIMULATE env var)
    simulate = os.environ.get("SIMULATE", "auto")
    if simulate == "auto":
        try:
            import RPi.GPIO
            config.simulation_mode = False
        except ImportError:
            config.simulation_mode = True
    else:
        config.simulation_mode = simulate == "1"
    
    service = CaptureService(config)
    
    def on_new_capture(metadata: VideoMetadata):
        print(f"\n>>> New capture: {metadata.filepath.name}")
        print(f"    Reason: {metadata.reason.value}")
        print(f"    Duration: {metadata.duration_seconds}s\n")
    
    service.on_capture(on_new_capture)
    service.start()
    
    try:
        _wait_for_exit()
    finally:
        service.stop()
