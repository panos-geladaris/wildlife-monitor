"""
Motion detector module using PIR sensor connected to Raspberry Pi GPIO.
"""

import time
import logging
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False
    logging.warning("RPi.GPIO not available - running in simulation mode")

logger = logging.getLogger(__name__)


class SensorState(Enum):
    IDLE = "idle"
    MOTION_DETECTED = "motion_detected"
    COOLDOWN = "cooldown"


@dataclass
class MotionEvent:
    timestamp: float
    duration: float


class MotionDetector:
    """
    PIR motion sensor controller.
    
    Typical wiring for HC-SR501:
    - VCC -> 5V (Pin 2)
    - GND -> Ground (Pin 6)
    - OUT -> GPIO pin (default: GPIO 17, Pin 11)
    """
    
    DEFAULT_GPIO_PIN = 17
    DEFAULT_COOLDOWN_SECONDS = 3.0
    
    def __init__(
        self,
        gpio_pin: int = DEFAULT_GPIO_PIN,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        simulation_mode: bool = False
    ):
        self.gpio_pin = gpio_pin
        self.cooldown_seconds = cooldown_seconds
        self.simulation_mode = simulation_mode or not RPI_AVAILABLE
        self.state = SensorState.IDLE
        self._last_motion_time: float = 0
        self._running = False
        self._on_motion_callback: Optional[Callable[[MotionEvent], None]] = None
        
        if not self.simulation_mode:
            self._setup_gpio()
        else:
            logger.info("Motion detector running in simulation mode")
    
    def _setup_gpio(self) -> None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        logger.info(f"GPIO {self.gpio_pin} configured for motion detection")
    
    def on_motion(self, callback: Callable[[MotionEvent], None]) -> None:
        """Register callback to be called when motion is detected."""
        self._on_motion_callback = callback
    
    def _is_in_cooldown(self) -> bool:
        return (time.time() - self._last_motion_time) < self.cooldown_seconds
    
    def _handle_motion(self) -> None:
        if self._is_in_cooldown():
            return
        
        motion_start = time.time()
        self._last_motion_time = motion_start
        self.state = SensorState.MOTION_DETECTED
        
        logger.info("Motion detected!")
        
        if self._on_motion_callback:
            event = MotionEvent(
                timestamp=motion_start,
                duration=0  # Duration determined after motion stops
            )
            self._on_motion_callback(event)
        
        self.state = SensorState.COOLDOWN
    
    def check_motion(self) -> bool:
        """
        Check if motion is currently detected.
        Returns True if motion detected and not in cooldown.
        """
        if self.simulation_mode:
            return False
        
        if GPIO.input(self.gpio_pin) and not self._is_in_cooldown():
            self._handle_motion()
            return True
        return False
    
    def start_monitoring(self, poll_interval: float = 0.1) -> None:
        """
        Start continuous motion monitoring.
        
        Args:
            poll_interval: How often to check sensor (seconds)
        """
        self._running = True
        logger.info("Starting motion monitoring...")
        
        try:
            while self._running:
                self.check_motion()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Motion monitoring stopped by user")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop monitoring and cleanup GPIO."""
        self._running = False
        self.state = SensorState.IDLE
        if not self.simulation_mode:
            GPIO.cleanup(self.gpio_pin)
        logger.info("Motion detector stopped")
    
    def simulate_motion(self) -> None:
        """Simulate a motion event (for testing)."""
        if self.simulation_mode:
            logger.info("Simulating motion event")
            self._handle_motion()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = MotionDetector(simulation_mode=True)
    
    def on_motion_detected(event: MotionEvent):
        print(f"Motion at {event.timestamp}")
    
    detector.on_motion(on_motion_detected)
    detector.simulate_motion()
