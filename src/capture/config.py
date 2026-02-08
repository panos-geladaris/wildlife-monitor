"""
Configuration loader for wildlife monitor.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


@dataclass
class CaptureServiceConfig:
    """Configuration for the capture service."""
    video_output_dir: Path
    gpio_pin: int = 17
    video_duration: float = 2.5
    scheduled_video_duration: float = 15.0
    resolution: tuple[int, int] = (1280, 720)
    framerate: int = 30
    cooldown_seconds: float = 5.0
    hourly_capture_minute: int = 0
    hourly_capture_enabled: bool = True
    simulation_mode: bool = False
    zoom_level: float = 1.0
    autofocus: bool = True
    interval_capture_enabled: bool = True
    interval_capture_minutes: int = 15
    interval_capture_duration: float = 7.0
    interval_capture_zoom_levels: tuple[float, ...] = (1.0, 2.0)
    daylight_enabled: bool = False
    daylight_lat: float = 51.5074
    daylight_lng: float = -0.1278
    daylight_tzid: str = "Europe/London"
    daylight_start_offset_minutes: int = 0
    daylight_end_offset_minutes: int = 0
    daylight_fallback: str = "allow"


def load_config(config_path: Optional[Path] = None) -> CaptureServiceConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (uses default if None)
        
    Returns:
        CaptureServiceConfig populated from file
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return CaptureServiceConfig(
            video_output_dir=Path(__file__).parent.parent.parent / "data" / "videos"
        )
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    
    motion = data.get("motion", {})
    hourly = data.get("hourly", {})
    interval = data.get("interval", {})
    camera = data.get("camera", {})
    hardware = data.get("hardware", {})
    output = data.get("output", {})
    daylight = data.get("daylight", {})
    
    resolution = camera.get("resolution", [1280, 720])
    zoom_levels = interval.get("zoom_levels", [1.0, 2.0])
    
    video_dir = output.get("video_dir", "data/videos")
    if not Path(video_dir).is_absolute():
        video_dir = Path(__file__).parent.parent.parent / video_dir
    
    return CaptureServiceConfig(
        video_output_dir=Path(video_dir),
        gpio_pin=hardware.get("gpio_pin", 17),
        video_duration=motion.get("video_duration", 2.5),
        cooldown_seconds=motion.get("cooldown_seconds", 5.0),
        scheduled_video_duration=hourly.get("video_duration", 15.0),
        hourly_capture_enabled=hourly.get("enabled", True),
        hourly_capture_minute=hourly.get("minute", 0),
        resolution=tuple(resolution),
        framerate=camera.get("framerate", 30),
        zoom_level=camera.get("zoom_level", 1.0),
        autofocus=camera.get("autofocus", True),
        interval_capture_enabled=interval.get("enabled", True),
        interval_capture_minutes=interval.get("minutes", 15),
        interval_capture_duration=interval.get("video_duration", 7.0),
        interval_capture_zoom_levels=tuple(zoom_levels),
        daylight_enabled=daylight.get("enabled", False),
        daylight_lat=daylight.get("lat", 51.5074),
        daylight_lng=daylight.get("lng", -0.1278),
        daylight_tzid=daylight.get("tzid", "Europe/London"),
        daylight_start_offset_minutes=daylight.get("start_offset_minutes", 0),
        daylight_end_offset_minutes=daylight.get("end_offset_minutes", 0),
        daylight_fallback=daylight.get("fallback", "allow"),
    )
