"""
Capture module for wildlife monitoring.

Provides motion detection, camera control, and scheduled captures.
"""

from .motion_detector import MotionDetector, MotionEvent, SensorState
from .camera import Camera, CaptureReason, VideoMetadata
from .scheduler import CaptureScheduler
from .capture_service import CaptureService, CaptureServiceConfig

__all__ = [
    "MotionDetector",
    "MotionEvent", 
    "SensorState",
    "Camera",
    "CaptureReason",
    "VideoMetadata",
    "CaptureScheduler",
    "CaptureService",
    "CaptureServiceConfig",
]
