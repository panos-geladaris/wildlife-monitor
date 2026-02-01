"""
Capture module for wildlife monitoring.

Provides motion detection, camera control, and scheduled captures.
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular import issues when running as __main__."""
    if name in ("MotionDetector", "MotionEvent", "SensorState"):
        from .motion_detector import MotionDetector, MotionEvent, SensorState
        return {"MotionDetector": MotionDetector, "MotionEvent": MotionEvent, "SensorState": SensorState}[name]
    if name in ("Camera", "CaptureReason", "VideoMetadata"):
        from .camera import Camera, CaptureReason, VideoMetadata
        return {"Camera": Camera, "CaptureReason": CaptureReason, "VideoMetadata": VideoMetadata}[name]
    if name == "CaptureScheduler":
        from .scheduler import CaptureScheduler
        return CaptureScheduler
    if name in ("CaptureService", "CaptureServiceConfig"):
        from .capture_service import CaptureService, CaptureServiceConfig
        return {"CaptureService": CaptureService, "CaptureServiceConfig": CaptureServiceConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
