"""
Storage module for wildlife monitor.

Provides database operations and video file management.
"""


def __getattr__(name: str):
    """Lazy imports to avoid circular import issues."""
    if name in ("Database", "Detection", "DailySummary"):
        from .database import Database, Detection, DailySummary
        return {"Database": Database, "Detection": Detection, "DailySummary": DailySummary}[name]
    if name == "VideoStore":
        from .video_store import VideoStore
        return VideoStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Database",
    "Detection",
    "DailySummary",
    "VideoStore",
]
