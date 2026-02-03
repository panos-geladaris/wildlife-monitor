"""
Web UI module for wildlife monitor.

Provides Flask-based web interface for viewing detections and statistics.
"""


def __getattr__(name: str):
    """Lazy imports."""
    if name == "create_app":
        from .app import create_app
        return create_app
    if name == "api":
        from . import api
        return api
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_app",
    "api",
]
