"""
Video file management for wildlife monitor.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_DIR = Path(__file__).parent.parent.parent / "data" / "videos"


@dataclass
class VideoInfo:
    """Information about a stored video file."""
    path: Path
    filename: str
    size_bytes: int
    created_at: datetime
    trigger_type: str  # Parsed from filename


class VideoStore:
    """
    Manages video file storage and cleanup.
    
    Features:
    - List and retrieve video files
    - Delete old videos based on retention policy
    - Calculate storage usage
    - Generate thumbnails (placeholder for future)
    """
    
    DEFAULT_RETENTION_DAYS = 5
    VIDEO_EXTENSIONS = {".mp4", ".h264", ".avi", ".mkv"}
    
    def __init__(
        self,
        video_dir: Optional[Path] = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ):
        self.video_dir = video_dir or DEFAULT_VIDEO_DIR
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        logger.info(f"VideoStore initialized: {self.video_dir}")
    
    def get_video_path(self, filename: str) -> Path:
        """Get full path for a video filename."""
        return self.video_dir / filename
    
    def video_exists(self, filename: str) -> bool:
        """Check if a video file exists."""
        return (self.video_dir / filename).exists()
    
    def get_video_info(self, filename: str) -> Optional[VideoInfo]:
        """Get information about a specific video."""
        path = self.video_dir / filename
        if not path.exists():
            return None
        
        return self._path_to_video_info(path)
    
    def list_videos(
        self,
        trigger_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[VideoInfo]:
        """
        List video files with optional filters.
        
        Args:
            trigger_type: Filter by trigger type ('motion', 'scheduled', 'manual')
            start_date: Filter videos created after this date
            end_date: Filter videos created before this date
            limit: Maximum number of results
            
        Returns:
            List of VideoInfo objects, sorted by creation time (newest first)
        """
        videos = []
        
        for path in self.video_dir.iterdir():
            if path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                continue
            
            info = self._path_to_video_info(path)
            if info is None:
                continue
            
            # Apply filters
            if trigger_type and info.trigger_type != trigger_type:
                continue
            
            if start_date and info.created_at < start_date:
                continue
            
            if end_date and info.created_at > end_date:
                continue
            
            videos.append(info)
        
        # Sort by creation time, newest first
        videos.sort(key=lambda v: v.created_at, reverse=True)
        
        return videos[:limit]
    
    def get_storage_usage(self) -> dict:
        """
        Calculate storage usage statistics.
        
        Returns:
            Dict with total_bytes, video_count, and breakdown by trigger_type
        """
        total_bytes = 0
        video_count = 0
        by_trigger_type = {}
        
        for path in self.video_dir.iterdir():
            if path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                continue
            
            info = self._path_to_video_info(path)
            if info is None:
                continue
            
            total_bytes += info.size_bytes
            video_count += 1
            
            if info.trigger_type not in by_trigger_type:
                by_trigger_type[info.trigger_type] = {"count": 0, "bytes": 0}
            by_trigger_type[info.trigger_type]["count"] += 1
            by_trigger_type[info.trigger_type]["bytes"] += info.size_bytes
        
        return {
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "video_count": video_count,
            "by_trigger_type": by_trigger_type,
        }
    
    def delete_video(self, filename: str) -> bool:
        """
        Delete a video file.
        
        Returns:
            True if deleted, False if not found.
        """
        path = self.video_dir / filename
        if path.exists():
            path.unlink()
            logger.info(f"Deleted video: {filename}")
            return True
        return False
    
    def cleanup_old_videos(self, retention_days: Optional[int] = None) -> int:
        """
        Delete videos older than retention period.
        
        Args:
            retention_days: Override default retention period
            
        Returns:
            Number of videos deleted.
        """
        retention = retention_days or self.retention_days
        cutoff = datetime.now() - timedelta(days=retention)
        deleted_count = 0
        
        for path in self.video_dir.iterdir():
            if path.suffix.lower() not in self.VIDEO_EXTENSIONS:
                continue
            
            info = self._path_to_video_info(path)
            if info and info.created_at < cutoff:
                path.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old video: {info.filename}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} videos older than {retention} days")
        
        return deleted_count
    
    def get_disk_free_space(self) -> dict:
        """Get free disk space information."""
        stat = shutil.disk_usage(self.video_dir)
        return {
            "total_bytes": stat.total,
            "used_bytes": stat.used,
            "free_bytes": stat.free,
            "free_gb": round(stat.free / (1024 ** 3), 2),
            "used_percent": round(stat.used / stat.total * 100, 1),
        }
    
    def _path_to_video_info(self, path: Path) -> Optional[VideoInfo]:
        """Convert a file path to VideoInfo."""
        try:
            stat = path.stat()
            trigger_type = self._parse_trigger_type(path.name)
            created_at = self._parse_timestamp(path.name) or datetime.fromtimestamp(stat.st_mtime)
            
            return VideoInfo(
                path=path,
                filename=path.name,
                size_bytes=stat.st_size,
                created_at=created_at,
                trigger_type=trigger_type,
            )
        except OSError as e:
            logger.warning(f"Could not read video info for {path}: {e}")
            return None
    
    def _parse_trigger_type(self, filename: str) -> str:
        """Parse trigger type from filename (e.g., 'motion_20240115_120000.mp4')."""
        name = filename.lower()
        if name.startswith("motion"):
            return "motion"
        elif name.startswith("scheduled"):
            return "scheduled"
        elif name.startswith("manual"):
            return "manual"
        return "unknown"
    
    def _parse_timestamp(self, filename: str) -> Optional[datetime]:
        """Parse timestamp from filename (e.g., 'motion_20240115_120000.mp4')."""
        try:
            # Extract date/time portion: type_YYYYMMDD_HHMMSS.ext
            parts = filename.rsplit(".", 1)[0].split("_")
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            pass
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    store = VideoStore()
    
    # List videos
    videos = store.list_videos(limit=10)
    print(f"Found {len(videos)} videos:")
    for video in videos:
        print(f"  - {video.filename} ({video.size_bytes} bytes, {video.trigger_type})")
    
    # Storage usage
    usage = store.get_storage_usage()
    print(f"\nStorage usage: {usage['total_mb']} MB, {usage['video_count']} videos")
    
    # Disk space
    disk = store.get_disk_free_space()
    print(f"Disk free: {disk['free_gb']} GB ({100 - disk['used_percent']:.1f}% free)")
