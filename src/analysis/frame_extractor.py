"""
Frame extraction from video files for analysis.
"""

import logging
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)

# Try to import cv2, fall back to PIL-based extraction
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - using PIL fallback (limited functionality)")


@dataclass
class ExtractedFrame:
    """Represents an extracted video frame."""
    image: Image.Image
    frame_number: int
    timestamp_seconds: float


class FrameExtractor:
    """
    Extracts frames from video files for ML analysis.
    
    Supports:
    - Extract single frame at specific time
    - Extract multiple frames at regular intervals
    - Extract key frames for analysis
    """
    
    DEFAULT_FRAMES_PER_VIDEO = 5
    
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._duration: float = 0.0
        
        if CV2_AVAILABLE:
            self._init_video()
    
    def _init_video(self) -> None:
        """Initialize video capture and get metadata."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0.0
        
        logger.debug(
            f"Video opened: {self.video_path.name}, "
            f"{self._frame_count} frames, {self._fps:.1f} fps, {self._duration:.2f}s"
        )
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def duration(self) -> float:
        return self._duration
    
    def extract_frame_at_time(self, time_seconds: float) -> Optional[ExtractedFrame]:
        """
        Extract a single frame at a specific time.
        
        Args:
            time_seconds: Time position in seconds
            
        Returns:
            ExtractedFrame or None if extraction fails
        """
        if not CV2_AVAILABLE or self._cap is None:
            logger.warning("OpenCV not available for frame extraction")
            return None
        
        frame_number = int(time_seconds * self._fps)
        return self.extract_frame_at_number(frame_number)
    
    def extract_frame_at_number(self, frame_number: int) -> Optional[ExtractedFrame]:
        """
        Extract a specific frame by number.
        
        Args:
            frame_number: Frame index (0-based)
            
        Returns:
            ExtractedFrame or None if extraction fails
        """
        if not CV2_AVAILABLE or self._cap is None:
            logger.warning("OpenCV not available for frame extraction")
            return None
        
        frame_number = max(0, min(frame_number, self._frame_count - 1))
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            logger.warning(f"Failed to extract frame {frame_number}")
            return None
        
        # Convert BGR to RGB and create PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        timestamp = frame_number / self._fps if self._fps > 0 else 0.0
        
        return ExtractedFrame(
            image=image,
            frame_number=frame_number,
            timestamp_seconds=timestamp,
        )
    
    def extract_frames(
        self,
        num_frames: int = DEFAULT_FRAMES_PER_VIDEO,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[ExtractedFrame, None, None]:
        """
        Extract multiple frames at regular intervals.
        
        Args:
            num_frames: Number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds (defaults to video end)
            
        Yields:
            ExtractedFrame objects
        """
        if not CV2_AVAILABLE or self._cap is None:
            logger.warning("OpenCV not available for frame extraction")
            return
        
        end_time = end_time or self._duration
        duration = end_time - start_time
        
        if duration <= 0 or num_frames <= 0:
            return
        
        # Calculate frame positions
        if num_frames == 1:
            positions = [start_time + duration / 2]
        else:
            interval = duration / (num_frames - 1)
            positions = [start_time + i * interval for i in range(num_frames)]
        
        for time_pos in positions:
            frame = self.extract_frame_at_time(time_pos)
            if frame:
                yield frame
    
    def extract_key_frames(
        self,
        num_frames: int = DEFAULT_FRAMES_PER_VIDEO,
    ) -> list[ExtractedFrame]:
        """
        Extract key frames spread across the video.
        
        This is the main method for getting frames for ML analysis.
        Frames are extracted at regular intervals to capture different
        moments in the video.
        
        Args:
            num_frames: Number of frames to extract
            
        Returns:
            List of ExtractedFrame objects
        """
        return list(self.extract_frames(num_frames=num_frames))
    
    def extract_middle_frame(self) -> Optional[ExtractedFrame]:
        """Extract the middle frame of the video."""
        return self.extract_frame_at_time(self._duration / 2)
    
    def close(self) -> None:
        """Release video resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.debug(f"Video closed: {self.video_path.name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.frame_extractor <video_path>")
        print("\nThis module requires a video file to test frame extraction.")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    with FrameExtractor(video_path) as extractor:
        print(f"Video: {video_path.name}")
        print(f"Duration: {extractor.duration:.2f}s")
        print(f"FPS: {extractor.fps:.1f}")
        print(f"Frame count: {extractor.frame_count}")
        
        # Extract key frames
        frames = extractor.extract_key_frames(num_frames=3)
        print(f"\nExtracted {len(frames)} frames:")
        for frame in frames:
            print(f"  Frame {frame.frame_number} at {frame.timestamp_seconds:.2f}s")
            print(f"    Size: {frame.image.size}")
