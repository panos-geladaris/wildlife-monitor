"""
Scheduler for periodic video captures.
"""

import logging
from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class CaptureScheduler:
    """
    Manages scheduled video captures using APScheduler.
    
    Supports:
    - Hourly captures (default)
    - Custom interval captures
    - Cron-based scheduling
    """
    
    def __init__(self):
        self._scheduler = BackgroundScheduler()
        self._capture_callback: Optional[Callable[[], None]] = None
        self._job_id = "scheduled_capture"
    
    def set_capture_callback(self, callback: Callable[[], None]) -> None:
        """Set the function to call for each scheduled capture."""
        self._capture_callback = callback
    
    def _execute_capture(self) -> None:
        if self._capture_callback:
            logger.info(f"Executing scheduled capture at {datetime.now()}")
            try:
                self._capture_callback()
            except Exception as e:
                logger.error(f"Scheduled capture failed: {e}")
        else:
            logger.warning("No capture callback configured")
    
    def schedule_hourly(self, minute: int = 0) -> None:
        """
        Schedule captures every hour at a specific minute.
        
        Args:
            minute: Minute of each hour to capture (0-59)
        """
        self._remove_existing_job()
        
        trigger = CronTrigger(minute=minute)
        self._scheduler.add_job(
            self._execute_capture,
            trigger=trigger,
            id=self._job_id,
            name="Hourly video capture"
        )
        logger.info(f"Scheduled hourly captures at minute {minute}")
    
    def schedule_interval(self, hours: int = 0, minutes: int = 0, seconds: int = 0) -> None:
        """
        Schedule captures at a fixed interval.
        
        Args:
            hours: Hours between captures
            minutes: Minutes between captures  
            seconds: Seconds between captures
        """
        if hours == 0 and minutes == 0 and seconds == 0:
            raise ValueError("At least one time unit must be non-zero")
        
        self._remove_existing_job()
        
        trigger = IntervalTrigger(hours=hours, minutes=minutes, seconds=seconds)
        self._scheduler.add_job(
            self._execute_capture,
            trigger=trigger,
            id=self._job_id,
            name="Interval video capture"
        )
        logger.info(f"Scheduled interval captures: {hours}h {minutes}m {seconds}s")
    
    def _remove_existing_job(self) -> None:
        try:
            self._scheduler.remove_job(self._job_id)
        except Exception:
            pass  # Job doesn't exist yet
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("Capture scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Capture scheduler stopped")
    
    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled capture time."""
        job = self._scheduler.get_job(self._job_id)
        if job:
            return job.next_run_time
        return None
    
    @property
    def is_running(self) -> bool:
        return self._scheduler.running


if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    scheduler = CaptureScheduler()
    
    capture_count = 0
    def mock_capture():
        global capture_count
        capture_count += 1
        print(f"Capture #{capture_count} at {datetime.now()}")
    
    scheduler.set_capture_callback(mock_capture)
    scheduler.schedule_interval(seconds=5)  # Every 5 seconds for testing
    scheduler.start()
    
    print(f"Next capture at: {scheduler.get_next_run_time()}")
    
    try:
        time.sleep(12)
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.stop()
        print(f"Total captures: {capture_count}")
