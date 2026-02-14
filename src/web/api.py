"""
REST API endpoints for wildlife monitor.
"""

import logging
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def get_database():
    """Get database instance."""
    from src.storage.database import Database
    db_path = current_app.config.get("DB_PATH")
    return Database(db_path)


def get_video_store():
    """Get video store instance."""
    from src.storage.video_store import VideoStore
    video_dir = current_app.config.get("VIDEO_DIR")
    return VideoStore(video_dir)


def _cleanup_detection_artifacts(detection, detection_id: int) -> None:
    """Remove thumbnail and annotated frames associated with a detection."""
    video_dir = Path(current_app.config.get("VIDEO_DIR")).resolve()

    thumb_path = video_dir / "thumbnails" / (Path(detection.video_path).stem + ".jpg")
    if thumb_path.exists():
        thumb_path.unlink()

    annotated_dir = video_dir.parent / "annotated" / str(detection_id)
    if annotated_dir.exists():
        shutil.rmtree(annotated_dir)


@api_bp.route("/status")
def get_status():
    """
    Get system status.
    
    Returns:
        System status including uptime, storage, and recent activity.
    """
    try:
        video_store = get_video_store()
        db = get_database()
        
        storage = video_store.get_storage_usage()
        disk = video_store.get_disk_free_space()
        detection_count = db.get_detection_count()
        
        # Get recent detections (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = db.get_detection_count(start_date=one_hour_ago)
        
        return jsonify({
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "storage": {
                "videos_mb": storage["total_mb"],
                "video_count": storage["video_count"],
                "disk_free_gb": disk["free_gb"],
                "disk_used_percent": disk["used_percent"],
            },
            "detections": {
                "total": detection_count,
                "last_hour": recent_count,
            },
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detections")
def list_detections():
    """
    List detections with optional filters.
    
    Query params:
        trigger_type: Filter by trigger type (motion, scheduled, manual)
        animal_class: Filter by animal class
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
        analyzed: Filter by analyzed status (true/false)
        limit: Maximum results (default 50)
        offset: Pagination offset (default 0)
    
    Returns:
        List of detection records.
    """
    try:
        db = get_database()
        
        # Parse query parameters
        trigger_type = request.args.get("trigger_type")
        animal_class = request.args.get("animal_class")
        analyzed = request.args.get("analyzed")
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        
        start_date = None
        end_date = None
        
        if request.args.get("start_date"):
            start_date = datetime.fromisoformat(request.args.get("start_date"))
        if request.args.get("end_date"):
            end_date = datetime.fromisoformat(request.args.get("end_date"))
        
        analyzed_bool = None
        if analyzed is not None:
            analyzed_bool = analyzed.lower() == "true"
        
        detections = db.get_detections(
            trigger_type=trigger_type,
            animal_class=animal_class,
            start_date=start_date,
            end_date=end_date,
            analyzed=analyzed_bool,
            limit=limit,
            offset=offset,
        )
        
        return jsonify({
            "detections": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "video_path": d.video_path,
                    "video_filename": Path(d.video_path).name,
                    "trigger_type": d.trigger_type,
                    "animal_class": d.animal_class,
                    "confidence": d.confidence,
                    "analyzed": d.analyzed,
                }
                for d in detections
            ],
            "count": len(detections),
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        logger.error(f"Error listing detections: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detections/<int:detection_id>")
def get_detection(detection_id: int):
    """
    Get a single detection by ID.
    
    Returns:
        Detection record or 404 if not found.
    """
    try:
        db = get_database()
        detection = db.get_detection(detection_id)
        
        if detection is None:
            return jsonify({"error": "Detection not found"}), 404
        
        return jsonify({
            "id": detection.id,
            "timestamp": detection.timestamp.isoformat(),
            "video_path": detection.video_path,
            "video_filename": Path(detection.video_path).name,
            "trigger_type": detection.trigger_type,
            "animal_class": detection.animal_class,
            "confidence": detection.confidence,
            "analyzed": detection.analyzed,
            "created_at": detection.created_at.isoformat(),
        })
    except Exception as e:
        logger.error(f"Error getting detection {detection_id}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detections/<int:detection_id>/objects")
def get_detection_objects(detection_id: int):
    """
    Get bounding box detections for a video's analyzed frames.
    
    Returns:
        List of detected objects grouped by frame number.
    """
    try:
        db = get_database()
        detection = db.get_detection(detection_id)
        
        if detection is None:
            return jsonify({"error": "Detection not found"}), 404
        
        frame_objects = db.get_frame_objects(detection_id)
        
        frames: dict[int, list] = {}
        for obj in frame_objects:
            fn = obj["frame_number"]
            if fn not in frames:
                frames[fn] = []
            frames[fn].append(obj)
        
        return jsonify({
            "detection_id": detection_id,
            "frames": {str(k): v for k, v in sorted(frames.items())},
            "total_objects": len(frame_objects),
        })
    except Exception as e:
        logger.error(f"Error getting detection objects {detection_id}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/videos")
def list_videos():
    """
    List video files.
    
    Query params:
        trigger_type: Filter by trigger type
        limit: Maximum results (default 50)
    
    Returns:
        List of video files with metadata.
    """
    try:
        video_store = get_video_store()
        
        trigger_type = request.args.get("trigger_type")
        limit = int(request.args.get("limit", 50))
        
        videos = video_store.list_videos(
            trigger_type=trigger_type,
            limit=limit,
        )
        
        return jsonify({
            "videos": [
                {
                    "filename": v.filename,
                    "size_bytes": v.size_bytes,
                    "created_at": v.created_at.isoformat(),
                    "trigger_type": v.trigger_type,
                    "url": f"/videos/{v.filename}",
                }
                for v in videos
            ],
            "count": len(videos),
        })
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/stats/daily")
def get_daily_stats():
    """
    Get daily detection counts.
    
    Query params:
        days: Number of days to include (default 7)
    
    Returns:
        Daily detection counts.
    """
    try:
        db = get_database()
        days = int(request.args.get("days", 7))
        
        stats = []
        today = date.today()
        
        for i in range(days):
            day = today - timedelta(days=i)
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())
            
            total = db.get_detection_count(start_date=start, end_date=end)
            motion = db.get_detection_count(trigger_type="motion", start_date=start, end_date=end)
            scheduled = db.get_detection_count(trigger_type="scheduled", start_date=start, end_date=end)
            
            stats.append({
                "date": day.isoformat(),
                "total": total,
                "motion": motion,
                "scheduled": scheduled,
            })
        
        return jsonify({
            "stats": stats,
            "days": days,
        })
    except Exception as e:
        logger.error(f"Error getting daily stats: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/stats/animals")
def get_animal_stats():
    """
    Get animal detection breakdown.
    
    Query params:
        days: Number of days to include (default 30)
    
    Returns:
        Animal class counts.
    """
    try:
        db = get_database()
        days = int(request.args.get("days", 30))
        
        start_date = datetime.now() - timedelta(days=days)
        detections = db.get_detections(
            start_date=start_date,
            analyzed=True,
            limit=10000,
        )
        
        # Count by animal class
        animal_counts = {}
        for d in detections:
            if d.animal_class:
                animal_counts[d.animal_class] = animal_counts.get(d.animal_class, 0) + 1
        
        # Sort by count
        sorted_animals = sorted(
            animal_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return jsonify({
            "animals": [
                {"animal_class": animal, "count": count}
                for animal, count in sorted_animals
            ],
            "total_analyzed": len(detections),
            "days": days,
        })
    except Exception as e:
        logger.error(f"Error getting animal stats: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/stats/summary")
def get_summary():
    """
    Get summary statistics for dashboard.
    
    Returns:
        Summary of today's detections and recent activity.
    """
    try:
        db = get_database()
        
        today = date.today()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())
        
        # Today's stats
        today_total = db.get_detection_count(start_date=today_start, end_date=today_end)
        today_motion = db.get_detection_count(
            trigger_type="motion",
            start_date=today_start,
            end_date=today_end,
        )
        
        # Get today's animal breakdown
        today_detections = db.get_detections(
            start_date=today_start,
            end_date=today_end,
            analyzed=True,
            limit=1000,
        )
        
        animal_counts = {}
        for d in today_detections:
            if d.animal_class and d.animal_class != "unknown":
                animal_counts[d.animal_class] = animal_counts.get(d.animal_class, 0) + 1
        
        # Recent detections
        recent = db.get_detections(limit=5)
        
        return jsonify({
            "today": {
                "total": today_total,
                "motion": today_motion,
                "animals": animal_counts,
            },
            "recent_detections": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "trigger_type": d.trigger_type,
                    "animal_class": d.animal_class,
                    "confidence": d.confidence,
                }
                for d in recent
            ],
        })
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detections/<int:detection_id>", methods=["DELETE"])
def delete_detection(detection_id: int):
    """
    Delete a single detection and its video file.
    
    Returns:
        Success message or 404 if not found.
    """
    try:
        db = get_database()
        detection = db.get_detection(detection_id)
        
        if detection is None:
            return jsonify({"error": "Detection not found"}), 404
        
        db.delete_detection(detection_id)
        
        video_path = Path(detection.video_path)
        video_deleted = False
        if video_path.exists():
            video_path.unlink()
            video_deleted = True
            logger.info(f"Deleted video file: {video_path}")
        
        _cleanup_detection_artifacts(detection, detection_id)
        
        return jsonify({
            "message": "Detection deleted",
            "id": detection_id,
            "video_deleted": video_deleted,
        })
    except Exception as e:
        logger.error(f"Error deleting detection {detection_id}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detections/bulk-delete", methods=["POST"])
def bulk_delete_detections():
    """
    Bulk delete detections and their video files.
    
    Request body:
        {"ids": [1, 2, 3]}
    
    Returns:
        Summary of deleted records and video files.
    """
    try:
        data = request.get_json()
        if not data or "ids" not in data:
            return jsonify({"error": "Missing 'ids' in request body"}), 400
        
        detection_ids = data["ids"]
        if not isinstance(detection_ids, list) or not detection_ids:
            return jsonify({"error": "'ids' must be a non-empty list"}), 400
        
        db = get_database()
        
        videos_deleted = 0
        for detection_id in detection_ids:
            detection = db.get_detection(detection_id)
            if detection:
                video_path = Path(detection.video_path)
                if video_path.exists():
                    video_path.unlink()
                    videos_deleted += 1
                _cleanup_detection_artifacts(detection, detection_id)
        
        records_deleted = db.delete_detections_bulk(detection_ids)
        
        logger.info(f"Bulk deleted {records_deleted} detections, {videos_deleted} videos")
        
        return jsonify({
            "message": f"Deleted {records_deleted} detections",
            "records_deleted": records_deleted,
            "videos_deleted": videos_deleted,
        })
    except Exception as e:
        logger.error(f"Error bulk deleting detections: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/test-capture", methods=["POST"])
def test_capture():
    """
    Trigger an on-demand 4-second test capture with full analysis.

    Requires the MONITOR instance to be available (not web-only mode).

    Returns:
        Detection ID and analysis results.
    """
    TEST_CAPTURE_DURATION = 4.0

    monitor = current_app.config.get("MONITOR")
    if monitor is None or getattr(monitor, "_capture_service", None) is None:
        return jsonify({"error": "Capture service not available (web-only mode)"}), 503

    try:
        metadata = monitor._capture_service.trigger_manual_capture(
            duration=TEST_CAPTURE_DURATION
        )

        db = get_database()
        detections = db.get_detections(limit=1)
        if detections:
            d = detections[0]
            return jsonify({
                "detection_id": d.id,
                "video_filename": Path(d.video_path).name,
                "animal_class": d.animal_class,
                "confidence": d.confidence,
                "message": "Test capture complete",
            })

        return jsonify({"error": "Capture completed but detection not found"}), 500
    except Exception as e:
        logger.error(f"Test capture failed: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/timelapses")
def list_timelapses():
    """List timelapses, newest first."""
    try:
        db = get_database()
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        timelapses = db.get_timelapses(limit=limit, offset=offset)
        return jsonify({
            "timelapses": [
                {
                    "id": t.id,
                    "date": t.date.isoformat(),
                    "video_path": t.video_path,
                    "video_filename": Path(t.video_path).name,
                    "detection_count": t.detection_count,
                    "animal_counts": t.animal_counts,
                    "created_at": t.created_at.isoformat(),
                }
                for t in timelapses
            ],
            "count": len(timelapses),
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        logger.error(f"Error listing timelapses: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/timelapses/<int:timelapse_id>")
def get_timelapse(timelapse_id: int):
    """Get a single timelapse by ID."""
    try:
        db = get_database()
        timelapse = db.get_timelapse(timelapse_id)
        if timelapse is None:
            return jsonify({"error": "Timelapse not found"}), 404
        return jsonify({
            "id": timelapse.id,
            "date": timelapse.date.isoformat(),
            "video_path": timelapse.video_path,
            "video_filename": Path(timelapse.video_path).name,
            "detection_count": timelapse.detection_count,
            "animal_counts": timelapse.animal_counts,
            "created_at": timelapse.created_at.isoformat(),
        })
    except Exception as e:
        logger.error(f"Error getting timelapse {timelapse_id}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/timelapses/<int:timelapse_id>", methods=["DELETE"])
def delete_timelapse(timelapse_id: int):
    """Delete a timelapse and its video file."""
    try:
        db = get_database()
        timelapse = db.get_timelapse(timelapse_id)
        if timelapse is None:
            return jsonify({"error": "Timelapse not found"}), 404

        video_path = Path(timelapse.video_path)
        video_deleted = False
        if video_path.exists():
            video_path.unlink()
            video_deleted = True

        thumb_path = video_path.parent / "thumbnails" / (video_path.stem + ".jpg")
        if thumb_path.exists():
            thumb_path.unlink()

        db.delete_timelapse(timelapse_id)

        return jsonify({
            "message": "Timelapse deleted",
            "id": timelapse_id,
            "video_deleted": video_deleted,
        })
    except Exception as e:
        logger.error(f"Error deleting timelapse {timelapse_id}: {e}")
        return jsonify({"error": str(e)}), 500
