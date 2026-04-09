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
                    "bird_species": d.bird_species,
                    "analyzed": d.analyzed,
                    "highlighted": d.highlighted,
                    "sound_class": d.sound_class,
                    "sound_species": d.sound_species,
                    "sound_confidence": d.sound_confidence,
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
            "bird_species": detection.bird_species,
            "analyzed": detection.analyzed,
            "highlighted": detection.highlighted,
            "sound_class": detection.sound_class,
            "sound_species": detection.sound_species,
            "sound_confidence": detection.sound_confidence,
            "created_at": detection.created_at.isoformat(),
        })
    except Exception as e:
        logger.error(f"Error getting detection {detection_id}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/detections/<int:detection_id>/highlight", methods=["POST"])
def toggle_highlight(detection_id: int):
    """Toggle the highlighted flag on a detection."""
    try:
        db = get_database()
        detection = db.get_detection(detection_id)

        if detection is None:
            return jsonify({"error": "Detection not found"}), 404

        new_state = not detection.highlighted
        db.set_highlighted(detection_id, new_state)

        return jsonify({
            "id": detection_id,
            "highlighted": new_state,
        })
    except Exception as e:
        logger.error(f"Error toggling highlight {detection_id}: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/highlights")
def list_highlights():
    """List highlighted detections, newest first."""
    try:
        db = get_database()
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        detections = db.get_highlighted_detections(limit=limit, offset=offset)

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
                    "bird_species": d.bird_species,
                    "analyzed": d.analyzed,
                    "highlighted": d.highlighted,
                    "sound_class": d.sound_class,
                    "sound_species": d.sound_species,
                    "sound_confidence": d.sound_confidence,
                }
                for d in detections
            ],
            "count": len(detections),
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        logger.error(f"Error listing highlights: {e}")
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
                    "bird_species": d.bird_species,
                    "sound_class": d.sound_class,
                    "sound_species": d.sound_species,
                    "sound_confidence": d.sound_confidence,
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


@api_bp.route("/detections/<int:detection_id>/reclassify", methods=["POST"])
def reclassify_detection(detection_id: int):
    """
    Re-run classification and object detection on an existing video.

    Returns:
        Updated detection record with new analysis results.
    """
    try:
        db = get_database()
        detection = db.get_detection(detection_id)

        if detection is None:
            return jsonify({"error": "Detection not found"}), 404

        video_path = Path(detection.video_path)
        if not video_path.exists():
            return jsonify({"error": "Video file not found"}), 404

        # Run classification
        from src.analysis import AnimalClassifier
        classifier = AnimalClassifier(
            model_name="mobilenet_v3_small",
            confidence_threshold=0.3,
        )
        classifier.load_model()
        result = classifier.classify_video(video_path)
        classifier.unload_model()

        db.update_detection(
            detection_id,
            animal_class=result.animal_class,
            confidence=result.confidence,
            bird_species=result.bird_species,
            analyzed=True,
        )

        # Re-run object detection
        try:
            from src.analysis import ObjectDetector
            detector = ObjectDetector(classify_crops=True)
            detector.load_model()
            frames_with_boxes = detector.detect_video_with_images(video_path)

            # Clear old frame objects and annotated frames
            db.delete_frame_objects(detection_id)
            video_dir = Path(current_app.config.get("VIDEO_DIR")).resolve()
            annotated_dir = video_dir.parent / "annotated" / str(detection_id)
            if annotated_dir.exists():
                import shutil
                shutil.rmtree(annotated_dir)

            all_boxes = []
            for _, boxes in frames_with_boxes:
                all_boxes.extend(boxes)

            if all_boxes:
                db.add_frame_objects(
                    detection_id,
                    [b.to_dict() for b in all_boxes],
                )
                from src.analysis.annotator import save_annotated_frames
                save_annotated_frames(
                    detection_id, frames_with_boxes, video_dir.parent / "annotated"
                )

                # If classifier found nothing, use best detector result
                current = db.get_detection(detection_id)
                if current and (not current.animal_class or current.animal_class == "unknown"):
                    best_box = max(
                        (b for b in all_boxes if b.animal_class),
                        key=lambda b: b.animal_confidence or 0,
                        default=None,
                    )
                    if best_box and best_box.animal_class:
                        db.update_detection(
                            detection_id,
                            animal_class=best_box.animal_class,
                            confidence=best_box.animal_confidence,
                            bird_species=best_box.bird_species,
                        )

            detector.unload_model()
        except ImportError:
            logger.warning("Object detection dependencies not available")
        except Exception as e:
            logger.warning(f"Object detection failed during reclassify: {e}")

        # Re-run audio classification if the original WAV is still on disk
        wav_path = video_path.with_suffix(".wav")
        if wav_path.exists():
            try:
                from src.analysis.audio_classifier import AudioClassifier

                monitor = current_app.config.get("MONITOR")
                audio_classifier = getattr(monitor, "_audio_classifier", None) if monitor else None

                if audio_classifier is None:
                    audio_classifier = AudioClassifier()

                current = db.get_detection(detection_id)
                visual_class = current.animal_class if current else None
                audio_result = audio_classifier.classify_audio(wav_path, visual_animal_class=visual_class)

                if audio_result.has_sound:
                    db.update_detection(
                        detection_id,
                        sound_class=audio_result.sound_class,
                        sound_species=audio_result.sound_species,
                        sound_confidence=audio_result.sound_confidence,
                    )
                    logger.info(
                        f"Reclassify audio: {audio_result.sound_class} "
                        f"({audio_result.sound_confidence:.1%})"
                    )
                else:
                    logger.info("Reclassify audio: no animal sound detected")
            except ImportError:
                logger.warning("Audio classification dependencies not available, skipping")
            except Exception as e:
                logger.warning(f"Audio classification failed during reclassify: {e}")
        else:
            logger.debug(f"No WAV file found alongside video, skipping audio reclassification")

        updated = db.get_detection(detection_id)
        return jsonify({
            "id": updated.id,
            "animal_class": updated.animal_class,
            "confidence": updated.confidence,
            "bird_species": updated.bird_species,
            "sound_class": updated.sound_class,
            "sound_species": updated.sound_species,
            "sound_confidence": updated.sound_confidence,
            "analyzed": updated.analyzed,
            "message": "Re-classification complete",
        })
    except ImportError:
        return jsonify({"error": "ML dependencies not available"}), 503
    except Exception as e:
        logger.error(f"Error reclassifying detection {detection_id}: {e}")
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


@api_bp.route("/environment")
def get_environment():
    """Read current values from all connected Breakout Garden sensors."""
    try:
        reader = current_app.config.get("SENSOR_READER")
        if reader is None:
            from src.sensors.reader import SensorReader
            reader = SensorReader()
            current_app.config["SENSOR_READER"] = reader

        data = reader.read_all()
        data["available_sensors"] = reader.available_sensors
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error reading sensors: {e}")
        return jsonify({"error": str(e)}), 500
