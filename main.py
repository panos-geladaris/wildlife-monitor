#!/usr/bin/env python3
"""
Wildlife Monitor - Main Entry Point

Integrates all modules:
- Capture: Motion detection and scheduled video recording
- Analysis: ML-based animal classification
- Storage: Database and video file management
- Web: Flask UI for viewing results

Usage:
    python main.py                    # Run full system
    python main.py --capture-only     # Run capture without web UI
    python main.py --web-only         # Run web UI only
    python main.py --analyze <path>   # Analyze a specific video
"""

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WildlifeMonitor:
    """
    Main application that integrates all wildlife monitoring components.
    
    Workflow:
    1. Capture service detects motion or runs scheduled captures
    2. New videos trigger the analysis pipeline
    3. Analysis results are stored in the database
    4. Web UI displays detections and statistics
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        video_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        simulation_mode: Optional[bool] = None,
        web_port: int = 5001,
        enable_analysis: bool = True,
        enable_detection: bool = True,
        enable_timelapse: bool = True,
        enable_cleanup: bool = True,
    ):
        self.config_path = config_path
        self.video_dir = video_dir or Path("data/videos")
        self.db_path = db_path or Path("data/wildlife.db")
        self.web_port = web_port
        self.enable_analysis = enable_analysis
        self.enable_detection = enable_detection
        self.enable_timelapse = enable_timelapse
        self.enable_cleanup = enable_cleanup
        
        self._capture_service = None
        self._classifier = None
        self._detector = None
        self._audio_classifier = None
        self._database = None
        self._cleanup_scheduler = None
        self._web_app = None
        self._web_thread = None
        self._cleanup_scheduler = None
        self._timelapse_scheduler = None
        self._running = False
        self._simulation_mode = simulation_mode
        self._config_data = self._load_config_data()
        
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_config_data(self) -> dict:
        """Load raw config data from YAML file."""
        import yaml
        config_file = self.config_path or Path("config.yaml")
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _detect_simulation_mode(self) -> bool:
        """Auto-detect if we should run in simulation mode."""
        if self._simulation_mode is not None:
            return self._simulation_mode
        
        env_simulate = os.environ.get("SIMULATE", "auto")
        if env_simulate != "auto":
            return env_simulate == "1"
        
        try:
            import RPi.GPIO
            return False
        except ImportError:
            return True
    
    def _init_database(self) -> None:
        """Initialize database connection."""
        from src.storage import Database
        self._database = Database(self.db_path)
        logger.info(f"Database initialized: {self.db_path}")
    
    def _init_classifier(self) -> None:
        """Initialize the ML classifier."""
        if not self.enable_analysis:
            logger.info("Analysis disabled, skipping classifier initialization")
            return
        
        try:
            from src.analysis import AnimalClassifier
            self._classifier = AnimalClassifier(
                model_name="mobilenet_v3_small",
                confidence_threshold=0.3,
            )
            self._classifier.load_model()
            logger.info("Animal classifier initialized")
        except ImportError as e:
            logger.warning(f"ML dependencies not available: {e}")
            logger.warning("Running without animal classification")
            self._classifier = None
        except Exception as e:
            logger.warning(f"Failed to initialize classifier: {e}")
            logger.warning("Running without animal classification")
            self._classifier = None
    
    def _init_detector(self) -> None:
        """Initialize the object detector."""
        if not self.enable_detection:
            logger.info("Object detection disabled, skipping detector initialization")
            return
        
        detection_cfg = self._config_data.get("detection", {})
        if detection_cfg.get("enabled") is False:
            logger.info("Object detection disabled in config, skipping")
            return
        
        try:
            from src.analysis import ObjectDetector
            self._detector = ObjectDetector(
                score_threshold=detection_cfg.get("score_threshold", 0.3),
                max_boxes_per_frame=detection_cfg.get("max_boxes_per_frame", 5),
                classify_crops=True,
            )
            self._detector.load_model()
            logger.info("Object detector initialized")
        except ImportError as e:
            logger.warning(f"Detection dependencies not available: {e}")
            self._detector = None
        except Exception as e:
            logger.warning(f"Failed to initialize detector: {e}")
            self._detector = None
    
    def _init_audio_classifier(self) -> None:
        """Initialize the audio sound classifier if enabled in config."""
        audio_analysis_cfg = self._config_data.get("audio_analysis", {})
        if not audio_analysis_cfg.get("enabled", False):
            logger.info("Audio analysis disabled, skipping audio classifier")
            return

        try:
            from src.analysis.audio_classifier import AudioClassifier
            daylight_cfg = self._config_data.get("daylight", {})
            self._audio_classifier = AudioClassifier(
                panns_min_confidence=audio_analysis_cfg.get("panns_min_confidence", 0.3),
                birdnet_min_confidence=audio_analysis_cfg.get("birdnet_min_confidence", 0.5),
                panns_enabled=audio_analysis_cfg.get("panns_enabled", True),
                birdnet_enabled=audio_analysis_cfg.get("birdnet_enabled", True),
                lat=daylight_cfg.get("lat"),
                lng=daylight_cfg.get("lng"),
            )
            logger.info("Audio classifier initialized (PANNs + BirdNET)")
        except ImportError as e:
            logger.warning(f"Audio classification dependencies not available: {e}")
            self._audio_classifier = None
        except Exception as e:
            logger.warning(f"Failed to initialize audio classifier: {e}")
            self._audio_classifier = None

    def _init_audio_recorder(self):
        """Initialize the audio recorder if enabled in config."""
        audio_cfg = self._config_data.get("audio", {})
        if not audio_cfg.get("enabled", False):
            return None

        if self._detect_simulation_mode():
            logger.info("Audio recording skipped in simulation mode")
            return None

        from src.capture.audio_recorder import AudioRecorder
        recorder = AudioRecorder(
            device=audio_cfg.get("device", AudioRecorder.DEFAULT_DEVICE),
            sample_rate=audio_cfg.get("sample_rate", AudioRecorder.DEFAULT_SAMPLE_RATE),
        )
        logger.info(f"Audio recorder initialized (device={recorder.device})")
        return recorder

    def _init_capture_service(self) -> None:
        """Initialize the capture service."""
        from src.capture import CaptureService, load_config
        
        config = load_config(self.config_path)
        config.video_output_dir = self.video_dir
        config.simulation_mode = self._detect_simulation_mode()
        
        audio_recorder = self._init_audio_recorder()
        self._capture_service = CaptureService(config, audio_recorder=audio_recorder)
        self._capture_service.on_capture(self._on_video_captured)
        
        logger.info(f"Capture service initialized (simulation={config.simulation_mode})")
    
    def _init_cleanup_scheduler(self) -> None:
        """Initialize scheduled cleanup of empty detections."""
        if not self.enable_cleanup:
            logger.info("Cleanup disabled via CLI flag, skipping")
            return

        cleanup_cfg = self._config_data.get("cleanup", {})
        if cleanup_cfg.get("enabled") is False:
            logger.info("Cleanup disabled in config, skipping")
            return

        max_age_hours = cleanup_cfg.get("max_age_hours", 24)
        interval_hours = cleanup_cfg.get("interval_hours", 6)

        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from src.storage.cleanup import run_cleanup

        self._cleanup_scheduler = BackgroundScheduler()
        self._cleanup_scheduler.add_job(
            run_cleanup,
            trigger=IntervalTrigger(hours=interval_hours),
            args=[self._database, self.video_dir, max_age_hours],
            id="empty_detection_cleanup",
            name="Empty detection cleanup",
        )
        self._cleanup_scheduler.start()

        logger.info(
            f"Cleanup scheduler started: max_age={max_age_hours}h, "
            f"interval={interval_hours}h"
        )

    def _on_video_captured(self, metadata) -> None:
        """Handle new video capture - run analysis in a background thread."""
        thread = threading.Thread(
            target=self._process_capture,
            args=(metadata,),
            daemon=True,
        )
        thread.start()

    def _process_capture(self, metadata) -> None:
        """Analyze a captured video and store results in the database."""
        from src.storage import Detection
        
        logger.info(f"New capture: {metadata.filepath.name} ({metadata.reason.value})")
        
        detection = Detection(
            timestamp=metadata.timestamp,
            video_path=str(metadata.filepath),
            trigger_type=metadata.reason.value,
        )
        detection_id = self._database.add_detection(detection)
        
        try:
            from src.storage.thumbnail import generate_thumbnail
            generate_thumbnail(metadata.filepath)
        except Exception as e:
            logger.warning(f"Thumbnail generation failed: {e}")
        
        if self._classifier and metadata.filepath.exists():
            try:
                result = self._classifier.classify_video(metadata.filepath)
                
                self._database.update_detection(
                    detection_id,
                    animal_class=result.animal_class,
                    confidence=result.confidence,
                    bird_species=result.bird_species,
                    analyzed=True,
                )
                
                if result.is_animal:
                    species_info = f" ({result.bird_species})" if result.bird_species else ""
                    logger.info(
                        f"Detected: {result.animal_class}{species_info} "
                        f"({result.confidence:.1%})"
                    )
                else:
                    logger.info("No animal detected in video")
                    
            except Exception as e:
                logger.error(f"Analysis failed for {metadata.filepath}: {e}")
                self._database.update_detection(detection_id, analyzed=True)
        
        if self._detector and metadata.filepath.exists():
            try:
                detection_cfg = self._config_data.get("detection", {})
                num_frames = detection_cfg.get("num_frames", 5)
                frames_with_boxes = self._detector.detect_video_with_images(
                    metadata.filepath, num_frames=num_frames
                )
                
                all_boxes = []
                for _, boxes in frames_with_boxes:
                    all_boxes.extend(boxes)
                
                if all_boxes:
                    self._database.add_frame_objects(
                        detection_id,
                        [b.to_dict() for b in all_boxes],
                    )
                    
                    from src.analysis.annotator import save_annotated_frames
                    annotated_dir = self.video_dir.parent / "annotated"
                    save_annotated_frames(
                        detection_id, frames_with_boxes, annotated_dir
                    )
                    
                    logger.info(
                        f"Detected {len(all_boxes)} objects across "
                        f"{len(frames_with_boxes)} frames"
                    )
                    
                    current = self._database.get_detection(detection_id)
                    if current and (not current.animal_class or current.animal_class == "unknown"):
                        best_box = max(
                            (b for b in all_boxes if b.animal_class),
                            key=lambda b: b.animal_confidence or 0,
                            default=None,
                        )
                        if best_box and best_box.animal_class:
                            self._database.update_detection(
                                detection_id,
                                animal_class=best_box.animal_class,
                                confidence=best_box.animal_confidence,
                                bird_species=best_box.bird_species,
                            )
                            logger.info(
                                f"Updated animal from detector: "
                                f"{best_box.animal_class} ({best_box.animal_confidence:.1%})"
                            )
            except Exception as e:
                logger.error(f"Object detection failed for {metadata.filepath}: {e}")
        
        # Audio classification (runs after visual analysis)
        if self._audio_classifier and metadata.audio_path and metadata.audio_path.exists():
            try:
                current = self._database.get_detection(detection_id)
                visual_class = current.animal_class if current else None

                audio_result = self._audio_classifier.classify_audio(
                    metadata.audio_path,
                    visual_animal_class=visual_class,
                )

                if audio_result.has_sound:
                    self._database.update_detection(
                        detection_id,
                        sound_class=audio_result.sound_class,
                        sound_species=audio_result.sound_species,
                        sound_confidence=audio_result.sound_confidence,
                    )
                    species_info = f" — {audio_result.sound_species}" if audio_result.sound_species else ""
                    logger.info(
                        f"Audio: {audio_result.sound_class} "
                        f"({audio_result.sound_confidence:.1%}){species_info}"
                    )
                else:
                    logger.info("No animal sound detected in audio")
            except Exception as e:
                logger.error(f"Audio classification failed: {e}")

        # Mux audio into video if an animal was detected, otherwise discard it
        if metadata.audio_path and metadata.audio_path.exists():
            det = self._database.get_detection(detection_id)
            animal_found = det and (
                (det.animal_class and det.animal_class != "unknown")
                or det.sound_class is not None
            )
            if animal_found:
                from src.capture.audio_mux import mux_audio
                mux_audio(metadata.filepath, metadata.audio_path)
            else:
                metadata.audio_path.unlink(missing_ok=True)
                logger.debug("No animal detected, audio discarded")
        
        self._database.update_daily_summary()
    
    def _start_web_server(self) -> None:
        """Start the Flask web server in a background thread."""
        from src.web import create_app
        
        self._web_app = create_app(
            video_dir=self.video_dir,
            db_path=self.db_path,
        )
        self._web_app.config["MONITOR"] = self
        
        def run_server():
            from werkzeug.serving import make_server
            self._server = make_server(
                "0.0.0.0", 
                self.web_port, 
                self._web_app,
                threaded=True,
            )
            logger.info(f"Web UI available at http://localhost:{self.web_port}")
            self._server.serve_forever()
        
        self._web_thread = threading.Thread(target=run_server, daemon=True)
        self._web_thread.start()
    
    def _init_timelapse_scheduler(self) -> None:
        """Initialize the daily timelapse generation scheduler."""
        if not self.enable_timelapse:
            logger.info("Timelapse disabled via CLI flag, skipping")
            return

        timelapse_cfg = self._config_data.get("timelapse", {})
        if timelapse_cfg.get("enabled") is False:
            logger.info("Timelapse disabled in config, skipping")
            return

        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from src.storage.timelapse import run_timelapse_job

        generation_hour = timelapse_cfg.get("generation_hour", 21)
        generation_minute = timelapse_cfg.get("generation_minute", 0)
        frame_duration = timelapse_cfg.get("frame_duration", 0.5)
        resolution = timelapse_cfg.get("resolution", [1280, 720])
        fps = 1.0 / frame_duration

        self._timelapse_scheduler = BackgroundScheduler()
        self._timelapse_scheduler.add_job(
            run_timelapse_job,
            trigger=CronTrigger(hour=generation_hour, minute=generation_minute),
            args=[self._database, self.video_dir],
            kwargs={"fps": fps, "resolution": tuple(resolution)},
        )
        self._timelapse_scheduler.start()

        logger.info(
            f"Timelapse scheduler started: generation_hour={generation_hour}, "
            f"generation_minute={generation_minute}, frame_duration={frame_duration}, "
            f"fps={fps}, resolution={tuple(resolution)}"
        )

    def start(self, capture: bool = True, web: bool = True) -> None:
        """
        Start the wildlife monitor.
        
        Args:
            capture: Enable capture service
            web: Enable web UI
        """
        if self._running:
            logger.warning("Wildlife monitor already running")
            return
        
        self._running = True
        
        self._init_database()
        
        if capture:
            self._init_classifier()
            self._init_detector()
            self._init_audio_classifier()
            self._init_capture_service()
            self._capture_service.start()
            self._init_timelapse_scheduler()
        
        self._init_cleanup_scheduler()
        
        if web:
            self._start_web_server()
        
        mode_parts = []
        if capture:
            mode_parts.append("capture")
        if web:
            mode_parts.append("web")
        
        logger.info(f"Wildlife Monitor started ({', '.join(mode_parts)})")
    
    def stop(self) -> None:
        """Stop the wildlife monitor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._capture_service:
            self._capture_service.stop()
        
        if self._timelapse_scheduler and self._timelapse_scheduler.running:
            self._timelapse_scheduler.shutdown(wait=False)
        
        if self._classifier:
            self._classifier.unload_model()
        
        if self._detector:
            self._detector.unload_model()
        
        if self._cleanup_scheduler and self._cleanup_scheduler.running:
            self._cleanup_scheduler.shutdown(wait=False)
        
        if hasattr(self, '_server'):
            self._server.shutdown()
        
        if self._database:
            self._database.close()
        
        logger.info("Wildlife Monitor stopped")
    
    def analyze_video(self, video_path: Path) -> dict:
        """
        Analyze a single video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Classification result as dictionary
        """
        self._init_classifier()
        
        if not self._classifier:
            return {"error": "Classifier not available"}
        
        result = self._classifier.classify_video(video_path)
        self._classifier.unload_model()
        
        return result.to_dict()
    
    @property
    def is_running(self) -> bool:
        return self._running


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def wait_for_shutdown(monitor: WildlifeMonitor) -> None:
    """Wait for shutdown signal."""
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\nWildlife Monitor running. Press Ctrl+C to stop.\n")
    
    shutdown_event.wait()
    monitor.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Wildlife Monitor - Detect and classify animals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      Run full system (capture + web)
  python main.py --capture-only       Run capture without web UI
  python main.py --web-only           Run web UI only (view existing data)
  python main.py --analyze video.mp4  Analyze a specific video file
  python main.py --simulate           Force simulation mode (no hardware)
        """,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/videos"),
        help="Directory for video files (default: data/videos)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/wildlife.db"),
        help="Path to SQLite database (default: data/wildlife.db)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Web UI port (default: 5001)",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Run capture service without web UI",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Run web UI only (no capture)",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable ML analysis (just record videos)",
    )
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Disable object detection (bounding boxes)",
    )
    parser.add_argument(
        "--no-timelapse",
        action="store_true",
        help="Disable daily timelapse generation",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable automatic cleanup of empty detections",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Force simulation mode (no Pi hardware required)",
    )
    parser.add_argument(
        "--analyze",
        type=Path,
        metavar="VIDEO",
        help="Analyze a single video file and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.analyze:
        if not args.analyze.exists():
            print(f"Error: Video file not found: {args.analyze}")
            sys.exit(1)
        
        monitor = WildlifeMonitor()
        result = monitor.analyze_video(args.analyze)
        
        print(f"\nAnalysis Result:")
        print(f"  Animal: {result.get('animal_class', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.1%}")
        print(f"  Is animal: {result.get('is_animal', False)}")
        
        if result.get("top_predictions"):
            print("  Top predictions:")
            for animal, conf in result["top_predictions"]:
                print(f"    - {animal}: {conf:.1%}")
        
        sys.exit(0)
    
    simulation_mode = True if args.simulate else None
    
    monitor = WildlifeMonitor(
        config_path=args.config,
        video_dir=args.video_dir,
        db_path=args.db_path,
        simulation_mode=simulation_mode,
        web_port=args.port,
        enable_analysis=not args.no_analysis,
        enable_detection=not args.no_detection,
        enable_timelapse=not args.no_timelapse,
        enable_cleanup=not args.no_cleanup,
    )
    
    capture = not args.web_only
    web = not args.capture_only
    
    monitor.start(capture=capture, web=web)
    
    try:
        wait_for_shutdown(monitor)
    except Exception as e:
        logger.error(f"Error: {e}")
        monitor.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
