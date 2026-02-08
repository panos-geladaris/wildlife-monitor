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
    ):
        self.config_path = config_path
        self.video_dir = video_dir or Path("data/videos")
        self.db_path = db_path or Path("data/wildlife.db")
        self.web_port = web_port
        self.enable_analysis = enable_analysis
        self.enable_detection = enable_detection
        
        self._capture_service = None
        self._classifier = None
        self._detector = None
        self._database = None
        self._web_app = None
        self._web_thread = None
        self._running = False
        self._simulation_mode = simulation_mode
        
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        
        try:
            from src.analysis import ObjectDetector
            self._detector = ObjectDetector(
                score_threshold=0.3,
                max_boxes_per_frame=5,
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
    
    def _init_capture_service(self) -> None:
        """Initialize the capture service."""
        from src.capture import CaptureService, load_config
        
        config = load_config(self.config_path)
        config.video_output_dir = self.video_dir
        config.simulation_mode = self._detect_simulation_mode()
        
        self._capture_service = CaptureService(config)
        self._capture_service.on_capture(self._on_video_captured)
        
        logger.info(f"Capture service initialized (simulation={config.simulation_mode})")
    
    def _on_video_captured(self, metadata) -> None:
        """Handle new video capture - analyze and store in database."""
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
                    analyzed=True,
                )
                
                if result.is_animal:
                    logger.info(
                        f"Detected: {result.animal_class} "
                        f"({result.confidence:.1%})"
                    )
                else:
                    logger.info("No animal detected in video")
                    
            except Exception as e:
                logger.error(f"Analysis failed for {metadata.filepath}: {e}")
                self._database.update_detection(detection_id, analyzed=True)
        
        if self._detector and metadata.filepath.exists():
            try:
                frames_with_boxes = self._detector.detect_video_with_images(
                    metadata.filepath, num_frames=5
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
            except Exception as e:
                logger.error(f"Object detection failed for {metadata.filepath}: {e}")
        
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
            self._init_capture_service()
            self._capture_service.start()
        
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
        
        if self._classifier:
            self._classifier.unload_model()
        
        if self._detector:
            self._detector.unload_model()
        
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
