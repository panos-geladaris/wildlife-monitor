"""
Flask application for wildlife monitor web UI.
"""

import logging
from pathlib import Path
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

from .api import api_bp

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_VIDEO_DIR = Path(__file__).parent.parent.parent / "data" / "videos"
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "wildlife.db"


def create_app(
    video_dir: Path = None,
    db_path: Path = None,
    debug: bool = False,
) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        video_dir: Path to video files directory
        db_path: Path to SQLite database
        debug: Enable debug mode
        
    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    
    # Configuration
    app.config["DEBUG"] = debug
    app.config["VIDEO_DIR"] = video_dir or DEFAULT_VIDEO_DIR
    app.config["DB_PATH"] = db_path or DEFAULT_DB_PATH
    
    # Enable CORS for API endpoints
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api")
    
    # Page routes
    @app.route("/")
    def index():
        """Dashboard page."""
        return render_template("index.html")
    
    @app.route("/gallery")
    def gallery():
        """Video gallery page."""
        return render_template("gallery.html")
    
    @app.route("/detection/<int:detection_id>")
    def detection_detail(detection_id: int):
        """Detection detail page."""
        return render_template("detection.html", detection_id=detection_id)
    
    @app.route("/statistics")
    def statistics():
        """Statistics page."""
        return render_template("statistics.html")
    
    @app.route("/videos/<path:filename>")
    def serve_video(filename: str):
        """Serve video files."""
        video_dir = Path(app.config["VIDEO_DIR"]).resolve()
        return send_from_directory(video_dir, filename)

    @app.route("/thumbnails/<path:filename>")
    def serve_thumbnail(filename: str):
        """Serve thumbnail images, generating on-demand if missing."""
        video_dir = Path(app.config["VIDEO_DIR"]).resolve()
        thumb_dir = video_dir / "thumbnails"
        thumb_path = thumb_dir / filename

        if not thumb_path.exists():
            stem = Path(filename).stem
            for ext in (".mp4", ".h264", ".avi", ".mkv"):
                video_path = video_dir / (stem + ext)
                if video_path.exists():
                    from src.storage.thumbnail import generate_thumbnail
                    generate_thumbnail(video_path, output_path=thumb_path)
                    break

        return send_from_directory(thumb_dir.resolve(), filename)

    @app.route("/annotated/<int:detection_id>/<path:filename>")
    def serve_annotated(detection_id: int, filename: str):
        """Serve annotated frame images."""
        video_dir = Path(app.config["VIDEO_DIR"]).resolve()
        annotated_dir = video_dir.parent / "annotated" / str(detection_id)
        return send_from_directory(annotated_dir.resolve(), filename)
    
    logger.info(f"Flask app created: video_dir={app.config['VIDEO_DIR']}")
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    app = create_app(debug=True)
    app.run(host="0.0.0.0", port=5001, debug=True)
