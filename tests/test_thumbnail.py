"""Tests for thumbnail generation."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def fake_video(tmp_dir):
    """Create a real video file using OpenCV so thumbnail generation works."""
    import cv2

    video_path = tmp_dir / "motion_20260131_120000.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for _ in range(15):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestGenerateThumbnail:

    def test_generates_thumbnail_in_default_location(self, fake_video):
        from src.storage.thumbnail import generate_thumbnail

        result = generate_thumbnail(fake_video)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".jpg"
        assert result.parent.name == "thumbnails"
        assert result.stem == fake_video.stem

    def test_generates_thumbnail_at_custom_path(self, fake_video, tmp_dir):
        from src.storage.thumbnail import generate_thumbnail

        custom = tmp_dir / "custom_thumb.jpg"
        result = generate_thumbnail(fake_video, output_path=custom)

        assert result == custom
        assert custom.exists()

    def test_thumbnail_has_correct_dimensions(self, fake_video, tmp_dir):
        import cv2
        from src.storage.thumbnail import generate_thumbnail

        result = generate_thumbnail(fake_video, width=160, height=90)

        assert result is not None
        img = cv2.imread(str(result))
        assert img.shape[1] == 160
        assert img.shape[0] == 90

    def test_returns_none_for_missing_video(self, tmp_dir):
        from src.storage.thumbnail import generate_thumbnail

        result = generate_thumbnail(tmp_dir / "nonexistent.mp4")
        assert result is None

    def test_returns_none_when_cv2_unavailable(self, fake_video):
        from src.storage import thumbnail

        with patch.object(thumbnail, "CV2_AVAILABLE", False):
            result = thumbnail.generate_thumbnail(fake_video)
            assert result is None


class TestThumbnailRoute:

    @pytest.fixture
    def app_and_video(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            video_dir = Path(tmpdir) / "videos"
            video_dir.mkdir()
            db_path = Path(tmpdir) / "test.db"

            video_path = video_dir / "motion_20260131_120000.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
            for _ in range(10):
                frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

            from src.web.app import create_app
            app = create_app(video_dir=video_dir, db_path=db_path)
            app.config["TESTING"] = True
            yield app, video_path

    def test_on_demand_thumbnail_generation(self, app_and_video):
        app, video_path = app_and_video
        client = app.test_client()

        response = client.get("/thumbnails/motion_20260131_120000.jpg")
        assert response.status_code == 200
        assert response.content_type == "image/jpeg"

    def test_serves_existing_thumbnail(self, app_and_video):
        app, video_path = app_and_video
        thumb_dir = video_path.parent / "thumbnails"
        thumb_dir.mkdir()
        thumb_path = thumb_dir / "motion_20260131_120000.jpg"
        thumb_path.write_bytes(b"fake jpeg")

        client = app.test_client()
        response = client.get("/thumbnails/motion_20260131_120000.jpg")
        assert response.status_code == 200

    def test_returns_404_for_no_matching_video(self, app_and_video):
        app, _ = app_and_video
        client = app.test_client()

        response = client.get("/thumbnails/nonexistent.jpg")
        assert response.status_code == 404
