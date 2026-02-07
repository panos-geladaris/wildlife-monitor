"""Tests for the object detection module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.analysis.detection_model import (
    DetectionModelLoader,
    COCO_LABELS,
    COCO_ANIMAL_LABELS,
)
from src.analysis.detector import ObjectDetector, DetectionBox
from src.analysis.annotator import (
    annotate_frame,
    save_annotated_frames,
    cleanup_annotated_frames,
    CLASS_COLORS,
)


class TestCOCOLabels:

    def test_coco_labels_has_91_entries(self):
        assert len(COCO_LABELS) == 91

    def test_coco_labels_background(self):
        assert COCO_LABELS[0] == "__background__"

    def test_coco_animal_labels_count(self):
        assert len(COCO_ANIMAL_LABELS) == 10

    def test_coco_animal_labels_content(self):
        expected = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
        assert COCO_ANIMAL_LABELS == expected

    def test_all_animal_labels_in_coco_labels(self):
        label_values = set(COCO_LABELS.values())
        for animal in COCO_ANIMAL_LABELS:
            assert animal in label_values


class TestDetectionModelLoader:

    def test_init_default_device(self):
        with patch("src.analysis.detection_model.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            loader = DetectionModelLoader()
            assert loader.device == "cpu"

    def test_init_custom_device(self):
        with patch("src.analysis.detection_model.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            loader = DetectionModelLoader(device="cpu")
            assert loader.device == "cpu"

    def test_unload_without_model(self):
        with patch("src.analysis.detection_model.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            loader = DetectionModelLoader()
            loader.unload()
            assert loader._model is None


class TestDetectionBox:

    def test_create_box(self):
        box = DetectionBox(
            label="cat",
            score=0.95,
            x1=10.0,
            y1=20.0,
            x2=100.0,
            y2=200.0,
            img_width=640,
            img_height=480,
            frame_number=5,
            frame_timestamp=0.5,
        )
        assert box.label == "cat"
        assert box.score == 0.95
        assert box.animal_class is None

    def test_box_with_animal_class(self):
        box = DetectionBox(
            label="cat",
            score=0.95,
            x1=10.0,
            y1=20.0,
            x2=100.0,
            y2=200.0,
            img_width=640,
            img_height=480,
            frame_number=5,
            frame_timestamp=0.5,
            animal_class="cat",
            animal_confidence=0.92,
        )
        assert box.animal_class == "cat"
        assert box.animal_confidence == 0.92

    def test_to_dict(self):
        box = DetectionBox(
            label="dog",
            score=0.88,
            x1=50.0,
            y1=60.0,
            x2=200.0,
            y2=300.0,
            img_width=1280,
            img_height=720,
            frame_number=10,
            frame_timestamp=1.0,
            animal_class="dog",
            animal_confidence=0.85,
        )
        d = box.to_dict()
        assert d["label"] == "dog"
        assert d["score"] == 0.88
        assert d["x1"] == 50.0
        assert d["y2"] == 300.0
        assert d["animal_class"] == "dog"
        assert d["frame_number"] == 10

    def test_to_dict_no_animal(self):
        box = DetectionBox(
            label="bird",
            score=0.7,
            x1=0,
            y1=0,
            x2=100,
            y2=100,
            img_width=640,
            img_height=480,
            frame_number=0,
            frame_timestamp=0.0,
        )
        d = box.to_dict()
        assert d["animal_class"] is None
        assert d["animal_confidence"] is None


class TestObjectDetector:

    @pytest.fixture
    def mock_detector(self):
        with patch.object(DetectionModelLoader, "load") as mock_load, \
             patch.object(DetectionModelLoader, "predict_animals") as mock_predict:
            mock_load.return_value = MagicMock()
            mock_predict.return_value = []

            detector = ObjectDetector(
                score_threshold=0.3,
                max_boxes_per_frame=5,
                classify_crops=False,
            )
            yield detector, mock_predict

    def test_init(self):
        with patch.object(DetectionModelLoader, "__init__", return_value=None):
            detector = ObjectDetector(
                score_threshold=0.5,
                max_boxes_per_frame=3,
            )
            assert detector.score_threshold == 0.5
            assert detector.max_boxes_per_frame == 3

    def test_detect_frame_no_animals(self, mock_detector):
        detector, mock_predict = mock_detector
        mock_predict.return_value = []

        image = Image.new("RGB", (640, 480))
        boxes = detector.detect_frame(image, frame_number=0)
        assert boxes == []

    def test_detect_frame_with_animal(self, mock_detector):
        detector, mock_predict = mock_detector
        mock_predict.return_value = [
            {"box": [10.0, 20.0, 100.0, 200.0], "label": "cat", "label_id": 17, "score": 0.9},
        ]

        image = Image.new("RGB", (640, 480))
        boxes = detector.detect_frame(image, frame_number=5, frame_timestamp=0.5)

        assert len(boxes) == 1
        assert boxes[0].label == "cat"
        assert boxes[0].score == 0.9
        assert boxes[0].frame_number == 5
        assert boxes[0].img_width == 640
        assert boxes[0].img_height == 480
        assert boxes[0].animal_class == "cat"

    def test_detect_frame_max_boxes(self, mock_detector):
        detector, mock_predict = mock_detector
        detector.max_boxes_per_frame = 2

        mock_predict.return_value = [
            {"box": [10, 20, 100, 200], "label": "cat", "label_id": 17, "score": 0.9},
            {"box": [200, 100, 300, 300], "label": "dog", "label_id": 18, "score": 0.8},
            {"box": [400, 200, 500, 400], "label": "bird", "label_id": 16, "score": 0.7},
        ]

        image = Image.new("RGB", (640, 480))
        boxes = detector.detect_frame(image)

        assert len(boxes) == 2
        assert boxes[0].score >= boxes[1].score

    def test_detect_frame_coco_animal_mapping(self, mock_detector):
        detector, mock_predict = mock_detector
        mock_predict.return_value = [
            {"box": [10, 20, 100, 200], "label": "bird", "label_id": 16, "score": 0.85},
        ]

        image = Image.new("RGB", (640, 480))
        boxes = detector.detect_frame(image)

        assert boxes[0].animal_class == "bird"
        assert boxes[0].animal_confidence == 0.85


class TestAnnotateFrame:

    def _make_box(self, **kwargs):
        defaults = {
            "label": "cat",
            "score": 0.9,
            "x1": 50,
            "y1": 60,
            "x2": 200,
            "y2": 300,
            "img_width": 640,
            "img_height": 480,
            "frame_number": 0,
            "frame_timestamp": 0.0,
            "animal_class": "cat",
            "animal_confidence": 0.9,
        }
        defaults.update(kwargs)
        return DetectionBox(**defaults)

    def test_annotate_preserves_dimensions(self):
        image = Image.new("RGB", (640, 480), color="green")
        boxes = [self._make_box()]
        result = annotate_frame(image, boxes)
        assert result.size == (640, 480)

    def test_annotate_empty_boxes(self):
        image = Image.new("RGB", (640, 480), color="green")
        result = annotate_frame(image, [])
        assert result.size == (640, 480)

    def test_annotate_multiple_boxes(self):
        image = Image.new("RGB", (640, 480), color="green")
        boxes = [
            self._make_box(label="cat", x1=10, y1=10, x2=100, y2=100),
            self._make_box(label="dog", animal_class="dog", x1=200, y1=200, x2=400, y2=400),
        ]
        result = annotate_frame(image, boxes)
        assert result.size == (640, 480)

    def test_annotate_uses_animal_class_for_label(self):
        image = Image.new("RGB", (640, 480), color="green")
        box = self._make_box(label="cat", animal_class="fox")
        result = annotate_frame(image, [box])
        assert result.size == (640, 480)

    def test_class_colors_defined(self):
        assert "bird" in CLASS_COLORS
        assert "cat" in CLASS_COLORS
        assert "dog" in CLASS_COLORS
        assert "unknown" in CLASS_COLORS


class TestSaveAnnotatedFrames:

    def _make_box(self, frame_number=0):
        return DetectionBox(
            label="cat",
            score=0.9,
            x1=50,
            y1=60,
            x2=200,
            y2=300,
            img_width=640,
            img_height=480,
            frame_number=frame_number,
            frame_timestamp=0.0,
            animal_class="cat",
            animal_confidence=0.9,
        )

    def test_save_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            image = Image.new("RGB", (640, 480), color="green")
            frames_with_boxes = [
                (image, [self._make_box(frame_number=0)]),
                (image, [self._make_box(frame_number=30)]),
            ]

            saved = save_annotated_frames(1, frames_with_boxes, output_dir)

            assert len(saved) == 2
            for path in saved:
                assert path.exists()
                assert path.suffix == ".jpg"

    def test_save_creates_detection_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            image = Image.new("RGB", (640, 480))
            frames_with_boxes = [(image, [self._make_box()])]

            save_annotated_frames(42, frames_with_boxes, output_dir)

            assert (output_dir / "42").is_dir()

    def test_save_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved = save_annotated_frames(1, [], output_dir)
            assert saved == []


class TestCleanupAnnotatedFrames:

    def test_cleanup_removes_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            det_dir = output_dir / "99"
            det_dir.mkdir()
            (det_dir / "frame_0.jpg").write_bytes(b"fake")

            cleanup_annotated_frames(99, output_dir)

            assert not det_dir.exists()

    def test_cleanup_nonexistent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cleanup_annotated_frames(999, output_dir)


class TestDatabaseFrameObjects:

    @pytest.fixture
    def db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        from src.storage.database import Database
        database = Database(db_path)
        yield database

        database.close()
        db_path.unlink(missing_ok=True)

    @pytest.fixture
    def detection_id(self, db):
        from src.storage.database import Detection
        return db.add_detection(Detection(
            video_path="/data/videos/test.mp4",
            trigger_type="motion",
        ))

    def test_frame_objects_table_created(self, db):
        with db._get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row["name"] for row in tables}
        assert "frame_objects" in table_names

    def test_add_frame_objects(self, db, detection_id):
        boxes = [
            {
                "frame_number": 0,
                "frame_timestamp": 0.0,
                "label": "cat",
                "score": 0.9,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 200.0,
                "img_width": 640,
                "img_height": 480,
                "animal_class": "cat",
                "animal_confidence": 0.9,
            },
        ]
        count = db.add_frame_objects(detection_id, boxes)
        assert count == 1

    def test_add_frame_objects_empty(self, db, detection_id):
        count = db.add_frame_objects(detection_id, [])
        assert count == 0

    def test_get_frame_objects(self, db, detection_id):
        boxes = [
            {
                "frame_number": 0,
                "frame_timestamp": 0.0,
                "label": "cat",
                "score": 0.9,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 200.0,
                "img_width": 640,
                "img_height": 480,
                "animal_class": "cat",
                "animal_confidence": 0.88,
            },
            {
                "frame_number": 30,
                "frame_timestamp": 1.0,
                "label": "bird",
                "score": 0.7,
                "x1": 200.0,
                "y1": 100.0,
                "x2": 300.0,
                "y2": 250.0,
                "img_width": 640,
                "img_height": 480,
                "animal_class": "bird",
                "animal_confidence": 0.65,
            },
        ]
        db.add_frame_objects(detection_id, boxes)

        retrieved = db.get_frame_objects(detection_id)
        assert len(retrieved) == 2
        assert retrieved[0]["frame_number"] == 0
        assert retrieved[0]["label"] == "cat"
        assert retrieved[0]["animal_confidence"] == 0.88
        assert retrieved[1]["frame_number"] == 30

    def test_get_frame_objects_empty(self, db, detection_id):
        retrieved = db.get_frame_objects(detection_id)
        assert retrieved == []

    def test_delete_frame_objects(self, db, detection_id):
        boxes = [
            {
                "frame_number": 0,
                "label": "cat",
                "score": 0.9,
                "x1": 10,
                "y1": 20,
                "x2": 100,
                "y2": 200,
            },
        ]
        db.add_frame_objects(detection_id, boxes)

        count = db.delete_frame_objects(detection_id)
        assert count == 1
        assert db.get_frame_objects(detection_id) == []

    def test_delete_frame_objects_empty(self, db, detection_id):
        count = db.delete_frame_objects(detection_id)
        assert count == 0

    def test_add_multiple_boxes_per_frame(self, db, detection_id):
        boxes = [
            {"frame_number": 5, "label": "cat", "score": 0.9, "x1": 10, "y1": 20, "x2": 100, "y2": 200},
            {"frame_number": 5, "label": "dog", "score": 0.8, "x1": 200, "y1": 100, "x2": 400, "y2": 300},
            {"frame_number": 5, "label": "bird", "score": 0.6, "x1": 400, "y1": 50, "x2": 500, "y2": 150},
        ]
        count = db.add_frame_objects(detection_id, boxes)
        assert count == 3

        retrieved = db.get_frame_objects(detection_id)
        assert len(retrieved) == 3
        assert all(r["frame_number"] == 5 for r in retrieved)
