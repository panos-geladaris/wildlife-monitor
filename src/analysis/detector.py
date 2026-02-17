"""
Object detector for wildlife videos using SSDLite320 + MobileNetV3-Large.

Supports tiled detection: splits high-resolution frames into overlapping
tiles so that small/distant animals occupy enough pixels for the 320×320
model to recognise them.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .detection_model import DetectionModelLoader, COCO_ANIMAL_LABELS
from .frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Apply greedy Non-Maximum Suppression across detections."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep: list[dict] = []

    for det in detections:
        box = det["box"]
        suppress = False
        for kept in keep:
            if _compute_iou(box, kept["box"]) > iou_threshold:
                suppress = True
                break
        if not suppress:
            keep.append(det)

    return keep


@dataclass
class DetectionBox:
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    img_width: int
    img_height: int
    frame_number: int
    frame_timestamp: float
    animal_class: Optional[str] = None
    animal_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "frame_number": self.frame_number,
            "frame_timestamp": self.frame_timestamp,
            "animal_class": self.animal_class,
            "animal_confidence": self.animal_confidence,
        }


class ObjectDetector:
    """
    Detects animals in videos using SSDLite320 and optionally refines
    labels by cropping detected regions and running the AnimalClassifier.
    """

    DEFAULT_SCORE_THRESHOLD = 0.3
    DEFAULT_FRAMES_TO_ANALYZE = 5
    DEFAULT_MAX_BOXES_PER_FRAME = 5
    DEFAULT_TILE_GRID = (1, 1)
    DEFAULT_TILE_OVERLAP = 0.2
    DEFAULT_NMS_IOU_THRESHOLD = 0.5

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_boxes_per_frame: int = DEFAULT_MAX_BOXES_PER_FRAME,
        device: Optional[str] = None,
        classify_crops: bool = True,
        tile_grid: tuple[int, int] = DEFAULT_TILE_GRID,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    ):
        self.score_threshold = score_threshold
        self.max_boxes_per_frame = max_boxes_per_frame
        self.classify_crops = classify_crops
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap
        self.nms_iou_threshold = nms_iou_threshold
        self._model_loader = DetectionModelLoader(device=device)
        self._classifier = None
        self._model_loaded = False

        tiling_enabled = tile_grid != (1, 1)
        logger.info(
            f"ObjectDetector initialized: threshold={score_threshold}, "
            f"max_boxes={max_boxes_per_frame}, classify_crops={classify_crops}, "
            f"tiling={'%dx%d overlap=%.0f%%' % (tile_grid[0], tile_grid[1], tile_overlap * 100) if tiling_enabled else 'off'}"
        )

    def load_model(self) -> None:
        self._model_loader.load()
        self._model_loaded = True

        if self.classify_crops:
            try:
                from .classifier import AnimalClassifier
                self._classifier = AnimalClassifier(
                    model_name="mobilenet_v3_small",
                    confidence_threshold=0.3,
                )
                self._classifier.load_model()
            except Exception as e:
                logger.warning(f"Could not load classifier for crop refinement: {e}")
                self._classifier = None

        logger.info("Object detector models loaded")

    def _classify_crop(self, image: Image.Image, box: DetectionBox) -> None:
        if self._classifier is None:
            return

        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        x2 = min(box.img_width, int(box.x2))
        y2 = min(box.img_height, int(box.y2))

        crop = image.crop((x1, y1, x2, y2))
        if crop.width < 10 or crop.height < 10:
            return

        result = self._classifier.classify_image(crop)
        if result.is_animal:
            box.animal_class = result.animal_class
            box.animal_confidence = result.confidence

    def _generate_tiles(
        self, width: int, height: int
    ) -> list[tuple[int, int, int, int]]:
        """Generate overlapping tile regions as (x1, y1, x2, y2) in pixel coords."""
        cols, rows = self.tile_grid
        overlap = self.tile_overlap

        tile_w = width / cols
        tile_h = height / rows
        pad_w = tile_w * overlap / 2
        pad_h = tile_h * overlap / 2

        tiles = []
        for r in range(rows):
            for c in range(cols):
                x1 = max(0, int(c * tile_w - pad_w))
                y1 = max(0, int(r * tile_h - pad_h))
                x2 = min(width, int((c + 1) * tile_w + pad_w))
                y2 = min(height, int((r + 1) * tile_h + pad_h))
                tiles.append((x1, y1, x2, y2))

        return tiles

    def _detect_tiled(
        self, image: Image.Image
    ) -> list[dict]:
        """Run detection on each tile, remap coords to full frame, then NMS."""
        w, h = image.size
        tiles = self._generate_tiles(w, h)
        all_detections: list[dict] = []

        for tx1, ty1, tx2, ty2 in tiles:
            tile_img = image.crop((tx1, ty1, tx2, ty2))
            tile_dets = self._model_loader.predict_animals(
                tile_img, score_threshold=self.score_threshold
            )

            for det in tile_dets:
                box = det["box"]
                det["box"] = [
                    box[0] + tx1,
                    box[1] + ty1,
                    box[2] + tx1,
                    box[3] + ty1,
                ]
                all_detections.append(det)

        return _nms(all_detections, iou_threshold=self.nms_iou_threshold)

    def detect_frame(
        self, image: Image.Image, frame_number: int = 0, frame_timestamp: float = 0.0
    ) -> list[DetectionBox]:
        tiling_enabled = self.tile_grid != (1, 1)

        if tiling_enabled:
            raw = self._detect_tiled(image)
        else:
            raw = self._model_loader.predict_animals(
                image, score_threshold=self.score_threshold
            )

        raw.sort(key=lambda d: d["score"], reverse=True)
        raw = raw[: self.max_boxes_per_frame]

        w, h = image.size
        boxes: list[DetectionBox] = []

        for det in raw:
            coords = det["box"]
            box = DetectionBox(
                label=det["label"],
                score=det["score"],
                x1=coords[0],
                y1=coords[1],
                x2=coords[2],
                y2=coords[3],
                img_width=w,
                img_height=h,
                frame_number=frame_number,
                frame_timestamp=frame_timestamp,
            )

            if det["label"] in COCO_ANIMAL_LABELS:
                box.animal_class = det["label"]
                box.animal_confidence = det["score"]

            if self.classify_crops:
                self._classify_crop(image, box)

            boxes.append(box)

        return boxes

    def detect_video(
        self,
        video_path: Path,
        num_frames: int = DEFAULT_FRAMES_TO_ANALYZE,
    ) -> dict[int, list[DetectionBox]]:
        video_path = Path(video_path)
        logger.info(f"Detecting objects in video: {video_path.name}")

        results: dict[int, list[DetectionBox]] = {}

        try:
            with FrameExtractor(video_path) as extractor:
                frames = extractor.extract_key_frames(num_frames=num_frames)

                if not frames:
                    logger.warning(f"No frames extracted from: {video_path}")
                    return results

                for frame in frames:
                    boxes = self.detect_frame(
                        frame.image,
                        frame_number=frame.frame_number,
                        frame_timestamp=frame.timestamp_seconds,
                    )
                    if boxes:
                        results[frame.frame_number] = boxes
                        logger.debug(
                            f"Frame {frame.frame_number}: {len(boxes)} detections"
                        )

        except Exception as e:
            logger.error(f"Error detecting objects in {video_path}: {e}")

        total_boxes = sum(len(b) for b in results.values())
        logger.info(
            f"Video detection complete: {total_boxes} objects in {len(results)} frames"
        )
        return results

    def detect_video_with_images(
        self,
        video_path: Path,
        num_frames: int = DEFAULT_FRAMES_TO_ANALYZE,
    ) -> list[tuple[Image.Image, list[DetectionBox]]]:
        video_path = Path(video_path)
        logger.info(f"Detecting objects (with images) in video: {video_path.name}")

        results: list[tuple[Image.Image, list[DetectionBox]]] = []

        try:
            with FrameExtractor(video_path) as extractor:
                frames = extractor.extract_key_frames(num_frames=num_frames)

                if not frames:
                    logger.warning(f"No frames extracted from: {video_path}")
                    return results

                for frame in frames:
                    boxes = self.detect_frame(
                        frame.image,
                        frame_number=frame.frame_number,
                        frame_timestamp=frame.timestamp_seconds,
                    )
                    results.append((frame.image, boxes))

        except Exception as e:
            logger.error(f"Error detecting objects in {video_path}: {e}")

        return results

    def unload_model(self) -> None:
        self._model_loader.unload()
        if self._classifier:
            self._classifier.unload_model()
            self._classifier = None
        self._model_loaded = False

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        return False
