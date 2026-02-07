"""
Bounding box annotation and annotated frame saving for wildlife detections.
"""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .detector import DetectionBox

logger = logging.getLogger(__name__)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "bird": (0, 165, 255),
    "cat": (0, 255, 0),
    "dog": (255, 144, 30),
    "squirrel": (0, 215, 255),
    "fox": (0, 69, 255),
    "rabbit": (203, 192, 255),
    "deer": (19, 69, 139),
    "hedgehog": (128, 128, 0),
    "mouse": (180, 105, 255),
    "hamster": (0, 255, 255),
    "beaver": (42, 42, 165),
    "unknown": (128, 128, 128),
}

DEFAULT_COLOR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2


def annotate_frame(image: Image.Image, boxes: list["DetectionBox"]) -> Image.Image:
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for box in boxes:
        animal = box.animal_class or box.label
        color = CLASS_COLORS.get(animal, DEFAULT_COLOR)
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        score_pct = box.animal_confidence if box.animal_confidence is not None else box.score
        text = f"{animal} {score_pct:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

        text_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 4, text_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, text, (x1 + 2, text_y - 2), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def save_annotated_frames(
    detection_id: int,
    frames_with_boxes: list[tuple[Image.Image, list["DetectionBox"]]],
    output_dir: Path,
) -> list[Path]:
    dest = output_dir / str(detection_id)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for idx, (image, boxes) in enumerate(frames_with_boxes):
        annotated = annotate_frame(image, boxes)
        frame_number = boxes[0].frame_number if boxes else idx
        path = dest / f"frame_{frame_number}.jpg"
        annotated.save(path, format="JPEG")
        saved.append(path)
        logger.debug(f"Saved annotated frame: {path}")

    logger.info(f"Saved {len(saved)} annotated frames for detection {detection_id}")
    return saved


def cleanup_annotated_frames(detection_id: int, output_dir: Path) -> None:
    dest = output_dir / str(detection_id)
    if dest.exists():
        shutil.rmtree(dest)
        logger.info(f"Cleaned up annotated frames for detection {detection_id}")
