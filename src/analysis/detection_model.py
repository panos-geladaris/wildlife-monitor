"""
SSDLite320 + MobileNetV3-Large object detection model for wildlife monitoring.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image

logger = logging.getLogger(__name__)

COCO_LABELS: dict[int, str] = {
    0: "__background__",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

COCO_ANIMAL_LABELS: set[str] = {
    "bird", "cat", "dog",
}


class DetectionModelLoader:
    """
    Loads and manages SSDLite320 + MobileNetV3-Large for object detection.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Optional[nn.Module] = None
        self._transform = None

        logger.info(f"DetectionModelLoader initialized on {self.device}")

    def load(self) -> nn.Module:
        if self._model is not None:
            return self._model

        logger.info("Loading SSDLite320 MobileNetV3-Large")

        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self._model = ssdlite320_mobilenet_v3_large(weights=weights)
        self._transform = weights.transforms()

        self._model = self._model.to(self.device)
        self._model.eval()

        logger.info("Detection model loaded successfully")
        return self._model

    def predict(self, image: Image.Image) -> list[dict]:
        model = self.load()
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = model(input_tensor)

        result = outputs[0]
        return [
            {
                "boxes": result["boxes"].cpu(),
                "labels": result["labels"].cpu(),
                "scores": result["scores"].cpu(),
            }
        ]

    def predict_animals(
        self, image: Image.Image, score_threshold: float = 0.3
    ) -> list[dict]:
        model = self.load()
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = model(input_tensor)

        result = outputs[0]
        detections: list[dict] = []

        for box, label, score in zip(
            result["boxes"], result["labels"], result["scores"]
        ):
            score_val = score.item()
            label_id = label.item()
            label_name = COCO_LABELS.get(label_id, "unknown")

            if label_name in COCO_ANIMAL_LABELS and score_val >= score_threshold:
                detections.append(
                    {
                        "box": box.cpu().tolist(),
                        "label": label_name,
                        "label_id": label_id,
                        "score": score_val,
                    }
                )

        return detections

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Detection model unloaded")
