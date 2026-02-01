"""
Analysis module for wildlife monitor.

Provides ML-based animal classification using TorchVision models.
"""


def __getattr__(name: str):
    """Lazy imports to avoid loading heavy ML dependencies until needed."""
    if name in ("AnimalClassifier", "ClassificationResult"):
        from .classifier import AnimalClassifier, ClassificationResult
        return {"AnimalClassifier": AnimalClassifier, "ClassificationResult": ClassificationResult}[name]
    if name == "FrameExtractor":
        from .frame_extractor import FrameExtractor
        return FrameExtractor
    if name in ("ModelLoader", "ANIMAL_CLASSES", "IMAGENET_TO_ANIMAL"):
        from .model import ModelLoader, ANIMAL_CLASSES, IMAGENET_TO_ANIMAL
        return {"ModelLoader": ModelLoader, "ANIMAL_CLASSES": ANIMAL_CLASSES, "IMAGENET_TO_ANIMAL": IMAGENET_TO_ANIMAL}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnimalClassifier",
    "ClassificationResult",
    "FrameExtractor",
    "ModelLoader",
    "ANIMAL_CLASSES",
    "IMAGENET_TO_ANIMAL",
]
