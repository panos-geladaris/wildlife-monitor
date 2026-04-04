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
    if name in ("ObjectDetector", "DetectionBox"):
        from .detector import ObjectDetector, DetectionBox
        return {"ObjectDetector": ObjectDetector, "DetectionBox": DetectionBox}[name]
    if name == "DetectionModelLoader":
        from .detection_model import DetectionModelLoader
        return DetectionModelLoader
    if name in ("annotate_frame", "save_annotated_frames", "cleanup_annotated_frames"):
        from .annotator import annotate_frame, save_annotated_frames, cleanup_annotated_frames
        return {"annotate_frame": annotate_frame, "save_annotated_frames": save_annotated_frames, "cleanup_annotated_frames": cleanup_annotated_frames}[name]
    if name in ("AudioClassifier", "AudioClassificationResult"):
        from .audio_classifier import AudioClassifier, AudioClassificationResult
        return {"AudioClassifier": AudioClassifier, "AudioClassificationResult": AudioClassificationResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnimalClassifier",
    "ClassificationResult",
    "FrameExtractor",
    "ModelLoader",
    "ANIMAL_CLASSES",
    "IMAGENET_TO_ANIMAL",
    "ObjectDetector",
    "DetectionBox",
    "DetectionModelLoader",
    "annotate_frame",
    "save_annotated_frames",
    "cleanup_annotated_frames",
    "AudioClassifier",
    "AudioClassificationResult",
]
