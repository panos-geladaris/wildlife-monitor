"""
Animal classifier for wildlife detection.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from PIL import Image

from .model import ModelLoader, IMAGENET_TO_ANIMAL, ANIMAL_CLASSES
from .frame_extractor import FrameExtractor, ExtractedFrame

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of animal classification."""
    animal_class: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: Optional[int] = None
    frame_timestamp: Optional[float] = None
    raw_class_idx: Optional[int] = None
    top_predictions: list[tuple[str, float]] = field(default_factory=list)
    
    @property
    def is_animal(self) -> bool:
        """Check if an animal was detected."""
        return self.animal_class != "unknown" and self.animal_class is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "animal_class": self.animal_class,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number,
            "frame_timestamp": self.frame_timestamp,
            "is_animal": self.is_animal,
            "top_predictions": self.top_predictions,
        }


class AnimalClassifier:
    """
    Classifies animals in images and videos.
    
    Uses a pretrained TorchVision model to identify animals
    from the ANIMAL_CLASSES list.
    """
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_FRAMES_TO_ANALYZE = 5
    
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self._model_loader = ModelLoader(model_name=model_name, device=device)
        self._model_loaded = False
        
        logger.info(
            f"AnimalClassifier initialized: {model_name}, "
            f"threshold={confidence_threshold}"
        )
    
    def load_model(self) -> None:
        """Pre-load the model (optional, lazy loading otherwise)."""
        self._model_loader.load()
        self._model_loaded = True
        logger.info("Model pre-loaded")
    
    def classify_image(
        self,
        image: Image.Image,
        frame_info: Optional[ExtractedFrame] = None,
    ) -> ClassificationResult:
        """
        Classify an image to detect animals.
        
        Args:
            image: PIL Image to classify
            frame_info: Optional frame metadata
            
        Returns:
            ClassificationResult with detected animal
        """
        # Get top-5 predictions
        top_predictions = self._model_loader.predict_top_k(image, k=5)
        
        # Find the best animal prediction
        best_animal = "unknown"
        best_confidence = 0.0
        best_class_idx = None
        
        top_animal_predictions = []
        
        for class_idx, confidence in top_predictions:
            if class_idx in IMAGENET_TO_ANIMAL:
                animal = IMAGENET_TO_ANIMAL[class_idx]
                top_animal_predictions.append((animal, confidence))
                
                if confidence > best_confidence:
                    best_animal = animal
                    best_confidence = confidence
                    best_class_idx = class_idx
        
        # Apply confidence threshold
        if best_confidence < self.confidence_threshold:
            best_animal = "unknown"
        
        result = ClassificationResult(
            animal_class=best_animal,
            confidence=best_confidence,
            raw_class_idx=best_class_idx,
            top_predictions=top_animal_predictions[:3],
        )
        
        if frame_info:
            result.frame_number = frame_info.frame_number
            result.frame_timestamp = frame_info.timestamp_seconds
        
        logger.debug(
            f"Classification: {best_animal} ({best_confidence:.2%})"
        )
        
        return result
    
    def classify_video(
        self,
        video_path: Path,
        num_frames: int = DEFAULT_FRAMES_TO_ANALYZE,
    ) -> ClassificationResult:
        """
        Classify a video by analyzing multiple frames.
        
        Extracts frames from the video and returns the most
        confident animal detection across all frames.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze
            
        Returns:
            Best ClassificationResult from analyzed frames
        """
        video_path = Path(video_path)
        logger.info(f"Classifying video: {video_path.name}")
        
        results: list[ClassificationResult] = []
        
        try:
            with FrameExtractor(video_path) as extractor:
                frames = extractor.extract_key_frames(num_frames=num_frames)
                
                if not frames:
                    logger.warning(f"No frames extracted from: {video_path}")
                    return ClassificationResult(
                        animal_class="unknown",
                        confidence=0.0,
                    )
                
                for frame in frames:
                    result = self.classify_image(frame.image, frame_info=frame)
                    results.append(result)
                    
                    # Log progress
                    if result.is_animal:
                        logger.debug(
                            f"Frame {frame.frame_number}: "
                            f"{result.animal_class} ({result.confidence:.2%})"
                        )
        
        except Exception as e:
            logger.error(f"Error classifying video {video_path}: {e}")
            return ClassificationResult(
                animal_class="unknown",
                confidence=0.0,
            )
        
        # Return the result with highest confidence animal detection
        animal_results = [r for r in results if r.is_animal]
        
        if animal_results:
            best_result = max(animal_results, key=lambda r: r.confidence)
            logger.info(
                f"Video result: {best_result.animal_class} "
                f"({best_result.confidence:.2%})"
            )
            return best_result
        
        # No animal detected - return the highest confidence result anyway
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            return best_result
        
        return ClassificationResult(
            animal_class="unknown",
            confidence=0.0,
        )
    
    def classify_video_all_frames(
        self,
        video_path: Path,
        num_frames: int = DEFAULT_FRAMES_TO_ANALYZE,
    ) -> list[ClassificationResult]:
        """
        Classify a video and return results for all analyzed frames.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze
            
        Returns:
            List of ClassificationResult for each frame
        """
        video_path = Path(video_path)
        results: list[ClassificationResult] = []
        
        try:
            with FrameExtractor(video_path) as extractor:
                frames = extractor.extract_key_frames(num_frames=num_frames)
                
                for frame in frames:
                    result = self.classify_image(frame.image, frame_info=frame)
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Error classifying video {video_path}: {e}")
        
        return results
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model_loader.unload()
        self._model_loaded = False
    
    def __enter__(self):
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        return False


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Animal Classifier Test")
    print("=" * 40)
    
    # Test with dummy image
    with AnimalClassifier() as classifier:
        # Test image classification
        dummy_image = Image.new("RGB", (224, 224), color="brown")
        result = classifier.classify_image(dummy_image)
        
        print(f"\nDummy image result:")
        print(f"  Animal: {result.animal_class}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Is animal: {result.is_animal}")
        
        # Test with video if provided
        if len(sys.argv) > 1:
            video_path = Path(sys.argv[1])
            print(f"\nClassifying video: {video_path}")
            
            result = classifier.classify_video(video_path)
            print(f"  Animal: {result.animal_class}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Frame: {result.frame_number}")
            
            if result.top_predictions:
                print("  Top predictions:")
                for animal, conf in result.top_predictions:
                    print(f"    - {animal}: {conf:.2%}")
