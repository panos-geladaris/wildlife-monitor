"""
Audio classifier for wildlife sound recognition.

Two-pass classification:
1. PANNs (Pretrained Audio Neural Networks) — broad animal sound detection
2. BirdNET — species-level bird identification (conditional on Pass 1 or visual class)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import panns_inference
    from panns_inference import AudioTagging
    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False
    logger.debug("panns_inference not installed — Pass 1 (general audio) disabled")

try:
    import birdnet
    BIRDNET_AVAILABLE = True
except ImportError:
    BIRDNET_AVAILABLE = False
    logger.debug("birdnet not installed — Pass 2 (bird species) disabled")


# AudioSet labels that map to animal categories.
# Keys are AudioSet class names, values are our animal categories.
AUDIOSET_ANIMAL_LABELS: dict[str, str] = {
    # Birds
    "Bird": "bird",
    "Bird vocalization, bird call, bird song": "bird",
    "Chirp, tweet": "bird",
    "Squawk": "bird",
    "Pigeon, dove": "bird",
    "Crow": "bird",
    "Owl": "bird",
    "Bird flight, flapping wings": "bird",
    # Dogs
    "Dog": "dog",
    "Bark": "dog",
    "Howl": "dog",
    "Growling": "dog",
    "Whimper (dog)": "dog",
    "Yip": "dog",
    # Cats
    "Cat": "cat",
    "Purr": "cat",
    "Meow": "cat",
    "Hiss": "cat",
    "Caterwaul": "cat",
    # Frogs
    "Frog": "frog",
    "Croak": "frog",
    # Insects
    "Insect": "insect",
    "Buzz": "insect",
    "Cricket": "insect",
    "Mosquito": "insect",
    "Fly, housefly": "insect",
    "Bee, wasp, etc.": "insect",
    # Livestock / large animals
    "Livestock, farm animals, working animals": "livestock",
    "Horse": "livestock",
    "Cow": "livestock",
    "Pig": "livestock",
    "Goat": "livestock",
    "Sheep": "livestock",
    "Chicken, rooster": "livestock",
    "Duck": "bird",
    "Goose": "bird",
    # Wild mammals
    "Roar": "wild_mammal",
    "Wild animals": "wild_mammal",
    "Fox": "wild_mammal",
    # Rodents
    "Mouse": "rodent",
    "Rat": "rodent",
    "Squeak": "rodent",
    # Snake
    "Snake": "snake",
    "Rattle": "snake",
    # General
    "Animal": "animal",
}


@dataclass
class AudioClassificationResult:
    """Result of audio classification."""
    sound_class: Optional[str] = None
    sound_species: Optional[str] = None
    sound_confidence: Optional[float] = None
    all_predictions: list[tuple[str, float]] = field(default_factory=list)

    @property
    def has_sound(self) -> bool:
        return self.sound_class is not None

    def to_dict(self) -> dict:
        return {
            "sound_class": self.sound_class,
            "sound_species": self.sound_species,
            "sound_confidence": self.sound_confidence,
            "has_sound": self.has_sound,
            "all_predictions": self.all_predictions,
        }


class AudioClassifier:
    """
    Two-pass audio classifier for animal sounds.

    Pass 1: PANNs — detects broad animal sound categories (dog, cat, frog, bird, …)
    Pass 2: BirdNET — species-level bird ID (runs only when bird is indicated)
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
    ):
        self.min_confidence = min_confidence
        self.lat = lat
        self.lng = lng

        self._panns_available = PANNS_AVAILABLE
        self._birdnet_available = BIRDNET_AVAILABLE

        self._panns_model = None
        self._birdnet_model = None

        logger.info(
            f"AudioClassifier initialized: "
            f"panns={'yes' if self._panns_available else 'no'}, "
            f"birdnet={'yes' if self._birdnet_available else 'no'}, "
            f"threshold={min_confidence}"
        )

    def classify_audio(
        self,
        wav_path: Path,
        visual_animal_class: Optional[str] = None,
    ) -> AudioClassificationResult:
        """
        Classify animal sounds in an audio file.

        Args:
            wav_path: Path to the .wav file.
            visual_animal_class: The animal class from visual classification,
                used to decide whether to run BirdNET even if PANNs is unsure.

        Returns:
            AudioClassificationResult with sound_class, sound_species, and confidence.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}")
            return AudioClassificationResult()

        # Pass 1: PANNs — broad animal sound detection
        panns_result: dict[str, float] = {}
        if self._panns_available:
            try:
                panns_result = self._classify_panns(wav_path)
            except Exception as e:
                logger.error(f"PANNs classification failed: {e}")

        # Find best animal match from PANNs
        best_class, best_confidence, all_preds = self._extract_animal_class(panns_result)

        # Determine if we should run BirdNET
        is_bird_from_panns = best_class == "bird"
        is_bird_from_visual = visual_animal_class == "bird"
        should_run_birdnet = (is_bird_from_panns or is_bird_from_visual) and self._birdnet_available

        # Pass 2: BirdNET — species-level bird ID
        if should_run_birdnet:
            try:
                birdnet_result = self._classify_birdnet(wav_path)
                if birdnet_result is not None:
                    species, confidence = birdnet_result
                    return AudioClassificationResult(
                        sound_class="bird",
                        sound_species=species,
                        sound_confidence=confidence,
                        all_predictions=all_preds,
                    )
            except Exception as e:
                logger.error(f"BirdNET classification failed: {e}")

        # No BirdNET result — return PANNs result if we have one
        if best_class is not None and best_confidence is not None:
            return AudioClassificationResult(
                sound_class=best_class,
                sound_confidence=best_confidence,
                all_predictions=all_preds,
            )

        # Nothing found
        return AudioClassificationResult(all_predictions=all_preds)

    def _extract_animal_class(
        self, panns_result: dict[str, float]
    ) -> tuple[Optional[str], Optional[float], list[tuple[str, float]]]:
        """
        Extract the best animal class from PANNs predictions.

        Returns:
            (animal_class, confidence, all_animal_predictions)
        """
        all_preds: list[tuple[str, float]] = []
        best_class: Optional[str] = None
        best_confidence: float = 0.0
        fallback_class: Optional[str] = None
        fallback_confidence: float = 0.0

        for label, confidence in panns_result.items():
            if label in AUDIOSET_ANIMAL_LABELS:
                category = AUDIOSET_ANIMAL_LABELS[label]
                all_preds.append((label, confidence))
                if category == "animal":
                    # Keep as fallback in case no specific category is found
                    if confidence > fallback_confidence and confidence >= self.min_confidence:
                        fallback_class = category
                        fallback_confidence = confidence
                    continue
                if confidence > best_confidence and confidence >= self.min_confidence:
                    best_class = category
                    best_confidence = confidence

        all_preds.sort(key=lambda x: x[1], reverse=True)

        # Use the generic "animal" label only if no specific category was detected
        if best_class is None and fallback_class is not None:
            return fallback_class, fallback_confidence, all_preds
        if best_class is None:
            return None, None, all_preds
        return best_class, best_confidence, all_preds

    def _classify_panns(self, wav_path: Path) -> dict[str, float]:
        """
        Run PANNs inference on an audio file.

        Returns:
            Dict mapping AudioSet label names to confidence scores.
        """
        if not self._panns_available:
            return {}

        if self._panns_model is None:
            self._panns_model = AudioTagging(checkpoint_path=None, device="cpu")
            logger.info("PANNs model loaded")

        import numpy as np
        import librosa

        audio, sr = librosa.load(str(wav_path), sr=32000, mono=True)
        audio = audio[np.newaxis, :]

        clipwise_output, _ = self._panns_model.inference(audio)
        probs = clipwise_output[0]

        labels = panns_inference.labels
        result = {}
        for idx, prob in enumerate(probs):
            if prob > 0.1:  # Only keep labels above noise floor
                result[labels[idx]] = float(prob)

        return result

    def _classify_birdnet(self, wav_path: Path) -> Optional[tuple[str, float]]:
        """
        Run BirdNET inference on an audio file.

        Returns:
            (species_name, confidence) tuple, or None if no species detected.
        """
        if not self._birdnet_available:
            return None

        if self._birdnet_model is None:
            self._birdnet_model = birdnet.load("acoustic", "2.4", "tflite")
            logger.info("BirdNET model loaded")

        kwargs = {}
        if self.lat is not None and self.lng is not None:
            kwargs["lat"] = self.lat
            kwargs["lon"] = self.lng

        predictions = self._birdnet_model.predict(str(wav_path), **kwargs)

        # Find the best prediction above threshold
        best_species = None
        best_confidence = 0.0

        for _, row in predictions.iterrows():
            confidence = row["confidence"]
            if confidence > best_confidence and confidence >= self.min_confidence:
                best_species = row["species_name"]
                best_confidence = confidence

        if best_species is not None:
            return best_species, best_confidence

        return None
