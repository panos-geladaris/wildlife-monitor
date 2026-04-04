"""Tests for the audio classification module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.analysis.audio_classifier import (
    AudioClassifier,
    AudioClassificationResult,
    AUDIOSET_ANIMAL_LABELS,
)


class TestAudioClassificationResult:
    """Tests for AudioClassificationResult dataclass."""

    def test_default_values(self):
        result = AudioClassificationResult()
        assert result.sound_class is None
        assert result.sound_species is None
        assert result.sound_confidence is None
        assert result.all_predictions == []

    def test_has_sound_true(self):
        result = AudioClassificationResult(sound_class="bird", sound_confidence=0.8)
        assert result.has_sound is True

    def test_has_sound_false_when_none(self):
        result = AudioClassificationResult()
        assert result.has_sound is False

    def test_to_dict(self):
        result = AudioClassificationResult(
            sound_class="dog",
            sound_species=None,
            sound_confidence=0.82,
            all_predictions=[("Dog", 0.82)],
        )
        d = result.to_dict()
        assert d["sound_class"] == "dog"
        assert d["sound_species"] is None
        assert d["sound_confidence"] == 0.82
        assert d["has_sound"] is True


class TestAudioSetAnimalLabels:
    """Tests for the AudioSet animal label mapping."""

    def test_bird_labels_exist(self):
        bird_labels = [l for l, c in AUDIOSET_ANIMAL_LABELS.items() if c == "bird"]
        assert len(bird_labels) > 0

    def test_dog_labels_exist(self):
        dog_labels = [l for l, c in AUDIOSET_ANIMAL_LABELS.items() if c == "dog"]
        assert len(dog_labels) > 0

    def test_cat_labels_exist(self):
        cat_labels = [l for l, c in AUDIOSET_ANIMAL_LABELS.items() if c == "cat"]
        assert len(cat_labels) > 0

    def test_frog_labels_exist(self):
        frog_labels = [l for l, c in AUDIOSET_ANIMAL_LABELS.items() if c == "frog"]
        assert len(frog_labels) > 0

    def test_insect_labels_exist(self):
        insect_labels = [l for l, c in AUDIOSET_ANIMAL_LABELS.items() if c == "insect"]
        assert len(insect_labels) > 0


class TestAudioClassifier:
    """Tests for AudioClassifier class."""

    def test_init_defaults(self):
        classifier = AudioClassifier()
        assert classifier.panns_min_confidence == 0.3
        assert classifier.birdnet_min_confidence == 0.5
        assert classifier.lat is None
        assert classifier.lng is None

    def test_init_custom_params(self):
        classifier = AudioClassifier(
            panns_min_confidence=0.4,
            birdnet_min_confidence=0.7,
            lat=51.5,
            lng=-0.12,
        )
        assert classifier.panns_min_confidence == 0.4
        assert classifier.birdnet_min_confidence == 0.7
        assert classifier.lat == 51.5
        assert classifier.lng == -0.12

    def test_init_model_toggles(self):
        classifier = AudioClassifier(panns_enabled=False, birdnet_enabled=False)
        assert classifier._panns_available is False
        assert classifier._birdnet_available is False

    @pytest.fixture
    def wav_file(self):
        """Create a temporary .wav file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)

    def _make_classifier(self, panns=True, birdnet=True):
        """Create a classifier with availability flags set for testing."""
        classifier = AudioClassifier(panns_min_confidence=0.5, birdnet_min_confidence=0.5)
        classifier._panns_available = panns
        classifier._birdnet_available = birdnet
        return classifier

    def test_classify_no_animal_sound(self, wav_file):
        """When PANNs finds no animal sound, result should be empty."""
        classifier = self._make_classifier()

        mock_panns_result = {}  # No animal labels above threshold
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class is None
        assert result.sound_confidence is None
        assert result.has_sound is False

    def test_classify_dog_bark(self, wav_file):
        """When PANNs detects a dog bark, result should have dog class."""
        classifier = self._make_classifier()

        mock_panns_result = {"Dog": 0.85, "Animal": 0.9}
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class == "dog"
        assert result.sound_confidence == 0.85

    def test_classify_cat_meow(self, wav_file):
        """When PANNs detects a cat sound, result should have cat class."""
        classifier = self._make_classifier()

        mock_panns_result = {"Cat": 0.78, "Meow": 0.72}
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class == "cat"
        assert result.sound_confidence == 0.78

    def test_classify_frog_croak(self, wav_file):
        """When PANNs detects a frog croak, result should have frog class."""
        classifier = self._make_classifier()

        mock_panns_result = {"Frog": 0.65}
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class == "frog"
        assert result.sound_confidence == 0.65

    def test_classify_bird_with_birdnet_species(self, wav_file):
        """When PANNs detects bird and BirdNET identifies species."""
        classifier = self._make_classifier()

        mock_panns_result = {"Bird": 0.75, "Bird vocalization, bird call, bird song": 0.8}
        mock_birdnet_result = ("Turdus merula_Eurasian Blackbird", 0.91)

        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result), \
             patch.object(classifier, "_classify_birdnet", return_value=mock_birdnet_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class == "bird"
        assert result.sound_species == "Turdus merula_Eurasian Blackbird"
        assert result.sound_confidence == 0.91

    def test_classify_bird_birdnet_no_match(self, wav_file):
        """When PANNs detects bird but BirdNET has no match, use PANNs result."""
        classifier = self._make_classifier()

        mock_panns_result = {"Bird": 0.75}
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result), \
             patch.object(classifier, "_classify_birdnet", return_value=None):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class == "bird"
        assert result.sound_species is None
        assert result.sound_confidence == 0.75

    def test_classify_visual_bird_triggers_birdnet(self, wav_file):
        """When visual class is bird, BirdNET should run even if PANNs doesn't detect bird."""
        classifier = self._make_classifier()

        # PANNs doesn't detect bird, but some general animal sound
        mock_panns_result = {"Animal": 0.6}
        mock_birdnet_result = ("Parus major_Great Tit", 0.88)

        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result), \
             patch.object(classifier, "_classify_birdnet", return_value=mock_birdnet_result):
            result = classifier.classify_audio(wav_file, visual_animal_class="bird")

        assert result.sound_class == "bird"
        assert result.sound_species == "Parus major_Great Tit"
        assert result.sound_confidence == 0.88

    def test_classify_below_threshold(self, wav_file):
        """When all predictions are below threshold, result should be empty."""
        classifier = self._make_classifier()

        mock_panns_result = {"Dog": 0.3}  # Below 0.5 threshold
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class is None
        assert result.has_sound is False

    def test_classify_nonexistent_file(self):
        """Should return empty result for nonexistent file."""
        classifier = AudioClassifier()
        result = classifier.classify_audio(Path("/nonexistent/audio.wav"))

        assert result.sound_class is None
        assert result.has_sound is False

    def test_classify_panns_unavailable(self, wav_file):
        """When panns_inference is not installed, should degrade gracefully."""
        classifier = self._make_classifier(panns=False, birdnet=True)

        # With visual_animal_class=bird, should still try BirdNET
        mock_birdnet_result = ("Parus major_Great Tit", 0.88)
        with patch.object(classifier, "_classify_birdnet", return_value=mock_birdnet_result):
            result = classifier.classify_audio(wav_file, visual_animal_class="bird")

        assert result.sound_class == "bird"
        assert result.sound_species == "Parus major_Great Tit"

    def test_classify_birdnet_unavailable_uses_panns(self, wav_file):
        """When birdnet is not installed, should use PANNs result for birds."""
        classifier = self._make_classifier(panns=True, birdnet=False)

        mock_panns_result = {"Bird": 0.75}
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert result.sound_class == "bird"
        assert result.sound_species is None
        assert result.sound_confidence == 0.75

    def test_classify_both_unavailable(self, wav_file):
        """When both libraries are missing, should return empty result."""
        classifier = self._make_classifier(panns=False, birdnet=False)

        result = classifier.classify_audio(wav_file)

        assert result.sound_class is None
        assert result.has_sound is False

    def test_all_predictions_populated(self, wav_file):
        """The all_predictions field should contain the raw PANNs predictions."""
        classifier = self._make_classifier()

        mock_panns_result = {"Dog": 0.85, "Animal": 0.9, "Bark": 0.7}
        with patch.object(classifier, "_classify_panns", return_value=mock_panns_result):
            result = classifier.classify_audio(wav_file)

        assert len(result.all_predictions) > 0
