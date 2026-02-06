"""Tests for the analysis module."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.analysis.model import (
    ModelLoader,
    IMAGENET_TO_ANIMAL,
    ANIMAL_CLASSES,
    _imagenet_preprocess,
    _softmax,
)
from src.analysis.classifier import AnimalClassifier, ClassificationResult
from src.analysis.frame_extractor import FrameExtractor, ExtractedFrame


class TestImageNetMapping:
    """Tests for ImageNet to animal class mapping."""
    
    def test_bird_classes_exist(self):
        """Test that bird classes are mapped."""
        bird_indices = [idx for idx, animal in IMAGENET_TO_ANIMAL.items() if animal == "bird"]
        assert len(bird_indices) > 0
    
    def test_cat_classes_exist(self):
        """Test that cat classes are mapped."""
        cat_indices = [idx for idx, animal in IMAGENET_TO_ANIMAL.items() if animal == "cat"]
        assert len(cat_indices) > 0
    
    def test_dog_classes_exist(self):
        """Test that dog classes are mapped."""
        dog_indices = [idx for idx, animal in IMAGENET_TO_ANIMAL.items() if animal == "dog"]
        assert len(dog_indices) > 0
    
    def test_animal_classes_list(self):
        """Test that all mapped animals are in ANIMAL_CLASSES."""
        mapped_animals = set(IMAGENET_TO_ANIMAL.values())
        for animal in mapped_animals:
            assert animal in ANIMAL_CLASSES, f"{animal} not in ANIMAL_CLASSES"
    
    def test_unknown_in_animal_classes(self):
        """Test that 'unknown' is in ANIMAL_CLASSES."""
        assert "unknown" in ANIMAL_CLASSES


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""
    
    def test_is_animal_true(self):
        """Test is_animal returns True for known animals."""
        result = ClassificationResult(animal_class="bird", confidence=0.8)
        assert result.is_animal is True
    
    def test_is_animal_false_unknown(self):
        """Test is_animal returns False for unknown."""
        result = ClassificationResult(animal_class="unknown", confidence=0.5)
        assert result.is_animal is False
    
    def test_is_animal_false_none(self):
        """Test is_animal returns False for None."""
        result = ClassificationResult(animal_class=None, confidence=0.0)
        assert result.is_animal is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ClassificationResult(
            animal_class="cat",
            confidence=0.92,
            frame_number=10,
        )
        d = result.to_dict()
        
        assert d["animal_class"] == "cat"
        assert d["confidence"] == 0.92
        assert d["frame_number"] == 10
        assert d["is_animal"] is True
        assert "timestamp" in d
    
    def test_default_values(self):
        """Test default values are set correctly."""
        result = ClassificationResult(animal_class="dog", confidence=0.5)
        
        assert result.frame_number is None
        assert result.frame_timestamp is None
        assert result.top_predictions == []
        assert result.timestamp is not None


class TestModelLoader:
    """Tests for ModelLoader class."""
    
    def test_init_default_device_cpu(self):
        """Test default device selection on CPU."""
        with patch("src.analysis.model._torch_available", return_value=True), \
             patch("src.analysis.model._onnx_available", return_value=False):
            loader = ModelLoader()
            assert loader.device == "cpu"
    
    def test_init_custom_model_name(self):
        """Test custom model name."""
        with patch("src.analysis.model._torch_available", return_value=True), \
             patch("src.analysis.model._onnx_available", return_value=False):
            loader = ModelLoader(model_name="resnet18")
            assert loader.model_name == "resnet18"
    
    def test_init_custom_device(self):
        """Test custom device selection."""
        with patch("src.analysis.model._torch_available", return_value=True), \
             patch("src.analysis.model._onnx_available", return_value=False):
            loader = ModelLoader(device="cpu")
            assert loader.device == "cpu"
    
    def test_backend_auto_prefers_onnx(self):
        """Test that auto backend prefers ONNX when available."""
        with patch("src.analysis.model._onnx_available", return_value=True), \
             tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = Path(f.name)
            try:
                loader = ModelLoader(onnx_model_path=onnx_path)
                assert loader.backend == "onnx"
            finally:
                onnx_path.unlink(missing_ok=True)
    
    def test_backend_auto_falls_back_to_torch(self):
        """Test that auto backend falls back to torch when ONNX unavailable."""
        with patch("src.analysis.model._onnx_available", return_value=False), \
             patch("src.analysis.model._torch_available", return_value=True):
            loader = ModelLoader()
            assert loader.backend == "torch"
    
    def test_backend_auto_no_backend_raises(self):
        """Test that auto backend raises when no backend is available."""
        with patch("src.analysis.model._onnx_available", return_value=False), \
             patch("src.analysis.model._torch_available", return_value=False):
            with pytest.raises(RuntimeError):
                ModelLoader()
    
    def test_backend_explicit_onnx_missing_file(self):
        """Test that explicit ONNX backend raises for missing file."""
        with patch("src.analysis.model._onnx_available", return_value=True):
            with pytest.raises(FileNotFoundError):
                ModelLoader(backend="onnx", onnx_model_path=Path("/nonexistent/model.onnx"))
    
    def test_backend_explicit_torch_unavailable(self):
        """Test that explicit torch backend raises when torch unavailable."""
        with patch("src.analysis.model._torch_available", return_value=False):
            with pytest.raises(RuntimeError):
                ModelLoader(backend="torch")


class TestImageNetPreprocess:
    """Tests for _imagenet_preprocess function."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        result = _imagenet_preprocess(Image.new("RGB", (640, 480)))
        assert result.shape == (1, 3, 224, 224)
    
    def test_output_dtype(self):
        """Test that output dtype is float32."""
        result = _imagenet_preprocess(Image.new("RGB", (640, 480)))
        assert result.dtype == np.float32
    
    def test_normalization_black_image(self):
        """Test normalization values for an all-black image."""
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        result = _imagenet_preprocess(image)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        expected = -mean / std
        for c in range(3):
            np.testing.assert_allclose(result[0, c, 112, 112], expected[c], atol=0.01)
    
    def test_non_square_image(self):
        """Test that non-square images produce correct output shape."""
        result = _imagenet_preprocess(Image.new("RGB", (300, 100)))
        assert result.shape == (1, 3, 224, 224)


class TestSoftmax:
    """Tests for _softmax function."""
    
    def test_softmax_uniform(self):
        """Test that uniform input gives uniform output."""
        result = _softmax(np.zeros(10))
        np.testing.assert_allclose(result, 0.1, atol=1e-7)
    
    def test_softmax_sums_to_one(self):
        """Test that softmax output sums to 1."""
        result = _softmax(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-7)
    
    def test_softmax_peak(self):
        """Test that softmax peak is at the correct index."""
        result = _softmax(np.array([10.0, 0.0, 0.0]))
        assert np.argmax(result) == 0


class TestAnimalClassifier:
    """Tests for AnimalClassifier class."""
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a classifier with mocked model."""
        with patch("src.analysis.model._torch_available", return_value=True), \
             patch("src.analysis.model._onnx_available", return_value=False), \
             patch.object(ModelLoader, "load") as mock_load, \
             patch.object(ModelLoader, "predict_top_k") as mock_predict:
            
            mock_load.return_value = MagicMock()
            mock_predict.return_value = [(500, 0.8), (501, 0.1)]
            
            classifier = AnimalClassifier(confidence_threshold=0.3)
            
            yield classifier, mock_predict
    
    def test_init(self):
        """Test classifier initialization."""
        with patch("src.analysis.model._torch_available", return_value=True), \
             patch("src.analysis.model._onnx_available", return_value=False):
            classifier = AnimalClassifier(
                model_name="mobilenet_v3_small",
                confidence_threshold=0.5,
            )
            assert classifier.confidence_threshold == 0.5
    
    def test_classify_image_no_animal(self, mock_classifier):
        """Test classification when no animal is detected."""
        classifier, mock_predict = mock_classifier
        
        # Mock returns non-animal classes
        mock_predict.return_value = [(500, 0.9), (501, 0.05)]
        
        image = Image.new("RGB", (224, 224))
        result = classifier.classify_image(image)
        
        assert result.animal_class == "unknown"
        assert result.is_animal is False
    
    def test_classify_image_bird(self, mock_classifier):
        """Test classification when a bird is detected."""
        classifier, mock_predict = mock_classifier
        
        # Class 15 is robin (bird)
        mock_predict.return_value = [(15, 0.85), (16, 0.1), (500, 0.05)]
        
        image = Image.new("RGB", (224, 224))
        result = classifier.classify_image(image)
        
        assert result.animal_class == "bird"
        assert result.confidence == 0.85
        assert result.is_animal is True
    
    def test_classify_image_cat(self, mock_classifier):
        """Test classification when a cat is detected."""
        classifier, mock_predict = mock_classifier
        
        # Class 281 is tabby cat
        mock_predict.return_value = [(281, 0.92), (282, 0.05)]
        
        image = Image.new("RGB", (224, 224))
        result = classifier.classify_image(image)
        
        assert result.animal_class == "cat"
        assert result.confidence == 0.92
    
    def test_classify_image_dog(self, mock_classifier):
        """Test classification when a dog is detected."""
        classifier, mock_predict = mock_classifier
        
        # Class 207 is golden retriever
        mock_predict.return_value = [(207, 0.88)]
        
        image = Image.new("RGB", (224, 224))
        result = classifier.classify_image(image)
        
        assert result.animal_class == "dog"
        assert result.confidence == 0.88
    
    def test_classify_image_below_threshold(self, mock_classifier):
        """Test classification below confidence threshold."""
        classifier, mock_predict = mock_classifier
        
        # Bird with low confidence
        mock_predict.return_value = [(15, 0.2)]  # Below 0.3 threshold
        
        image = Image.new("RGB", (224, 224))
        result = classifier.classify_image(image)
        
        assert result.animal_class == "unknown"
    
    def test_classify_image_with_frame_info(self, mock_classifier):
        """Test classification with frame metadata."""
        classifier, mock_predict = mock_classifier
        
        mock_predict.return_value = [(281, 0.9)]
        
        image = Image.new("RGB", (224, 224))
        frame_info = ExtractedFrame(
            image=image,
            frame_number=42,
            timestamp_seconds=1.5,
        )
        
        result = classifier.classify_image(image, frame_info=frame_info)
        
        assert result.frame_number == 42
        assert result.frame_timestamp == 1.5
    
    def test_classify_image_top_predictions(self, mock_classifier):
        """Test that top predictions are captured."""
        classifier, mock_predict = mock_classifier
        
        # Multiple animal predictions
        mock_predict.return_value = [
            (281, 0.6),   # cat
            (15, 0.2),    # bird
            (207, 0.1),   # dog
        ]
        
        image = Image.new("RGB", (224, 224))
        result = classifier.classify_image(image)
        
        assert len(result.top_predictions) > 0
        assert result.top_predictions[0] == ("cat", 0.6)


class TestFrameExtractor:
    """Tests for FrameExtractor class."""
    
    def test_init_file_not_found(self):
        """Test error when video file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            FrameExtractor(Path("/nonexistent/video.mp4"))
    
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)
    
    def test_init_with_file(self, mock_video_file):
        """Test initialization with existing file."""
        with patch("src.analysis.frame_extractor.CV2_AVAILABLE", False):
            extractor = FrameExtractor(mock_video_file)
            assert extractor.video_path == mock_video_file
    
    def test_context_manager(self, mock_video_file):
        """Test context manager protocol."""
        with patch("src.analysis.frame_extractor.CV2_AVAILABLE", False):
            with FrameExtractor(mock_video_file) as extractor:
                assert extractor is not None


class TestExtractedFrame:
    """Tests for ExtractedFrame dataclass."""
    
    def test_create_frame(self):
        """Test creating an extracted frame."""
        image = Image.new("RGB", (640, 480))
        frame = ExtractedFrame(
            image=image,
            frame_number=10,
            timestamp_seconds=0.5,
        )
        
        assert frame.frame_number == 10
        assert frame.timestamp_seconds == 0.5
        assert frame.image.size == (640, 480)
