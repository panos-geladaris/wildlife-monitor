"""
TorchVision / ONNX Runtime model loading and inference for animal classification.

Supports two backends:
- ONNX Runtime (preferred, works on Raspberry Pi 4 aarch64)
- PyTorch/TorchVision (fallback for development)

Backend is auto-detected at runtime based on available packages and model files.
"""

import logging
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_TO_ANIMAL = {
    7: "bird", 8: "bird", 9: "bird", 10: "bird", 11: "bird",
    12: "bird", 13: "bird", 14: "bird", 15: "bird", 16: "bird",
    17: "bird", 18: "bird", 19: "bird", 20: "bird", 21: "bird",
    22: "bird", 23: "bird", 24: "bird",
    80: "bird", 81: "bird", 82: "bird", 83: "bird", 84: "bird",
    85: "bird", 86: "bird", 87: "bird", 88: "bird", 89: "bird",
    90: "bird", 91: "bird", 92: "bird", 93: "bird", 94: "bird",
    95: "bird", 96: "bird", 97: "bird", 98: "bird", 99: "bird",
    100: "bird", 127: "bird", 128: "bird", 129: "bird", 130: "bird",
    131: "bird", 132: "bird", 133: "bird", 134: "bird", 135: "bird",
    136: "bird", 137: "bird", 138: "bird", 139: "bird", 140: "bird",
    141: "bird", 142: "bird", 143: "bird", 144: "bird", 145: "bird",
    146: "bird",
    281: "cat", 282: "cat", 283: "cat", 284: "cat", 285: "cat",
    286: "cat", 287: "cat", 288: "cat", 289: "cat", 290: "cat",
    291: "cat", 292: "cat", 293: "cat",
    151: "dog", 152: "dog", 153: "dog", 154: "dog", 155: "dog",
    156: "dog", 157: "dog", 158: "dog", 159: "dog", 160: "dog",
    161: "dog", 162: "dog", 163: "dog", 164: "dog", 165: "dog",
    166: "dog", 167: "dog", 168: "dog", 169: "dog", 170: "dog",
    171: "dog", 172: "dog", 173: "dog", 174: "dog", 175: "dog",
    176: "dog", 177: "dog", 178: "dog", 179: "dog", 180: "dog",
    181: "dog", 182: "dog", 183: "dog", 184: "dog", 185: "dog",
    186: "dog", 187: "dog", 188: "dog", 189: "dog", 190: "dog",
    191: "dog", 192: "dog", 193: "dog", 194: "dog", 195: "dog",
    196: "dog", 197: "dog", 198: "dog", 199: "dog", 200: "dog",
    201: "dog", 202: "dog", 203: "dog", 204: "dog", 205: "dog",
    206: "dog", 207: "dog", 208: "dog", 209: "dog", 210: "dog",
    211: "dog", 212: "dog", 213: "dog", 214: "dog", 215: "dog",
    216: "dog", 217: "dog", 218: "dog", 219: "dog", 220: "dog",
    221: "dog", 222: "dog", 223: "dog", 224: "dog", 225: "dog",
    226: "dog", 227: "dog", 228: "dog", 229: "dog", 230: "dog",
    231: "dog", 232: "dog", 233: "dog", 234: "dog", 235: "dog",
    236: "dog", 237: "dog", 238: "dog", 239: "dog", 240: "dog",
    241: "dog", 242: "dog", 243: "dog", 244: "dog", 245: "dog",
    246: "dog", 247: "dog", 248: "dog", 249: "dog", 250: "dog",
    251: "dog", 252: "dog", 253: "dog", 254: "dog", 255: "dog",
    256: "dog", 257: "dog", 258: "dog", 259: "dog", 260: "dog",
    261: "dog", 262: "dog", 263: "dog", 264: "dog", 265: "dog",
    266: "dog", 267: "dog", 268: "dog", 269: "dog", 270: "dog",
    271: "dog", 273: "dog", 274: "dog", 275: "dog",
    335: "squirrel", 336: "squirrel",
    277: "fox", 278: "fox", 279: "fox", 280: "fox",
    330: "rabbit", 331: "rabbit", 332: "rabbit",
    334: "hedgehog",
    337: "beaver",
    338: "hamster",
    351: "deer", 352: "deer", 353: "deer",
    333: "mouse",
}

ANIMAL_CLASSES = [
    "bird", "cat", "dog", "squirrel", "fox", "rabbit",
    "deer", "hedgehog", "mouse", "hamster", "beaver", "unknown",
]

DEFAULT_ONNX_DIR = Path(__file__).resolve().parent.parent.parent / "models"

ONNX_MODEL_FILENAMES = {
    "mobilenet_v3_small": "mobilenet_v3_small.onnx",
    "mobilenet_v3_large": "mobilenet_v3_large.onnx",
    "resnet18": "resnet18.onnx",
}


def _imagenet_preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")

    w, h = image.size
    if w < h:
        new_w = 256
        new_h = round(h * 256 / w)
    else:
        new_h = 256
        new_w = round(w * 256 / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    w, h = image.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


class _OnnxBackend:

    def __init__(self, onnx_path: Path):
        self._onnx_path = onnx_path
        self._session = None
        self._input_name: Optional[str] = None

    def load(self):
        import onnxruntime as ort

        logger.info(f"Loading ONNX model: {self._onnx_path}")
        self._session = ort.InferenceSession(
            str(self._onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNX model loaded successfully")

    def predict(self, image: Image.Image) -> tuple[int, float]:
        input_array = _imagenet_preprocess(image)
        logits = self._session.run(None, {self._input_name: input_array})[0]
        probs = _softmax(logits[0])
        class_idx = int(np.argmax(probs))
        return class_idx, float(probs[class_idx])

    def predict_top_k(self, image: Image.Image, k: int = 5) -> list[tuple[int, float]]:
        input_array = _imagenet_preprocess(image)
        logits = self._session.run(None, {self._input_name: input_array})[0]
        probs = _softmax(logits[0])
        top_indices = np.argsort(probs)[-k:][::-1]
        return [(int(i), float(probs[i])) for i in top_indices]

    def unload(self):
        self._session = None
        logger.info("ONNX model unloaded")


class _TorchBackend:

    def __init__(self, model_name: str, device: str):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._transform = None

    def load(self):
        import torch
        from torchvision import models, transforms

        logger.info(f"Loading PyTorch model: {self._model_name}")

        if self._model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self._model = models.mobilenet_v3_small(weights=weights)
            self._transform = weights.transforms()
        elif self._model_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            self._model = models.mobilenet_v3_large(weights=weights)
            self._transform = weights.transforms()
        elif self._model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self._model = models.resnet18(weights=weights)
            self._transform = weights.transforms()
        else:
            raise ValueError(f"Unsupported model: {self._model_name}")

        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("PyTorch model loaded successfully")

    def _preprocess(self, image: Image.Image):
        import torch
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self._device)

    def predict(self, image: Image.Image) -> tuple[int, float]:
        import torch

        input_tensor = self._preprocess(image)
        with torch.no_grad():
            output = self._model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = probs.argmax().item()
        return class_idx, probs[class_idx].item()

    def predict_top_k(self, image: Image.Image, k: int = 5) -> list[tuple[int, float]]:
        import torch

        input_tensor = self._preprocess(image)
        with torch.no_grad():
            output = self._model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_indices = probs.topk(k)
        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    def unload(self):
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("PyTorch model unloaded")


def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


class ModelLoader:
    """
    Loads and manages models for inference.

    Auto-selects backend:
    - ONNX Runtime if available and .onnx model file exists (recommended for Pi)
    - PyTorch/TorchVision as fallback
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: Optional[str] = None,
        weights_path: Optional[Path] = None,
        onnx_model_path: Optional[Path] = None,
        backend: str = "auto",
    ):
        self.model_name = model_name
        self._backend_impl = None
        self._backend_type: Optional[str] = None

        if onnx_model_path is None:
            filename = ONNX_MODEL_FILENAMES.get(model_name)
            if filename:
                onnx_model_path = DEFAULT_ONNX_DIR / filename

        if backend == "auto":
            if _onnx_available() and onnx_model_path and onnx_model_path.exists():
                self._backend_type = "onnx"
                self._backend_impl = _OnnxBackend(onnx_model_path)
            elif _torch_available():
                self._backend_type = "torch"
                _device = device or "cpu"
                self._backend_impl = _TorchBackend(model_name, _device)
            else:
                raise RuntimeError(
                    "No inference backend available. Install onnxruntime "
                    "(recommended for Raspberry Pi) or torch + torchvision."
                )
        elif backend == "onnx":
            if not _onnx_available():
                raise RuntimeError("onnxruntime is not installed")
            if not onnx_model_path or not onnx_model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {onnx_model_path}. "
                    f"Run: python scripts/export_to_onnx.py --model {model_name}"
                )
            self._backend_type = "onnx"
            self._backend_impl = _OnnxBackend(onnx_model_path)
        elif backend == "torch":
            if not _torch_available():
                raise RuntimeError("torch/torchvision is not installed")
            _device = device or "cpu"
            self._backend_type = "torch"
            self._backend_impl = _TorchBackend(model_name, _device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'onnx', or 'torch'.")

        self.device = device or "cpu"
        logger.info(f"ModelLoader initialized: {model_name} (backend={self._backend_type})")

    @property
    def backend(self) -> str:
        return self._backend_type

    def load(self):
        self._backend_impl.load()
        return self

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return _imagenet_preprocess(image)

    def predict(self, image: Image.Image) -> tuple[int, float]:
        return self._backend_impl.predict(image)

    def predict_top_k(self, image: Image.Image, k: int = 5) -> list[tuple[int, float]]:
        return self._backend_impl.predict_top_k(image, k)

    def unload(self):
        if self._backend_impl:
            self._backend_impl.unload()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = ModelLoader()
    loader.load()
    print(f"Model loaded: {loader.model_name} (backend: {loader.backend})")
    print(f"Device: {loader.device}")

    dummy_image = Image.new("RGB", (224, 224), color="green")
    class_idx, confidence = loader.predict(dummy_image)
    print(f"Prediction: class {class_idx}, confidence {confidence:.4f}")

    if class_idx in IMAGENET_TO_ANIMAL:
        print(f"Detected animal: {IMAGENET_TO_ANIMAL[class_idx]}")
    else:
        print("No animal detected")

    loader.unload()
