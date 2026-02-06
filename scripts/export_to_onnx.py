#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models


SUPPORTED_MODELS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


def load_model(model_name: str) -> torch.nn.Module:
    if model_name not in SUPPORTED_MODELS:
        print(f"Error: unsupported model '{model_name}'. Choose from: {list(SUPPORTED_MODELS.keys())}")
        sys.exit(1)

    model_fn, weights = SUPPORTED_MODELS[model_name]
    print(f"Loading {model_name} with IMAGENET1K_V1 weights...")
    model = model_fn(weights=weights)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def verify(model: torch.nn.Module, output_path: Path) -> None:
    import onnxruntime as ort

    print("Verifying ONNX model against PyTorch...")

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch_output = model(dummy_input)
    torch_class = torch_output.argmax(dim=1).item()

    session = ort.InferenceSession(str(output_path))
    ort_output = session.run(None, {"input": dummy_input.numpy()})
    ort_class = np.argmax(ort_output[0], axis=1).item()

    print(f"  PyTorch top-1 class: {torch_class}")
    print(f"  ONNX Runtime top-1 class: {ort_class}")

    if torch_class == ort_class:
        print("Verification passed: outputs match.")
    else:
        print("Verification FAILED: top-1 class indices differ.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a TorchVision model to ONNX format.")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model architecture to export (default: mobilenet_v3_small)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX file path (default: models/<model_name>.onnx)",
    )
    args = parser.parse_args()

    output_path = args.output or Path(f"models/{args.model}.onnx")

    model = load_model(args.model)
    export_onnx(model, output_path)
    verify(model, output_path)


if __name__ == "__main__":
    main()
