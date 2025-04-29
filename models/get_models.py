"""A script for generating PyTorch models to be used in GravyLensing.

Load a chosen segmentation model and save it in one of several
high-performance formats for direct C++ inference via libtorch or ONNXRuntime.

Supported models:
  - deeplab       : DeepLabV3 MobileNetV3 Large (torchvision)
  - lraspp        : LR-ASPP MobileNetV3 Large (torchvision)

Supported formats:
  - torchscript-scripted
  - torchscript-traced
  - quantized          (dynamic quantization)
  - onnx               (opset 14, dynamic axes)

Outputs always go to `models/{model}_model.{ext}`

Example:
    python export_segmentation_models.py --model lraspp --format quantized
"""

import argparse
import os

import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate and return the chosen segmentation model on the given device.
    """
    if model_name == "deeplab":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)

    elif model_name == "lraspp":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    model.eval()
    return model


def export_torchscript_scripted(model, dummy, path: str):
    ts = torch.jit.script(model)
    ts.save(path)
    print(f"[✔] Scripted TorchScript saved to {path}")


def export_torchscript_traced(model, dummy, path: str):
    ts = torch.jit.trace(model, dummy, strict=False)
    ts.save(path)
    print(f"[✔] Traced TorchScript saved to {path}")


def export_quantized(model, dummy, path: str):
    torch.backends.quantized.engine = "qnnpack"
    qmodel = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    qscript = torch.jit.script(qmodel)
    qscript.save(path)
    print(f"[✔] Dynamic-quantized TorchScript saved to {path}")


def export_onnx(model, dummy, path: str):
    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=14,
        input_names=["input"],
        output_names=["out"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "out": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"[✔] ONNX model saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export segmentation models for C++ inference"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["deeplab", "lraspp"],
        help="Which segmentation model to export",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=[
            "torchscript-scripted",
            "torchscript-traced",
            "quantized",
            "onnx",
        ],
        help="Format for export",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load model on",
    )
    parser.add_argument(
        "--width", type=int, default=320, help="Width for dummy input (traced/onnx)"
    )
    parser.add_argument(
        "--height", type=int, default=320, help="Height for dummy input (traced/onnx)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device)

    # Prepare output path
    os.makedirs("models", exist_ok=True)
    ext = "onnx" if args.format == "onnx" else "pt"
    out_path = f"{args.model}_{args.format}_model.{ext}"

    # Dummy tensor for shape definitions
    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    # Dispatch export
    if args.format == "torchscript-scripted":
        export_torchscript_scripted(model, dummy, out_path)
    elif args.format == "torchscript-traced":
        export_torchscript_traced(model, dummy, out_path)
    elif args.format == "quantized":
        export_quantized(model, dummy, out_path)
    elif args.format == "onnx":
        export_onnx(model, dummy, out_path)


if __name__ == "__main__":
    main()
