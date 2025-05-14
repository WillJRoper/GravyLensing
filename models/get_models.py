#!/usr/bin/env python3
"""
A script for generating PyTorch models to be used in GravyLensing.

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

Supported precisions (for TorchScript):
  - float32 (default)
  - float16

Outputs always go to `models/{model}_{format}_{precision}.pt` or `.onnx`
"""

import argparse
import os

import torch
from torch.jit import optimize_for_inference
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name == "deeplab":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    elif model_name == "lraspp":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval().to(device)
    return model


def prepare_torchscript(
    model: torch.nn.Module, dummy: torch.Tensor, precision: str
) -> torch.jit.ScriptModule:
    """
    Script (or trace) the model, then freeze and optimize it.
    Returns a frozen, optimized ScriptModule on the correct precision.
    """
    # Optionally convert to float16
    if precision == "float16":
        model = model.half()
        dummy = dummy.half()

    # Script the model
    scripted = torch.jit.script(model)

    # Freeze and optimize for inference
    frozen = torch.jit.freeze(scripted)
    optimized = optimize_for_inference(frozen)

    return optimized


def export_torchscript_scripted(model, dummy, path: str, precision: str):
    ts_opt = prepare_torchscript(model, dummy, precision)
    ts_opt.save(path)
    print(f"[✔] Scripted+Frozen+Optimized ({precision}) TorchScript saved to {path}")


def export_torchscript_traced(model, dummy, path: str, precision: str):
    # Optionally convert to float16 before tracing
    if precision == "float16":
        model = model.half()
        dummy = dummy.half()

    ts = torch.jit.trace(model, dummy, strict=False)
    frozen = torch.jit.freeze(ts)
    optimized = optimize_for_inference(frozen)
    optimized.save(path)
    print(f"[✔] Traced+Frozen+Optimized ({precision}) TorchScript saved to {path}")


def export_quantized(model, dummy, path: str, precision: str):
    assert precision == "float32", "Quantization only supports float32"
    torch.backends.quantized.engine = "qnnpack"
    qmodel = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    qscript = torch.jit.freeze(torch.jit.script(qmodel))
    qscript.save(path)
    print(f"[✔] Dynamic-quantized TorchScript saved to {path}")


def export_onnx(model, dummy, path: str, precision: str):
    assert precision == "float32", "ONNX export only supports float32"
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
        choices=["cpu", "cuda", "mps"],
        help="Device to load model on",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float16"],
        help="Precision for TorchScript exports (float16 may be faster on MPS)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width for dummy input",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height for dummy input",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model, device)

    os.makedirs("models", exist_ok=True)
    ext = "onnx" if args.format == "onnx" else "pt"
    out_path = os.path.join(
        "models",
        f"{args.model}_{args.format}_{args.precision}_{args.width}_{args.height}.{ext}",
    )

    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    if args.format == "torchscript-scripted":
        export_torchscript_scripted(model, dummy, out_path, args.precision)
    elif args.format == "torchscript-traced":
        export_torchscript_traced(model, dummy, out_path, args.precision)
    elif args.format == "quantized":
        export_quantized(model, dummy, out_path, args.precision)
    elif args.format == "onnx":
        export_onnx(model, dummy, out_path, args.precision)


if __name__ == "__main__":
    main()
