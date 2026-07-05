"""Export trained models to TorchScript or ONNX."""

import argparse
import os

import torch

from src.config import COLOR_SIZE, DEVICE, OUTPUT_DIR, SR_LR_SIZE
from src.models.rrdb import RRDBNet
from src.models.tiny_vit import TinyViTColorNet
from src.models.unet import UNetGenerator
from src.utils import load_checkpoint


def _load_model(task: str, checkpoint: str):
    if task == "sr":
        model = RRDBNet().to(DEVICE)
        state_key = "model"
        dummy = torch.randn(1, 1, SR_LR_SIZE, SR_LR_SIZE, device=DEVICE)
    elif task == "color":
        model = UNetGenerator().to(DEVICE)
        state_key = "generator"
        dummy = torch.randn(1, 1, COLOR_SIZE, COLOR_SIZE, device=DEVICE)
    elif task == "vit":
        model = TinyViTColorNet().to(DEVICE)
        state_key = "model"
        dummy = torch.randn(1, 1, COLOR_SIZE, COLOR_SIZE, device=DEVICE)
    else:
        raise ValueError(f"Unsupported export task: {task}")

    ckpt = load_checkpoint(checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, dummy


def export_torchscript(task: str, checkpoint: str, output_path: str) -> str:
    model, dummy = _load_model(task, checkpoint)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(output_path)
    return output_path


def export_onnx(task: str, checkpoint: str, output_path: str) -> str:
    model, dummy = _load_model(task, checkpoint)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["tir"],
        output_names=["prediction"],
        dynamic_axes={
            "tir": {0: "batch"},
            "prediction": {0: "batch"},
        },
        opset_version=17,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export SR/colorization models")
    parser.add_argument("--task", choices=["sr", "color", "vit"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    suffix = "pt" if args.format == "torchscript" else "onnx"
    output = args.output or os.path.join(OUTPUT_DIR, "exports", f"{args.task}.{suffix}")

    if args.format == "torchscript":
        path = export_torchscript(args.task, args.checkpoint, output)
    else:
        path = export_onnx(args.task, args.checkpoint, output)
    print(f"Exported {args.task} model: {path}")


if __name__ == "__main__":
    main()
