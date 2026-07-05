"""Export trained models to TorchScript or ONNX."""

import argparse
import os

import torch

from src.config import (
    CHECKPOINT_DIR,
    COLOR_CNN_CHECKPOINT,
    COLOR_SIZE,
    COLOR_VIT_CHECKPOINT,
    DEVICE,
    OUTPUT_DIR,
    SR_CNN_CHECKPOINT,
    SR_LR_SIZE,
)
from src.models.simple_sr import SimpleSRNet
from src.models.tiny_vit import TinyViTColorNet
from src.models.unet import ColorUNet
from src.utils import load_checkpoint


def _load_model(task: str, checkpoint: str):
    if task == "sr":
        model = SimpleSRNet(channels=64, num_blocks=6).to(DEVICE)
        dummy = torch.randn(1, 1, SR_LR_SIZE, SR_LR_SIZE, device=DEVICE)
    elif task == "color":
        model = ColorUNet(base=32).to(DEVICE)
        dummy = torch.randn(1, 1, COLOR_SIZE, COLOR_SIZE, device=DEVICE)
    elif task == "vit":
        model = TinyViTColorNet(dim=128, depth=4, heads=4, patch=16).to(DEVICE)
        dummy = torch.randn(1, 1, COLOR_SIZE, COLOR_SIZE, device=DEVICE)
    else:
        raise ValueError(f"Unsupported export task: {task}")

    ckpt = load_checkpoint(checkpoint, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    suffix = "pt" if args.format == "torchscript" else "onnx"
    defaults = {
        "sr": SR_CNN_CHECKPOINT,
        "color": COLOR_CNN_CHECKPOINT,
        "vit": COLOR_VIT_CHECKPOINT,
    }
    checkpoint = args.checkpoint or os.path.join(CHECKPOINT_DIR, defaults[args.task])
    output = args.output or os.path.join(OUTPUT_DIR, "exports", f"{args.task}.{suffix}")

    if args.format == "torchscript":
        path = export_torchscript(args.task, checkpoint, output)
    else:
        path = export_onnx(args.task, checkpoint, output)
    print(f"Exported {args.task} model: {path}")


if __name__ == "__main__":
    main()
