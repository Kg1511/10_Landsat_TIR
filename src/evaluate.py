"""Evaluate notebook-aligned checkpoints."""

import argparse
import json
import os

import torch

from src.compare_models import compare_saved_models
from src.config import (
    CHECKPOINT_DIR,
    COLOR_BATCH_SIZE,
    COLOR_CNN_CHECKPOINT,
    COLOR_GAN_CHECKPOINT,
    COLOR_VIT_CHECKPOINT,
    DATASET_ROOT,
    SR_BATCH_SIZE,
    SR_CNN_CHECKPOINT,
)
from src.data.color_dataset import LandsatColorDataset
from src.data.sr_dataset import LandsatSRDataset
from src.models.simple_sr import SimpleSRNet
from src.models.tiny_vit import TinyViTColorNet
from src.models.unet import ColorUNet
from src.notebook_pipeline import evaluate_color, evaluate_sr, load_model_checkpoint, make_loader


def evaluate_sr_checkpoint(dataset_root=DATASET_ROOT, save_dir=CHECKPOINT_DIR, checkpoint=None):
    ds = LandsatSRDataset(dataset_root, "test", augment=False)
    loader = make_loader(ds, SR_BATCH_SIZE, False)
    path = checkpoint or os.path.join(save_dir, SR_CNN_CHECKPOINT)
    model = load_model_checkpoint(SimpleSRNet(channels=64, num_blocks=6), path)
    return evaluate_sr(model, loader, stats=ds.stats)


def evaluate_color_checkpoint(kind="cnn", dataset_root=DATASET_ROOT, save_dir=CHECKPOINT_DIR, checkpoint=None):
    ds = LandsatColorDataset(dataset_root, "test", augment=False)
    loader = make_loader(ds, COLOR_BATCH_SIZE, False)
    if kind == "gan":
        path = checkpoint or os.path.join(save_dir, COLOR_GAN_CHECKPOINT)
        model = ColorUNet(base=32)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["G_state_dict"])
    elif kind == "vit":
        path = checkpoint or os.path.join(save_dir, COLOR_VIT_CHECKPOINT)
        model = load_model_checkpoint(TinyViTColorNet(dim=128, depth=4, heads=4, patch=16), path)
    else:
        path = checkpoint or os.path.join(save_dir, COLOR_CNN_CHECKPOINT)
        model = load_model_checkpoint(ColorUNet(base=32), path)
    return evaluate_color(model, loader, stats=ds.stats)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SR/colorization models")
    parser.add_argument("--task", choices=["sr", "color", "gan", "vit", "all", "both"], default="all")
    parser.add_argument("--dataset-root", "--root", dest="dataset_root", default=DATASET_ROOT)
    parser.add_argument("--save-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.task in ["all", "both"]:
        metrics = compare_saved_models(args.dataset_root, args.save_dir, args.output)
    elif args.task == "sr":
        metrics = {"sr_cnn": evaluate_sr_checkpoint(args.dataset_root, args.save_dir, args.checkpoint)}
    elif args.task == "gan":
        metrics = {"color_pix2pix_gan": evaluate_color_checkpoint("gan", args.dataset_root, args.save_dir, args.checkpoint)}
    elif args.task == "vit":
        metrics = {"color_tiny_transformer": evaluate_color_checkpoint("vit", args.dataset_root, args.save_dir, args.checkpoint)}
    else:
        metrics = {"color_cnn_unet": evaluate_color_checkpoint("cnn", args.dataset_root, args.save_dir, args.checkpoint)}

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
