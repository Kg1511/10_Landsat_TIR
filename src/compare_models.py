"""Compare saved notebook-aligned checkpoints and write metrics JSON."""

import argparse
import json
import os

import torch

from src.config import (
    CHECKPOINT_DIR,
    COLOR_BATCH_SIZE,
    COLOR_CNN_CHECKPOINT,
    COLOR_GAN_CHECKPOINT,
    COLOR_VIT_CHECKPOINT,
    DATASET_ROOT,
    MODEL_COMPARISON_JSON,
    SR_BATCH_SIZE,
    SR_CNN_CHECKPOINT,
)
from src.data.color_dataset import LandsatColorDataset
from src.data.sr_dataset import LandsatSRDataset
from src.models.simple_sr import SimpleSRNet
from src.models.tiny_vit import TinyViTColorNet
from src.models.unet import ColorUNet
from src.notebook_pipeline import evaluate_color, evaluate_sr, load_model_checkpoint, make_loader


def _exists(path):
    return os.path.exists(path)


def compare_saved_models(dataset_root=DATASET_ROOT, save_dir=CHECKPOINT_DIR, output_path=None):
    output_path = output_path or os.path.join(save_dir, MODEL_COMPARISON_JSON)
    metrics = {}

    sr_ds = LandsatSRDataset(dataset_root, "test", augment=False)
    color_ds = LandsatColorDataset(dataset_root, "test", augment=False, stats=sr_ds.stats)
    sr_loader = make_loader(sr_ds, SR_BATCH_SIZE, False)
    color_loader = make_loader(color_ds, COLOR_BATCH_SIZE, False)

    sr_path = os.path.join(save_dir, SR_CNN_CHECKPOINT)
    if _exists(sr_path):
        sr_model = load_model_checkpoint(SimpleSRNet(channels=64, num_blocks=6), sr_path)
        metrics["sr_cnn"] = evaluate_sr(sr_model, sr_loader, stats=sr_ds.stats)

    color_path = os.path.join(save_dir, COLOR_CNN_CHECKPOINT)
    if _exists(color_path):
        color_model = load_model_checkpoint(ColorUNet(base=32), color_path)
        metrics["color_cnn_unet"] = evaluate_color(color_model, color_loader, stats=color_ds.stats)

    gan_path = os.path.join(save_dir, COLOR_GAN_CHECKPOINT)
    if _exists(gan_path):
        gan_model = ColorUNet(base=32)
        ckpt = torch.load(gan_path, map_location="cpu")
        gan_model.load_state_dict(ckpt["G_state_dict"])
        metrics["color_pix2pix_gan"] = evaluate_color(gan_model, color_loader, stats=color_ds.stats)

    vit_path = os.path.join(save_dir, COLOR_VIT_CHECKPOINT)
    if _exists(vit_path):
        vit_model = load_model_checkpoint(TinyViTColorNet(dim=128, depth=4, heads=4, patch=16), vit_path)
        metrics["color_tiny_transformer"] = evaluate_color(vit_model, color_loader, stats=color_ds.stats)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare saved model checkpoints")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--save-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = compare_saved_models(args.dataset_root, args.save_dir, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
