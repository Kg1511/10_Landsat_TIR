"""Train notebook-aligned colorization models."""

import argparse
import os

import torch

from src.config import (
    CHECKPOINT_DIR,
    COLOR_BATCH_SIZE,
    COLOR_CNN_CHECKPOINT,
    COLOR_EPOCHS,
    COLOR_GAN_CHECKPOINT,
    COLOR_LR_G,
    DATASET_ROOT,
)
from src.data.color_dataset import LandsatColorDataset
from src.models.discriminator import PatchDiscriminator
from src.models.unet import ColorUNet
from src.notebook_pipeline import (
    load_model_checkpoint,
    make_loader,
    train_image_regression,
    train_pix2pix,
)


def train_cnn(args):
    train_ds = LandsatColorDataset(args.dataset_root, "train", augment=True)
    val_ds = LandsatColorDataset(args.dataset_root, "val", augment=False, stats=train_ds.stats)
    train_loader = make_loader(train_ds, args.batch_size, True)
    val_loader = make_loader(val_ds, args.batch_size, False)

    history, path = train_image_regression(
        ColorUNet(base=32),
        train_loader,
        val_loader,
        task="colorization_cnn",
        epochs=args.epochs,
        lr=args.lr,
        save_name=args.save_name or COLOR_CNN_CHECKPOINT,
        model_name="ColorUNet",
        save_dir=args.save_dir,
        stats=train_ds.stats,
    )
    print(f"Saved best color CNN checkpoint: {path}")
    print(f"Best val L1: {history['best_val_l1']}")


def train_gan(args):
    train_ds = LandsatColorDataset(args.dataset_root, "train", augment=True)
    val_ds = LandsatColorDataset(args.dataset_root, "val", augment=False, stats=train_ds.stats)
    train_loader = make_loader(train_ds, args.batch_size, True)
    val_loader = make_loader(val_ds, args.batch_size, False)

    generator = ColorUNet(base=32)
    warm_start = args.warm_start or os.path.join(args.save_dir, COLOR_CNN_CHECKPOINT)
    if os.path.exists(warm_start):
        print(f"Warm-starting GAN generator from {warm_start}")
        generator = load_model_checkpoint(generator, warm_start)

    history, path = train_pix2pix(
        generator,
        PatchDiscriminator(in_channels=4, base=64),
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        lambda_l1=100.0,
        save_name=args.save_name or COLOR_GAN_CHECKPOINT,
        save_dir=args.save_dir,
        stats=train_ds.stats,
    )
    print(f"Saved best Pix2Pix checkpoint: {path}")
    print(f"Best val L1: {min(history['val_l1']) if history['val_l1'] else None}")


def main():
    parser = argparse.ArgumentParser(description="Train colorization model")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--save-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--mode", choices=["cnn", "gan"], default="cnn")
    parser.add_argument("--epochs", type=int, default=COLOR_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=COLOR_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=COLOR_LR_G)
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--resume", default=None, help="Accepted for backward compatibility; not used.")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(42)
    if args.mode == "gan":
        train_gan(args)
    else:
        train_cnn(args)


if __name__ == "__main__":
    main()
