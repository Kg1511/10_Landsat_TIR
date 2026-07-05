"""Train the notebook-aligned TinyViT colorization comparison model."""

import argparse
import os

from src.config import (
    CHECKPOINT_DIR,
    COLOR_BATCH_SIZE,
    COLOR_VIT_CHECKPOINT,
    DATASET_ROOT,
    VIT_EPOCHS,
    VIT_LR,
)
from src.data.color_dataset import LandsatColorDataset
from src.models.tiny_vit import TinyViTColorNet
from src.notebook_pipeline import make_loader, train_image_regression


def train(args):
    train_ds = LandsatColorDataset(args.dataset_root, "train", augment=True)
    val_ds = LandsatColorDataset(args.dataset_root, "val", augment=False, stats=train_ds.stats)
    train_loader = make_loader(train_ds, args.batch_size, True)
    val_loader = make_loader(val_ds, args.batch_size, False)

    history, path = train_image_regression(
        TinyViTColorNet(dim=128, depth=4, heads=4, patch=16),
        train_loader,
        val_loader,
        task="colorization_transformer",
        epochs=args.epochs,
        lr=args.lr,
        save_name=args.save_name or COLOR_VIT_CHECKPOINT,
        model_name="TinyViTColorNet",
        save_dir=args.save_dir,
        stats=train_ds.stats,
    )
    print(f"Saved best TinyViT checkpoint: {path}")
    print(f"Best val L1: {history['best_val_l1']}")


def main():
    parser = argparse.ArgumentParser(description="Train TinyViT TIR colorization model")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--save-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=VIT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=COLOR_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=VIT_LR)
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--resume", default=None, help="Accepted for backward compatibility; not used.")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
