"""Train the notebook-aligned SimpleSRNet baseline."""

import argparse
import os

from src.config import (
    CHECKPOINT_DIR,
    DATASET_ROOT,
    SR_BATCH_SIZE,
    SR_CNN_CHECKPOINT,
    SR_LR_G,
    SR_STAGE1_EPOCHS,
)
from src.data.sr_dataset import LandsatSRDataset
from src.models.rrdb import RRDBNet
from src.models.simple_sr import SimpleSRNet
from src.notebook_pipeline import make_loader, train_image_regression


def train(args):
    if args.model == "rrdb":
        model = RRDBNet()
        save_name = args.save_name or "sr_rrdb_optional_experiment.pth"
        model_name = "RRDBNet_optional_experiment"
    else:
        model = SimpleSRNet(channels=64, num_blocks=6)
        save_name = args.save_name or SR_CNN_CHECKPOINT
        model_name = "SimpleSRNet"

    train_ds = LandsatSRDataset(args.dataset_root, "train", augment=True)
    val_ds = LandsatSRDataset(args.dataset_root, "val", augment=False, stats=train_ds.stats)
    train_loader = make_loader(train_ds, args.batch_size, True)
    val_loader = make_loader(val_ds, args.batch_size, False)

    history, path = train_image_regression(
        model,
        train_loader,
        val_loader,
        task="sr",
        epochs=args.epochs,
        lr=args.lr,
        save_name=save_name,
        model_name=model_name,
        save_dir=args.save_dir,
        stats=train_ds.stats,
    )
    print(f"Saved best SR checkpoint: {path}")
    print(f"Best val L1: {history['best_val_l1']}")


def main():
    parser = argparse.ArgumentParser(description="Train SR model")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--save-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=SR_STAGE1_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=SR_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=SR_LR_G)
    parser.add_argument("--model", choices=["simple", "rrdb"], default="simple")
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--stage", type=int, default=None, help="Accepted for backward compatibility; ignored.")
    parser.add_argument("--resume", default=None, help="Accepted for backward compatibility; not used by this baseline.")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
