"""
Shared utilities for training, checkpointing, logging, and visualisation.
"""

import os
import logging
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import TIR_MIN, TIR_RANGE, CHECKPOINT_DIR, UNET_USE_TANH


# ────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: str = None):
    """Create a logger that writes to console and (optionally) file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s %(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ────────────────────────────────────────────────────────
#  Running average meter
# ────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the running mean of a scalar."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ────────────────────────────────────────────────────────
#  Checkpoint helpers
# ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, filename: str, ckpt_dir: str = CHECKPOINT_DIR):
    """Save a training checkpoint."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(filename: str, ckpt_dir: str = CHECKPOINT_DIR, map_location="cpu"):
    """Load a training checkpoint."""
    path = os.path.join(ckpt_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


# ────────────────────────────────────────────────────────
#  Denormalisation
# ────────────────────────────────────────────────────────

def denormalize_tir(tensor):
    """[0, 1] → Kelvin  [250, 350]."""
    return tensor * TIR_RANGE + TIR_MIN


def denormalize_rgb(tensor):
    """[-1, 1] → [0, 1]  (for Tanh output) or identity."""
    if UNET_USE_TANH:
        return (tensor + 1.0) / 2.0
    return tensor


# ────────────────────────────────────────────────────────
#  Visualisation
# ────────────────────────────────────────────────────────

def tensor_to_numpy(t):
    """Move tensor to CPU / numpy for plotting."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def create_sr_comparison(lr, sr, hr, save_path=None, title="SR comparison"):
    """Plot LR → SR → HR triplet for one sample.

    All inputs are 2-D numpy arrays in normalised [0,1] range.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, lbl in zip(axes, [lr, sr, hr], ["LR 200m", "SR 100m", "HR 100m (GT)"]):
        ax.imshow(img, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(lbl)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig


def create_color_comparison(tir, pred_rgb, gt_rgb, save_path=None, title="Colorization"):
    """Plot TIR → Predicted RGB → Ground-truth RGB.

    tir      : 2-D  [H, W]           normalised [0,1]
    pred_rgb : 3-D  [3, H, W] or [H, W, 3]  in [0,1]
    gt_rgb   : same as pred_rgb
    """
    if pred_rgb.ndim == 3 and pred_rgb.shape[0] == 3:
        pred_rgb = np.moveaxis(pred_rgb, 0, -1)
    if gt_rgb.ndim == 3 and gt_rgb.shape[0] == 3:
        gt_rgb = np.moveaxis(gt_rgb, 0, -1)

    pred_rgb = np.clip(pred_rgb, 0, 1)
    gt_rgb = np.clip(gt_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tir, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("TIR 100m (input)")
    axes[0].axis("off")

    axes[1].imshow(pred_rgb)
    axes[1].set_title("Predicted RGB")
    axes[1].axis("off")

    axes[2].imshow(gt_rgb)
    axes[2].set_title("Ground Truth RGB")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fig
