"""Shared utilities for training, checkpointing, logging, and visualization."""

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import CHECKPOINT_DIR, UNET_USE_TANH
from src.preprocessing import load_preprocess_stats


def setup_logger(name: str, log_dir: str = None):
    """Create a logger that writes to console and optionally a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s %(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_dir is not None and not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


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


def denormalize_tir(tensor, stats=None):
    """TIR z-score tensor back to Kelvin."""
    stats = stats or load_preprocess_stats()
    return stats.denormalize_tir_tensor(tensor)


def denormalize_rgb(tensor, stats=None, original_scale=False):
    """RGB model tensor to [0, 1], optionally back to original reference scale."""
    stats = stats or load_preprocess_stats()
    if UNET_USE_TANH:
        tensor = (tensor + 1.0) / 2.0
    if original_scale:
        return stats.denormalize_rgb_tensor(tensor)
    return tensor


def tensor_to_numpy(t):
    """Move tensor to CPU and NumPy for plotting or metrics."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def create_sr_comparison(lr, sr, hr, save_path=None, title="SR comparison"):
    """Plot LR, SR, and HR triplet for one sample in display range [0, 1]."""
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
    """Plot TIR, predicted RGB, and ground-truth RGB in display range [0, 1]."""
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
