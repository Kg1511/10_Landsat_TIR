"""
Super-Resolution dataset loader.

Loads paired (TIR 200m → TIR 100m) patches from .npy files.
Applies consistent augmentation to input-target pairs.

Tensor shapes returned
----------------------
LR : [1, 256, 256]   (TIR at 200 m)
HR : [1, 512, 512]   (TIR at 100 m)
"""

import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    DATASET_ROOT,
    TIR_MIN,
    TIR_RANGE,
    SR_LR_SIZE,
    SR_HR_SIZE,
)


class SRDataset(Dataset):
    """Landsat 9 TIR super-resolution dataset (2× upscale)."""

    def __init__(self, root=DATASET_ROOT, split="train", augment=True):
        self.lr_dir = os.path.join(root, "sr", split, "tir_200m")
        self.hr_dir = os.path.join(root, "sr", split, "tir_100m")
        self.augment = augment and (split == "train")

        self.files = sorted(
            [os.path.basename(f) for f in glob.glob(os.path.join(self.lr_dir, "*.npy"))]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.lr_dir}. "
                f"Check that DATASET_ROOT is set correctly in config.py."
            )

        print(f"[SRDataset] {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    # ── normalisation ─────────────────────────────
    @staticmethod
    def normalize_tir(arr: np.ndarray) -> np.ndarray:
        """Map Kelvin [250, 350] → [0, 1]."""
        return np.clip((arr - TIR_MIN) / TIR_RANGE, 0.0, 1.0).astype(np.float32)

    # ── augmentation (applied identically to LR+HR) ──
    @staticmethod
    def _augment_pair(lr: np.ndarray, hr: np.ndarray):
        """Random horizontal flip, vertical flip, 90° rotation."""
        if random.random() > 0.5:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        if random.random() > 0.5:
            lr = np.rot90(lr, k=1).copy()
            hr = np.rot90(hr, k=1).copy()
        return lr, hr

    # ── item access ───────────────────────────────
    def __getitem__(self, idx):
        fname = self.files[idx]

        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)  # (256, 256)
        hr = np.load(os.path.join(self.hr_dir, fname)).astype(np.float32)  # (512, 512)

        # Sanity shape check
        assert lr.shape == (SR_LR_SIZE, SR_LR_SIZE), f"Bad LR shape {lr.shape}"
        assert hr.shape == (SR_HR_SIZE, SR_HR_SIZE), f"Bad HR shape {hr.shape}"

        # Normalise
        lr = self.normalize_tir(lr)
        hr = self.normalize_tir(hr)

        # Augment
        if self.augment:
            lr, hr = self._augment_pair(lr, hr)

        # Convert to tensors  →  [1, H, W]
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        hr_t = torch.from_numpy(hr).unsqueeze(0)

        return lr_t, hr_t, fname
