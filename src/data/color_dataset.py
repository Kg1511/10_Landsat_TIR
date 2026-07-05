"""
Colorization dataset loader.

Loads paired (TIR 100m → RGB 100m) patches from .npy files.
Applies consistent augmentation to input-target pairs.

Tensor shapes returned
----------------------
TIR : [1, 256, 256]   (normalised to [0, 1])
RGB : [3, 256, 256]   (reflectance clipped to [0, 1])
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
    COLOR_SIZE,
    UNET_USE_TANH,
)


class ColorizationDataset(Dataset):
    """Landsat 9 TIR → RGB colorization dataset."""

    def __init__(self, root=DATASET_ROOT, split="train", augment=True):
        self.tir_dir = os.path.join(root, "colorization", split, "tir_100m")
        self.rgb_dir = os.path.join(root, "colorization", split, "rgb_100m")
        self.augment = augment and (split == "train")
        self.use_tanh = UNET_USE_TANH

        self.files = sorted(
            [os.path.basename(f) for f in glob.glob(os.path.join(self.tir_dir, "*.npy"))]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.tir_dir}. "
                f"Check that DATASET_ROOT is set correctly in config.py."
            )

        print(f"[ColorizationDataset] {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    # ── normalisation ─────────────────────────────
    @staticmethod
    def normalize_tir(arr: np.ndarray) -> np.ndarray:
        """Map Kelvin [250, 350] → [0, 1]."""
        return np.clip((arr - TIR_MIN) / TIR_RANGE, 0.0, 1.0).astype(np.float32)

    def normalize_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Clip reflectance to [0,1]; optionally rescale to [-1,1] for Tanh."""
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        if self.use_tanh:
            arr = arr * 2.0 - 1.0  # [0,1] → [-1,1]
        return arr

    # ── augmentation ──────────────────────────────
    @staticmethod
    def _augment_pair(tir: np.ndarray, rgb: np.ndarray):
        """Random horizontal flip, vertical flip, 90° rotation.

        tir : (H, W)
        rgb : (3, H, W)
        """
        if random.random() > 0.5:
            tir = np.flip(tir, axis=0).copy()
            rgb = np.flip(rgb, axis=1).copy()
        if random.random() > 0.5:
            tir = np.flip(tir, axis=1).copy()
            rgb = np.flip(rgb, axis=2).copy()
        if random.random() > 0.5:
            tir = np.rot90(tir, k=1).copy()
            rgb = np.rot90(rgb, k=1, axes=(1, 2)).copy()
        return tir, rgb

    # ── item access ───────────────────────────────
    def __getitem__(self, idx):
        fname = self.files[idx]

        tir = np.load(os.path.join(self.tir_dir, fname)).astype(np.float32)  # (256, 256)
        rgb = np.load(os.path.join(self.rgb_dir, fname)).astype(np.float32)  # (3, 256, 256)

        assert tir.shape == (COLOR_SIZE, COLOR_SIZE), f"Bad TIR shape {tir.shape}"
        assert rgb.shape == (3, COLOR_SIZE, COLOR_SIZE), f"Bad RGB shape {rgb.shape}"

        # Normalise
        tir = self.normalize_tir(tir)
        rgb = self.normalize_rgb(rgb)

        # Augment  (consistent transforms on both)
        if self.augment:
            tir, rgb = self._augment_pair(tir, rgb)

        # Convert to tensors
        tir_t = torch.from_numpy(tir).unsqueeze(0)  # [1, 256, 256]
        rgb_t = torch.from_numpy(rgb)                 # [3, 256, 256]

        return tir_t, rgb_t, fname
