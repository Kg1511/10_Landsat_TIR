"""
Colorization dataset loader.

Loads paired TIR 100 m -> RGB 100 m patches from original .npy sensor arrays.
TIR normalization and RGB scaling come from the saved preprocessing stats.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import COLOR_SIZE, DATASET_ROOT, PREPROCESS_STATS_PATH, UNET_USE_TANH
from src.preprocessing import PreprocessStats, load_preprocess_stats


class ColorizationDataset(Dataset):
    """Landsat 9 TIR to RGB colorization dataset."""

    def __init__(
        self,
        root=DATASET_ROOT,
        split="train",
        augment=True,
        stats: PreprocessStats = None,
        stats_path: str = PREPROCESS_STATS_PATH,
    ):
        self.tir_dir = os.path.join(root, "colorization", split, "tir_100m")
        self.rgb_dir = os.path.join(root, "colorization", split, "rgb_100m")
        self.augment = augment and (split == "train")
        self.use_tanh = UNET_USE_TANH
        self.stats = stats or load_preprocess_stats(stats_path)

        self.files = sorted(
            [os.path.basename(f) for f in glob.glob(os.path.join(self.tir_dir, "*.npy"))]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.tir_dir}. "
                "Check that DATASET_ROOT is set correctly in config.py."
            )

        print(f"[ColorizationDataset] {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def normalize_tir(self, arr: np.ndarray) -> np.ndarray:
        """Map original Kelvin values to saved z-score model input."""
        return self.stats.normalize_tir_array(arr)

    def normalize_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Scale RGB/reference values with train-set min/max."""
        arr = self.stats.normalize_rgb_array(arr)
        if self.use_tanh:
            arr = arr * 2.0 - 1.0
        return arr.astype(np.float32)

    @staticmethod
    def _augment_pair(tir: np.ndarray, rgb: np.ndarray):
        """Random horizontal flip, vertical flip, and 90-degree rotation."""
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

    def __getitem__(self, idx):
        fname = self.files[idx]

        tir = np.load(os.path.join(self.tir_dir, fname)).astype(np.float32)
        rgb = np.load(os.path.join(self.rgb_dir, fname)).astype(np.float32)

        assert tir.shape == (COLOR_SIZE, COLOR_SIZE), f"Bad TIR shape {tir.shape}"
        assert rgb.shape == (3, COLOR_SIZE, COLOR_SIZE), f"Bad RGB shape {rgb.shape}"

        tir = self.normalize_tir(tir)
        rgb = self.normalize_rgb(rgb)

        if self.augment:
            tir, rgb = self._augment_pair(tir, rgb)

        tir_t = torch.from_numpy(tir).unsqueeze(0)
        rgb_t = torch.from_numpy(rgb)

        return tir_t, rgb_t, fname
