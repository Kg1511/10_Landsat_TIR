"""
Super-resolution dataset loader.

Loads paired TIR 200 m -> TIR 100 m patches from original .npy sensor arrays.
The arrays are normalized with the saved preprocessing statistics; plotting
helpers handle visualization separately.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DATASET_ROOT, PREPROCESS_STATS_PATH, SR_HR_SIZE, SR_LR_SIZE
from src.preprocessing import PreprocessStats, load_preprocess_stats, safe_load_npy


class SRDataset(Dataset):
    """Landsat 9 TIR super-resolution dataset with 2x upscale pairs."""

    def __init__(
        self,
        root=DATASET_ROOT,
        split="train",
        augment=True,
        stats: PreprocessStats = None,
        stats_path: str = PREPROCESS_STATS_PATH,
    ):
        self.lr_dir = os.path.join(root, "sr", split, "tir_200m")
        self.hr_dir = os.path.join(root, "sr", split, "tir_100m")
        self.augment = augment and (split == "train")
        self.stats = stats or load_preprocess_stats(stats_path)

        self.files = sorted(
            [os.path.basename(f) for f in glob.glob(os.path.join(self.lr_dir, "*.npy"))]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.lr_dir}. "
                "Check that DATASET_ROOT is set correctly in config.py."
            )

        print(f"[SRDataset] {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def normalize_tir(self, arr: np.ndarray) -> np.ndarray:
        """Map original Kelvin values to saved z-score model input."""
        return self.stats.normalize_tir_array(arr)

    @staticmethod
    def _augment_pair(lr: np.ndarray, hr: np.ndarray):
        """Random horizontal flip, vertical flip, and 90-degree rotation."""
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

    def __getitem__(self, idx):
        fname = self.files[idx]

        lr = safe_load_npy(os.path.join(self.lr_dir, fname))
        hr = safe_load_npy(os.path.join(self.hr_dir, fname))

        assert lr.shape == (SR_LR_SIZE, SR_LR_SIZE), f"Bad LR shape {lr.shape}"
        assert hr.shape == (SR_HR_SIZE, SR_HR_SIZE), f"Bad HR shape {hr.shape}"

        lr = self.normalize_tir(lr)
        hr = self.normalize_tir(hr)

        if self.augment:
            lr, hr = self._augment_pair(lr, hr)

        lr_t = torch.from_numpy(lr).unsqueeze(0)
        hr_t = torch.from_numpy(hr).unsqueeze(0)

        return lr_t, hr_t, fname


class LandsatSRDataset(SRDataset):
    """Notebook-compatible class name for SR patches."""
