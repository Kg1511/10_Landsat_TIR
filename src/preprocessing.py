"""
Preprocessing statistics and normalization helpers.

The project trains on original sensor arrays, not visualization images. This
module stores the numeric transform used by datasets, evaluation, and final
inference so the same scaling is applied everywhere.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

import numpy as np

from src.config import DATASET_ROOT, PREPROCESS_STATS_PATH

if TYPE_CHECKING:
    import torch


EPS = 1e-8


@dataclass
class PreprocessStats:
    """Saved preprocessing contract for TIR and RGB arrays."""

    tir_mean: float = 300.0
    tir_std: float = 20.0
    tir_min: float = 250.0
    tir_max: float = 350.0
    rgb_min: float = 0.0
    rgb_max: float = 1.0
    tir_count: int = 0
    rgb_count: int = 0
    dataset_root: str = ""
    split: str = "train"
    created_at: str = ""
    source_files: dict = field(default_factory=dict)

    @property
    def tir_range(self) -> float:
        return max(float(self.tir_max) - float(self.tir_min), EPS)

    @property
    def rgb_range(self) -> float:
        return max(float(self.rgb_max) - float(self.rgb_min), EPS)

    def normalize_tir_array(self, arr: np.ndarray) -> np.ndarray:
        """Kelvin array to z-score model input."""
        return ((arr.astype(np.float32) - self.tir_mean) / max(self.tir_std, EPS)).astype(
            np.float32
        )

    def denormalize_tir_array(self, arr: np.ndarray) -> np.ndarray:
        """TIR z-score array back to Kelvin."""
        return (arr.astype(np.float32) * max(self.tir_std, EPS) + self.tir_mean).astype(
            np.float32
        )

    def denormalize_tir_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        return tensor * max(self.tir_std, EPS) + self.tir_mean

    def tir_to_display_array(self, arr: np.ndarray, is_normalized: bool = True) -> np.ndarray:
        """Map TIR to [0, 1] only for plots and metrics displays."""
        kelvin = self.denormalize_tir_array(arr) if is_normalized else arr.astype(np.float32)
        return np.clip((kelvin - self.tir_min) / self.tir_range, 0.0, 1.0).astype(np.float32)

    def tir_to_display_tensor(self, tensor: "torch.Tensor", is_normalized: bool = True) -> "torch.Tensor":
        kelvin = self.denormalize_tir_tensor(tensor) if is_normalized else tensor
        return ((kelvin - self.tir_min) / self.tir_range).clamp(0.0, 1.0)

    def normalize_rgb_array(self, arr: np.ndarray) -> np.ndarray:
        """RGB/reference array to [0, 1] using train-set range."""
        arr = (arr.astype(np.float32) - self.rgb_min) / self.rgb_range
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def denormalize_rgb_array(self, arr: np.ndarray) -> np.ndarray:
        """RGB [0, 1] array back to original reference scale."""
        return (arr.astype(np.float32) * self.rgb_range + self.rgb_min).astype(np.float32)

    def denormalize_rgb_tensor(self, tensor: "torch.Tensor") -> "torch.Tensor":
        return tensor * self.rgb_range + self.rgb_min

    def to_dict(self) -> dict:
        data = asdict(self)
        data["version"] = 1
        data["normalization"] = {
            "tir": "zscore_kelvin",
            "rgb": "minmax_train_range",
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessStats":
        fields = {name for name in cls.__dataclass_fields__}
        values = {key: value for key, value in data.items() if key in fields}
        return cls(**values)


DEFAULT_STATS = PreprocessStats()


def _npy_files(path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(path, "*.npy")))


def _unique(paths: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(os.path.abspath(path) for path in paths))


def _stream_numeric_stats(paths: list[str]) -> dict:
    count = 0
    total = 0.0
    total_sq = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    for path in paths:
        arr = np.load(path).astype(np.float64, copy=False)
        values = arr[np.isfinite(arr)]
        if values.size == 0:
            continue
        count += int(values.size)
        total += float(values.sum())
        total_sq += float(np.dot(values, values))
        min_val = min(min_val, float(values.min()))
        max_val = max(max_val, float(values.max()))

    if count == 0:
        raise ValueError("No finite values found while computing preprocessing stats.")

    mean = total / count
    variance = max(total_sq / count - mean * mean, EPS)
    return {
        "count": count,
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "min": min_val,
        "max": max_val,
    }


def collect_training_files(dataset_root: str = DATASET_ROOT, split: str = "train") -> tuple[list[str], list[str]]:
    """Return TIR and RGB training files used to compute preprocessing stats."""
    tir_files = _unique(
        _npy_files(os.path.join(dataset_root, "sr", split, "tir_200m"))
        + _npy_files(os.path.join(dataset_root, "sr", split, "tir_100m"))
        + _npy_files(os.path.join(dataset_root, "colorization", split, "tir_100m"))
    )
    rgb_files = _unique(
        _npy_files(os.path.join(dataset_root, "colorization", split, "rgb_100m"))
    )
    return tir_files, rgb_files


def compute_preprocess_stats(dataset_root: str = DATASET_ROOT, split: str = "train") -> PreprocessStats:
    """Compute TIR z-score and RGB min/max stats from the training split."""
    tir_files, rgb_files = collect_training_files(dataset_root, split)

    if not tir_files:
        raise FileNotFoundError(f"No TIR .npy files found under {dataset_root!r} for split {split!r}.")
    if not rgb_files:
        raise FileNotFoundError(f"No RGB .npy files found under {dataset_root!r} for split {split!r}.")

    tir = _stream_numeric_stats(tir_files)
    rgb = _stream_numeric_stats(rgb_files)

    return PreprocessStats(
        tir_mean=tir["mean"],
        tir_std=tir["std"],
        tir_min=tir["min"],
        tir_max=tir["max"],
        rgb_min=rgb["min"],
        rgb_max=rgb["max"],
        tir_count=tir["count"],
        rgb_count=rgb["count"],
        dataset_root=os.path.abspath(dataset_root),
        split=split,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_files={
            "tir": len(tir_files),
            "rgb": len(rgb_files),
        },
    )


def save_preprocess_stats(stats: PreprocessStats, path: str = PREPROCESS_STATS_PATH) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_preprocess_stats(path: str = PREPROCESS_STATS_PATH, strict: bool = False) -> PreprocessStats:
    """Load saved stats, or return documented defaults when no file exists."""
    if not os.path.isfile(path):
        if strict:
            raise FileNotFoundError(
                f"Preprocessing stats not found: {path}. Run `python -m src.preprocessing` first."
            )
        return DEFAULT_STATS

    with open(path, "r", encoding="utf-8") as f:
        return PreprocessStats.from_dict(json.load(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute preprocessing statistics from train .npy arrays")
    parser.add_argument("--dataset-root", default=DATASET_ROOT, help="Dataset folder containing sr/ and colorization/")
    parser.add_argument("--split", default="train", help="Split used to compute stats")
    parser.add_argument("--output", default=PREPROCESS_STATS_PATH, help="Output JSON path")
    args = parser.parse_args()

    stats = compute_preprocess_stats(args.dataset_root, args.split)
    out_path = save_preprocess_stats(stats, args.output)

    print(f"Saved preprocessing stats: {out_path}")
    print(f"TIR mean/std: {stats.tir_mean:.6f} / {stats.tir_std:.6f}")
    print(f"TIR min/max:  {stats.tir_min:.6f} / {stats.tir_max:.6f}")
    print(f"RGB min/max:  {stats.rgb_min:.6f} / {stats.rgb_max:.6f}")


if __name__ == "__main__":
    main()
