"""Dataset sanity checks for the Landsat TIR SR/colorization project."""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from src.config import COLOR_SIZE, DATASET_ROOT, OUTPUT_DIR, SR_HR_SIZE, SR_LR_SIZE
from src.preprocessing import compute_preprocess_stats, save_preprocess_stats


@dataclass
class ArraySummary:
    count: int
    min_value: float | None
    max_value: float | None
    finite: bool


def _npy_names(path: str) -> set[str]:
    if not os.path.isdir(path):
        return set()
    return {name for name in os.listdir(path) if name.endswith(".npy")}


def _summarize_arrays(path: str, expected_shape: tuple[int, ...]) -> tuple[ArraySummary, list[str]]:
    names = sorted(_npy_names(path))
    bad = []
    min_value = float("inf")
    max_value = float("-inf")
    all_finite = True

    for name in names:
        arr = np.load(os.path.join(path, name))
        if arr.shape != expected_shape:
            bad.append(f"{name}: shape {arr.shape}, expected {expected_shape}")
        finite = np.isfinite(arr)
        if not finite.all():
            all_finite = False
            bad.append(f"{name}: contains non-finite values")
        if finite.any():
            values = arr[finite]
            min_value = min(min_value, float(values.min()))
            max_value = max(max_value, float(values.max()))

    if not names:
        return ArraySummary(0, None, None, False), [f"No .npy files found in {path}"]

    return (
        ArraySummary(
            count=len(names),
            min_value=min_value,
            max_value=max_value,
            finite=all_finite,
        ),
        bad,
    )


def _check_pair_group(root: str, split: str, left: str, right: str) -> dict:
    left_names = _npy_names(os.path.join(root, split, left))
    right_names = _npy_names(os.path.join(root, split, right))
    return {
        "left_count": len(left_names),
        "right_count": len(right_names),
        "missing_left": sorted(right_names - left_names),
        "missing_right": sorted(left_names - right_names),
        "paired": left_names == right_names and len(left_names) > 0,
    }


def run_sanity_check(dataset_root: str = DATASET_ROOT, write_stats: bool = False) -> dict:
    report = {"dataset_root": os.path.abspath(dataset_root), "splits": {}, "errors": []}

    for split in ["train", "val", "test"]:
        sr_root = os.path.join(dataset_root, "sr")
        color_root = os.path.join(dataset_root, "colorization")

        sr_pairs = _check_pair_group(sr_root, split, "tir_200m", "tir_100m")
        color_pairs = _check_pair_group(color_root, split, "tir_100m", "rgb_100m")

        sr_lr_summary, sr_lr_bad = _summarize_arrays(
            os.path.join(sr_root, split, "tir_200m"), (SR_LR_SIZE, SR_LR_SIZE)
        )
        sr_hr_summary, sr_hr_bad = _summarize_arrays(
            os.path.join(sr_root, split, "tir_100m"), (SR_HR_SIZE, SR_HR_SIZE)
        )
        color_tir_summary, color_tir_bad = _summarize_arrays(
            os.path.join(color_root, split, "tir_100m"), (COLOR_SIZE, COLOR_SIZE)
        )
        rgb_summary, rgb_bad = _summarize_arrays(
            os.path.join(color_root, split, "rgb_100m"), (3, COLOR_SIZE, COLOR_SIZE)
        )

        split_errors = (
            sr_lr_bad
            + sr_hr_bad
            + color_tir_bad
            + rgb_bad
            + [f"SR missing target: {name}" for name in sr_pairs["missing_right"]]
            + [f"SR missing input: {name}" for name in sr_pairs["missing_left"]]
            + [f"Color missing target: {name}" for name in color_pairs["missing_right"]]
            + [f"Color missing input: {name}" for name in color_pairs["missing_left"]]
        )

        report["splits"][split] = {
            "sr_pairs": sr_pairs,
            "color_pairs": color_pairs,
            "sr_tir_200m": asdict(sr_lr_summary),
            "sr_tir_100m": asdict(sr_hr_summary),
            "color_tir_100m": asdict(color_tir_summary),
            "rgb_100m": asdict(rgb_summary),
            "errors": split_errors,
        }
        report["errors"].extend([f"{split}: {error}" for error in split_errors])

    if write_stats:
        stats = compute_preprocess_stats(dataset_root)
        stats_path = save_preprocess_stats(stats)
        report["preprocess_stats_path"] = stats_path
        report["preprocess_stats"] = stats.to_dict()

    return report


def main():
    parser = argparse.ArgumentParser(description="Check dataset pairs, shapes, and preprocessing stats")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--write-json", default=os.path.join(OUTPUT_DIR, "dataset_sanity_report.json"))
    parser.add_argument("--write-stats", action="store_true", help="Also compute checkpoints/preprocess_stats.json")
    args = parser.parse_args()

    report = run_sanity_check(args.dataset_root, write_stats=args.write_stats)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_json)), exist_ok=True)
    with open(args.write_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if report["errors"]:
        print(f"Dataset sanity check found {len(report['errors'])} issue(s).")
        for error in report["errors"][:20]:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Dataset sanity check passed. Report: {args.write_json}")


if __name__ == "__main__":
    main()
