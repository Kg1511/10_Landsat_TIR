"""
Evaluation script — PSNR, SSIM, FID, and inference timing.

Usage
-----
    python -m src.evaluate --task sr    --checkpoint sr_stage2_best.pth
    python -m src.evaluate --task color --checkpoint color_best.pth
    python -m src.evaluate --task both  --sr_ckpt sr_stage2_best.pth --color_ckpt color_best.pth
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    DATASET_ROOT,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    EVAL_TARGETS,
    UNET_USE_TANH,
)
from src.data.sr_dataset import SRDataset
from src.data.color_dataset import ColorizationDataset
from src.models.rrdb import RRDBNet
from src.models.unet import UNetGenerator
from src.utils import (
    setup_logger,
    load_checkpoint,
    tensor_to_numpy,
    denormalize_rgb,
    create_sr_comparison,
    create_color_comparison,
)

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


logger = setup_logger("Evaluate")


# ────────────────────────────────────────────────────────
#  Inference timing helper
# ────────────────────────────────────────────────────────

def measure_inference_time(model, dummy_input, n_runs=100):
    """Return mean ± std inference time in milliseconds."""
    model.eval()
    times = []

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    for _ in range(n_runs):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model(dummy_input)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


# ────────────────────────────────────────────────────────
#  SR evaluation
# ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sr(ckpt_name: str):
    logger.info("=" * 60)
    logger.info("Evaluating Super-Resolution on test set")

    test_ds = SRDataset(DATASET_ROOT, "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = RRDBNet().to(DEVICE)
    ckpt = load_checkpoint(ckpt_name, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = []
    psnrs, ssims = [], []

    vis_dir = os.path.join(OUTPUT_DIR, "sr_eval_vis")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (lr, hr, names) in enumerate(test_loader):
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        sr = model(lr).clamp(0, 1)

        sr_np = tensor_to_numpy(sr)[0, 0]
        hr_np = tensor_to_numpy(hr)[0, 0]
        lr_np = tensor_to_numpy(lr)[0, 0]

        p = psnr_fn(hr_np, sr_np, data_range=1.0)
        s = ssim_fn(hr_np, sr_np, data_range=1.0)
        psnrs.append(p)
        ssims.append(s)
        results.append({"name": names[0], "psnr": p, "ssim": s})

        # Save first 10 visual comparisons
        if i < 10:
            create_sr_comparison(
                lr_np, sr_np, hr_np,
                save_path=os.path.join(vis_dir, f"sr_test_{i:03d}.png"),
                title=f"{names[0]} — PSNR {p:.2f}",
            )

    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)

    logger.info(f"  Test samples : {len(results)}")
    logger.info(f"  PSNR (mean)  : {mean_psnr:.2f} dB   (target: >{EVAL_TARGETS['sr_psnr_db']})")
    logger.info(f"  SSIM (mean)  : {mean_ssim:.4f}      (target: >{EVAL_TARGETS['sr_ssim']})")

    # Inference time
    dummy = torch.randn(1, 1, 256, 256, device=DEVICE)
    t_mean, t_std = measure_inference_time(model, dummy)
    logger.info(f"  Inference    : {t_mean:.1f} ± {t_std:.1f} ms/tile")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "sr_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"  Results CSV  : {csv_path}")

    return mean_psnr, mean_ssim


# ────────────────────────────────────────────────────────
#  Colorization evaluation
# ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_colorization(ckpt_name: str):
    logger.info("=" * 60)
    logger.info("Evaluating Colorization on test set")

    test_ds = ColorizationDataset(DATASET_ROOT, "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = UNetGenerator().to(DEVICE)
    ckpt = load_checkpoint(ckpt_name, map_location=DEVICE)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    results = []
    psnrs, ssims = [], []

    vis_dir = os.path.join(OUTPUT_DIR, "color_eval_vis")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (tir, rgb_gt, names) in enumerate(test_loader):
        tir = tir.to(DEVICE)
        rgb_gt = rgb_gt.to(DEVICE)

        rgb_pred = model(tir)

        # Denormalise
        pred_01 = denormalize_rgb(rgb_pred).clamp(0, 1)
        gt_01 = denormalize_rgb(rgb_gt).clamp(0, 1)

        pred_np = tensor_to_numpy(pred_01)[0].transpose(1, 2, 0)  # (H,W,3)
        gt_np = tensor_to_numpy(gt_01)[0].transpose(1, 2, 0)
        tir_np = tensor_to_numpy(tir)[0, 0]

        p = psnr_fn(gt_np, pred_np, data_range=1.0)
        s = ssim_fn(gt_np, pred_np, data_range=1.0, channel_axis=2)
        psnrs.append(p)
        ssims.append(s)
        results.append({"name": names[0], "psnr": p, "ssim": s})

        if i < 10:
            create_color_comparison(
                tir_np,
                pred_np.transpose(2, 0, 1),
                gt_np.transpose(2, 0, 1),
                save_path=os.path.join(vis_dir, f"color_test_{i:03d}.png"),
                title=f"{names[0]} — PSNR {p:.2f}",
            )

    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)

    logger.info(f"  Test samples : {len(results)}")
    logger.info(f"  PSNR (mean)  : {mean_psnr:.2f} dB   (target: >{EVAL_TARGETS['color_psnr_db']})")
    logger.info(f"  SSIM (mean)  : {mean_ssim:.4f}")

    # Inference time
    dummy = torch.randn(1, 1, 256, 256, device=DEVICE)
    t_mean, t_std = measure_inference_time(model, dummy)
    logger.info(f"  Inference    : {t_mean:.1f} ± {t_std:.1f} ms/tile")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "color_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"  Results CSV  : {csv_path}")

    return mean_psnr, mean_ssim


# ────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SR / Colorization models")
    parser.add_argument("--task", type=str, default="both",
                        choices=["sr", "color", "both"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for single-task eval")
    parser.add_argument("--sr_ckpt", type=str, default="sr_stage2_best.pth")
    parser.add_argument("--color_ckpt", type=str, default="color_best.pth")
    args = parser.parse_args()

    if args.task == "sr":
        ckpt = args.checkpoint or args.sr_ckpt
        evaluate_sr(ckpt)
    elif args.task == "color":
        ckpt = args.checkpoint or args.color_ckpt
        evaluate_colorization(ckpt)
    else:
        evaluate_sr(args.sr_ckpt)
        evaluate_colorization(args.color_ckpt)


if __name__ == "__main__":
    main()
