"""
End-to-end inference pipeline.

Full flow
---------
Input TIR scene @200m  →  tile  →  SR model  →  stitch  →  TIR @100m
                                                         →  tile  →  Colorization model  →  stitch  →  RGB @100m

Outputs are saved as GeoTIFF with original CRS and geotransform.
Colorized TIFF uses BGR band order per the challenge spec.

Usage
-----
    python -m src.infer --input scene_B10.tif --output output/
    python -m src.infer --input scene_B10.tif --sr_ckpt sr_stage2_best.pth --color_ckpt color_best.pth
"""

import argparse
import os
import time

import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds

from src.config import (
    DEVICE,
    CHECKPOINT_DIR,
    SR_OUTPUT_DIR,
    COLOR_OUTPUT_DIR,
    TIR_MIN,
    TIR_RANGE,
    SR_LR_SIZE,
    SR_HR_SIZE,
    COLOR_SIZE,
    UNET_USE_TANH,
)
from src.models.rrdb import RRDBNet
from src.models.unet import UNetGenerator
from src.utils import load_checkpoint, setup_logger, denormalize_tir, denormalize_rgb


logger = setup_logger("Inference")


# ────────────────────────────────────────────────────────
#  Normalisation / Denormalisation
# ────────────────────────────────────────────────────────

def normalize_tir_np(arr):
    """Kelvin → [0, 1]."""
    return np.clip((arr - TIR_MIN) / TIR_RANGE, 0.0, 1.0).astype(np.float32)


def denormalize_tir_np(arr):
    """[0, 1] → Kelvin."""
    return arr * TIR_RANGE + TIR_MIN


def denormalize_rgb_np(arr):
    """[-1, 1] → [0, 1] if Tanh, else identity."""
    if UNET_USE_TANH:
        return (arr + 1.0) / 2.0
    return arr


# ────────────────────────────────────────────────────────
#  Gaussian blending weights for overlap removal
# ────────────────────────────────────────────────────────

def _gaussian_weights(tile_size: int, sigma_frac: float = 0.3):
    """Create 2-D Gaussian window for tile blending."""
    ax = np.arange(tile_size, dtype=np.float32) - tile_size / 2.0
    xx, yy = np.meshgrid(ax, ax)
    sigma = tile_size * sigma_frac
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g


# ────────────────────────────────────────────────────────
#  Tiled prediction helpers
# ────────────────────────────────────────────────────────

def tile_predict_sr(model, tir_full, tile_size=256, overlap=16, device=DEVICE):
    """Run SR model on overlapping tiles and stitch with Gaussian blending.

    Parameters
    ----------
    tir_full : ndarray (H, W) normalised [0, 1]  in LR space
    Returns
    -------
    sr_full  : ndarray (2*H, 2*W) normalised [0, 1]
    """
    H, W = tir_full.shape
    out_H, out_W = H * 2, W * 2
    result = np.zeros((out_H, out_W), dtype=np.float64)
    weight = np.zeros((out_H, out_W), dtype=np.float64)

    g_lr = _gaussian_weights(tile_size)
    g_hr = _gaussian_weights(tile_size * 2)

    stride = tile_size - overlap
    model.eval()

    with torch.no_grad():
        for r in range(0, H, stride):
            for c in range(0, W, stride):
                r_end = min(r + tile_size, H)
                c_end = min(c + tile_size, W)
                r_start = r_end - tile_size
                c_start = c_end - tile_size

                patch = tir_full[r_start:r_end, c_start:c_end]
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,256,256]
                sr = model(t).clamp(0, 1).cpu().numpy()[0, 0]  # [512, 512]

                # Output coordinates
                or_s = r_start * 2
                oc_s = c_start * 2
                or_e = or_s + tile_size * 2
                oc_e = oc_s + tile_size * 2

                result[or_s:or_e, oc_s:oc_e] += sr * g_hr
                weight[or_s:or_e, oc_s:oc_e] += g_hr

    weight = np.maximum(weight, 1e-8)
    return (result / weight).astype(np.float32)


def tile_predict_color(model, tir_full, tile_size=256, overlap=16, device=DEVICE):
    """Run colorization model on overlapping tiles.

    Parameters
    ----------
    tir_full : ndarray (H, W) normalised [0, 1]
    Returns
    -------
    rgb_full : ndarray (3, H, W)  in [0, 1]
    """
    H, W = tir_full.shape
    result = np.zeros((3, H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    g = _gaussian_weights(tile_size)
    stride = tile_size - overlap
    model.eval()

    with torch.no_grad():
        for r in range(0, H, stride):
            for c in range(0, W, stride):
                r_end = min(r + tile_size, H)
                c_end = min(c + tile_size, W)
                r_start = r_end - tile_size
                c_start = c_end - tile_size

                patch = tir_full[r_start:r_end, c_start:c_end]
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                rgb = model(t).cpu().numpy()[0]  # [3, 256, 256]

                # Denormalise from [-1,1] to [0,1] if Tanh
                rgb = denormalize_rgb_np(rgb)
                rgb = np.clip(rgb, 0, 1)

                result[:, r_start:r_end, c_start:c_end] += rgb * g[np.newaxis]
                weight[r_start:r_end, c_start:c_end] += g

    weight = np.maximum(weight, 1e-8)
    return (result / weight[np.newaxis]).astype(np.float32)


# ────────────────────────────────────────────────────────
#  Main inference pipeline
# ────────────────────────────────────────────────────────

def run_inference(
    input_path: str,
    output_dir: str,
    sr_ckpt: str = "sr_stage2_best.pth",
    color_ckpt: str = "color_best.pth",
):
    """Full inference: TIR → SR → Colorization → GeoTIFF output."""

    product_id = os.path.splitext(os.path.basename(input_path))[0]
    logger.info(f"Input: {input_path}")
    logger.info(f"Product ID: {product_id}")

    # ── Load input scene ──────────────────────────
    with rasterio.open(input_path) as src:
        tir_raw = src.read(1).astype(np.float32)  # single-band TIR
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        logger.info(f"Input shape: {tir_raw.shape}, CRS: {crs}")

    # Normalise
    tir_norm = normalize_tir_np(tir_raw)

    # ── SR Model ──────────────────────────────────
    logger.info("Loading SR model...")
    sr_model = RRDBNet().to(DEVICE)
    sr_ckpt_data = load_checkpoint(sr_ckpt, map_location=DEVICE)
    sr_model.load_state_dict(sr_ckpt_data["model"])
    sr_model.eval()

    logger.info("Running super-resolution...")
    t0 = time.time()
    sr_norm = tile_predict_sr(sr_model, tir_norm, tile_size=256, overlap=32)
    logger.info(f"  SR done in {time.time() - t0:.1f}s  |  Output shape: {sr_norm.shape}")

    # Denormalise to Kelvin
    sr_kelvin = denormalize_tir_np(sr_norm)

    # Save SR GeoTIFF
    sr_out_dir = os.path.join(output_dir, "model_outputs", "tir_superresolved_100m")
    os.makedirs(sr_out_dir, exist_ok=True)
    sr_path = os.path.join(sr_out_dir, f"{product_id}.tif")

    sr_profile = profile.copy()
    sr_profile.update(
        height=sr_kelvin.shape[0],
        width=sr_kelvin.shape[1],
        count=1,
        dtype="float32",
    )
    # Update transform for 2× resolution
    if transform is not None:
        sr_profile["transform"] = rasterio.Affine(
            transform.a / 2, transform.b, transform.c,
            transform.d, transform.e / 2, transform.f,
        )

    with rasterio.open(sr_path, "w", **sr_profile) as dst:
        dst.write(sr_kelvin, 1)
    logger.info(f"  Saved SR TIR: {sr_path}")

    # ── Colorization Model ────────────────────────
    logger.info("Loading colorization model...")
    color_model = UNetGenerator().to(DEVICE)
    color_ckpt_data = load_checkpoint(color_ckpt, map_location=DEVICE)
    color_model.load_state_dict(color_ckpt_data["generator"])
    color_model.eval()

    logger.info("Running colorization...")
    t0 = time.time()
    rgb_01 = tile_predict_color(color_model, sr_norm, tile_size=256, overlap=32)
    logger.info(f"  Colorization done in {time.time() - t0:.1f}s  |  Output shape: {rgb_01.shape}")

    # Convert RGB → BGR for challenge output format
    bgr = rgb_01[[2, 1, 0], :, :]

    # Save colorized GeoTIFF
    color_out_dir = os.path.join(output_dir, "model_outputs", "colorized_tir_100m")
    os.makedirs(color_out_dir, exist_ok=True)
    color_path = os.path.join(color_out_dir, f"{product_id}.tif")

    color_profile = sr_profile.copy()
    color_profile.update(count=3, dtype="float32")

    with rasterio.open(color_path, "w", **color_profile) as dst:
        for band_idx in range(3):
            dst.write(bgr[band_idx], band_idx + 1)
    logger.info(f"  Saved Colorized: {color_path}")

    logger.info("Inference complete!")
    return sr_path, color_path


# ────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TIR SR + Colorization Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input TIR GeoTIFF scene")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--sr_ckpt", type=str, default="sr_stage2_best.pth",
                        help="SR model checkpoint filename")
    parser.add_argument("--color_ckpt", type=str, default="color_best.pth",
                        help="Colorization model checkpoint filename")
    args = parser.parse_args()

    run_inference(args.input, args.output, args.sr_ckpt, args.color_ckpt)


if __name__ == "__main__":
    main()
