"""
End-to-end inference for TIR super-resolution and colorization.

The pipeline loads original TIR sensor values, applies saved preprocessing
statistics, runs SR and colorization models, and writes arrays plus GeoTIFFs.
Visualization transforms are not used as model input.
"""

import argparse
import os
import time

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine, from_origin

from src.config import DEVICE, PREPROCESS_STATS_PATH, UNET_USE_TANH
from src.models.rrdb import RRDBNet
from src.models.tiny_vit import TinyViTColorNet
from src.models.unet import UNetGenerator
from src.preprocessing import load_preprocess_stats
from src.utils import load_checkpoint, setup_logger


logger = setup_logger("Inference")


def _gaussian_weights(tile_size: int, sigma_frac: float = 0.3):
    """Create a 2D Gaussian window for tile blending."""
    ax = np.arange(tile_size, dtype=np.float32) - tile_size / 2.0
    xx, yy = np.meshgrid(ax, ax)
    sigma = tile_size * sigma_frac
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def _start_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _pad_to_tile(arr: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    h, w = arr.shape
    pad_h = max(tile_size - h, 0)
    pad_w = max(tile_size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return arr, h, w
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge"), h, w


def _load_tir_input(input_path: str):
    """Load a single-band TIR input from GeoTIFF or .npy."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".npy":
        arr = np.load(input_path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D .npy TIR array, got shape {arr.shape}")
        profile = {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": "float32",
            "transform": from_origin(0, 0, 1, 1),
        }
        return arr, profile

    with rasterio.open(input_path) as src:
        arr = src.read(1).astype(np.float32)
        return arr, src.profile.copy()


def _sr_profile_from_input(profile: dict, height: int, width: int) -> dict:
    sr_profile = profile.copy()
    sr_profile.update(height=height, width=width, count=1, dtype="float32")

    transform = sr_profile.get("transform")
    if isinstance(transform, Affine):
        sr_profile["transform"] = Affine(
            transform.a / 2,
            transform.b,
            transform.c,
            transform.d,
            transform.e / 2,
            transform.f,
        )
    return sr_profile


def tile_predict_sr(model, tir_full, tile_size=256, overlap=32, device=DEVICE):
    """Run SR on overlapping normalized TIR tiles and blend the result."""
    tir_work, original_h, original_w = _pad_to_tile(tir_full, tile_size)
    h, w = tir_work.shape
    out_h, out_w = h * 2, w * 2
    result = np.zeros((out_h, out_w), dtype=np.float64)
    weight = np.zeros((out_h, out_w), dtype=np.float64)

    g_hr = _gaussian_weights(tile_size * 2)
    stride = tile_size - overlap
    model.eval()

    with torch.no_grad():
        for r in _start_positions(h, tile_size, stride):
            for c in _start_positions(w, tile_size, stride):
                patch = tir_work[r : r + tile_size, c : c + tile_size]
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                sr = model(t).cpu().numpy()[0, 0]

                out_r = r * 2
                out_c = c * 2
                result[out_r : out_r + tile_size * 2, out_c : out_c + tile_size * 2] += sr * g_hr
                weight[out_r : out_r + tile_size * 2, out_c : out_c + tile_size * 2] += g_hr

    sr_full = (result / np.maximum(weight, 1e-8)).astype(np.float32)
    return sr_full[: original_h * 2, : original_w * 2]


def tile_predict_color(model, tir_full, tile_size=256, overlap=32, device=DEVICE):
    """Run colorization on overlapping normalized TIR tiles and blend RGB [0, 1]."""
    tir_work, original_h, original_w = _pad_to_tile(tir_full, tile_size)
    h, w = tir_work.shape
    result = np.zeros((3, h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    g = _gaussian_weights(tile_size)
    stride = tile_size - overlap
    model.eval()

    with torch.no_grad():
        for r in _start_positions(h, tile_size, stride):
            for c in _start_positions(w, tile_size, stride):
                patch = tir_work[r : r + tile_size, c : c + tile_size]
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                rgb = model(t).cpu().numpy()[0]
                if UNET_USE_TANH:
                    rgb = (rgb + 1.0) / 2.0
                rgb = np.clip(rgb, 0.0, 1.0)

                result[:, r : r + tile_size, c : c + tile_size] += rgb * g[np.newaxis]
                weight[r : r + tile_size, c : c + tile_size] += g

    rgb_full = (result / np.maximum(weight[np.newaxis], 1e-8)).astype(np.float32)
    return rgb_full[:, :original_h, :original_w]


def run_inference(
    input_path: str,
    output_dir: str,
    sr_ckpt: str = "sr_stage2_best.pth",
    color_ckpt: str = "color_best.pth",
    color_model_name: str = "unet",
    stats_path: str = PREPROCESS_STATS_PATH,
):
    """Full inference: TIR input -> SR TIR Kelvin -> colorized RGB/BGR outputs."""
    product_id = os.path.splitext(os.path.basename(input_path))[0]
    logger.info(f"Input: {input_path}")
    logger.info(f"Product ID: {product_id}")

    if not os.path.isfile(stats_path):
        logger.warning(f"Preprocessing stats not found at {stats_path}; using documented defaults.")
    stats = load_preprocess_stats(stats_path)

    tir_raw, profile = _load_tir_input(input_path)
    logger.info(f"Input shape: {tir_raw.shape}")

    tir_norm = stats.normalize_tir_array(tir_raw)

    logger.info("Loading SR model...")
    sr_model = RRDBNet().to(DEVICE)
    sr_ckpt_data = load_checkpoint(sr_ckpt, map_location=DEVICE)
    sr_model.load_state_dict(sr_ckpt_data["model"])

    logger.info("Running super-resolution...")
    t0 = time.time()
    sr_norm = tile_predict_sr(sr_model, tir_norm, tile_size=256, overlap=32)
    sr_kelvin = stats.denormalize_tir_array(sr_norm)
    logger.info(f"SR done in {time.time() - t0:.1f}s; output shape: {sr_kelvin.shape}")

    model_root = os.path.join(output_dir, "model_outputs")
    sr_out_dir = os.path.join(model_root, "tir_superresolved_100m")
    color_out_dir = os.path.join(model_root, "colorized_tir_100m")
    array_out_dir = os.path.join(model_root, "arrays")
    os.makedirs(sr_out_dir, exist_ok=True)
    os.makedirs(color_out_dir, exist_ok=True)
    os.makedirs(array_out_dir, exist_ok=True)

    np.save(os.path.join(array_out_dir, f"{product_id}_pred_tir100m.npy"), sr_kelvin)

    sr_path = os.path.join(sr_out_dir, f"{product_id}.tif")
    sr_profile = _sr_profile_from_input(profile, sr_kelvin.shape[0], sr_kelvin.shape[1])
    with rasterio.open(sr_path, "w", **sr_profile) as dst:
        dst.write(sr_kelvin.astype(np.float32), 1)
    logger.info(f"Saved SR TIR: {sr_path}")

    logger.info("Loading colorization model...")
    if color_model_name == "tiny_vit":
        color_model = TinyViTColorNet().to(DEVICE)
        color_state_key = "model"
    else:
        color_model = UNetGenerator().to(DEVICE)
        color_state_key = "generator"
    color_ckpt_data = load_checkpoint(color_ckpt, map_location=DEVICE)
    color_model.load_state_dict(color_ckpt_data[color_state_key])

    logger.info("Running colorization...")
    t0 = time.time()
    rgb_01 = tile_predict_color(color_model, sr_norm, tile_size=256, overlap=32)
    rgb_original = stats.denormalize_rgb_array(rgb_01)
    logger.info(f"Colorization done in {time.time() - t0:.1f}s; output shape: {rgb_original.shape}")

    np.save(os.path.join(array_out_dir, f"{product_id}_pred_rgb_chw.npy"), rgb_original)

    bgr = rgb_original[[2, 1, 0], :, :]
    color_path = os.path.join(color_out_dir, f"{product_id}.tif")
    color_profile = sr_profile.copy()
    color_profile.update(count=3, dtype="float32")
    with rasterio.open(color_path, "w", **color_profile) as dst:
        for band_idx in range(3):
            dst.write(bgr[band_idx].astype(np.float32), band_idx + 1)
    logger.info(f"Saved colorized BGR: {color_path}")

    logger.info("Inference complete.")
    return sr_path, color_path


def main():
    parser = argparse.ArgumentParser(description="TIR SR + colorization inference")
    parser.add_argument("--input", type=str, required=True, help="Input TIR GeoTIFF or 2D .npy array")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--sr_ckpt", type=str, default="sr_stage2_best.pth", help="SR checkpoint filename")
    parser.add_argument("--color_ckpt", type=str, default="color_best.pth", help="Colorization checkpoint filename")
    parser.add_argument("--color_model", type=str, default="unet", choices=["unet", "tiny_vit"])
    parser.add_argument("--stats", type=str, default=PREPROCESS_STATS_PATH, help="preprocess_stats.json path")
    args = parser.parse_args()

    run_inference(
        args.input,
        args.output,
        args.sr_ckpt,
        args.color_ckpt,
        args.color_model,
        args.stats,
    )


if __name__ == "__main__":
    main()
