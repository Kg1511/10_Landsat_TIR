"""Notebook-aligned Landsat-9 TIR SR/colorization pipeline helpers.

This module mirrors the function names and safe-default behavior from
`landsat9_correct_full_model_notebook.ipynb`. Display stretching and previews are
strictly display-only and are never used as model input.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINT_DIR,
    COLOR_CNN_CHECKPOINT,
    COLOR_GAN_CHECKPOINT,
    COLOR_VIT_CHECKPOINT,
    DATASET_ROOT,
    DEVICE,
    DEVICE_TYPE,
    MODEL_COMPARISON_JSON,
    NUM_WORKERS,
    PIN_MEMORY,
    PREPROCESS_STATS_PATH,
    SR_CNN_CHECKPOINT,
)
from src.data.color_dataset import LandsatColorDataset
from src.data.sr_dataset import LandsatSRDataset
from src.models.discriminator import PatchDiscriminator
from src.models.simple_sr import ResidualBlock, SimpleSRNet
from src.models.tiny_vit import TinyViTColorNet
from src.models.unet import ColorUNet, ConvBlock
from src.preprocessing import (
    PreprocessStats,
    compute_rgb_min_max_from_train,
    compute_tir_mean_std_from_train,
    ensure_rgb_chw,
    load_preprocess_stats,
    safe_load_npy,
)


def list_npy(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.npy")))


def count_npy(folder: str) -> int:
    return len(list_npy(folder))


def summarize_array(arr: np.ndarray) -> Dict[str, object]:
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "finite": bool(np.isfinite(arr).all()),
    }


def check_pair(
    task: str,
    split: str,
    x_folder: str,
    y_folder: str,
    root: str = DATASET_ROOT,
    expected_x_shape=None,
    expected_y_shape=None,
):
    x_dir = os.path.join(root, task, split, x_folder)
    y_dir = os.path.join(root, task, split, y_folder)
    x_files = list_npy(x_dir)
    y_files = list_npy(y_dir)
    x_names = {os.path.basename(path) for path in x_files}
    y_names = {os.path.basename(path) for path in y_files}

    if not x_files:
        raise FileNotFoundError(f"No files found in {x_dir}")
    if x_names != y_names:
        raise ValueError(f"Pair mismatch in {task}/{split}")

    sample_name = sorted(x_names)[0]
    x = safe_load_npy(os.path.join(x_dir, sample_name))
    y = safe_load_npy(os.path.join(y_dir, sample_name))
    if task == "colorization":
        y = ensure_rgb_chw(y)

    if expected_x_shape is not None and tuple(x.shape) != tuple(expected_x_shape):
        raise ValueError(f"Expected x shape {expected_x_shape}, got {x.shape}")
    if expected_y_shape is not None and tuple(y.shape) != tuple(expected_y_shape):
        raise ValueError(f"Expected y shape {expected_y_shape}, got {y.shape}")

    return {
        "x_count": len(x_files),
        "y_count": len(y_files),
        "sample": sample_name,
        "x": summarize_array(x),
        "y": summarize_array(y),
    }


def _stats(stats: PreprocessStats | None = None) -> PreprocessStats:
    return stats or load_preprocess_stats()


def normalize_tir_tensor(x: torch.Tensor, stats: PreprocessStats | None = None) -> torch.Tensor:
    stats = _stats(stats)
    return (x - stats.tir_mean) / (stats.tir_std + 1e-8)


def denormalize_tir_tensor(x: torch.Tensor, stats: PreprocessStats | None = None) -> torch.Tensor:
    stats = _stats(stats)
    return x * (stats.tir_std + 1e-8) + stats.tir_mean


def rgb_min_tensor(device=None, dtype=torch.float32, stats: PreprocessStats | None = None):
    return torch.tensor(_stats(stats).rgb_min_array, dtype=dtype, device=device)


def rgb_max_tensor(device=None, dtype=torch.float32, stats: PreprocessStats | None = None):
    return torch.tensor(_stats(stats).rgb_max_array, dtype=dtype, device=device)


def normalize_rgb_tensor(y: torch.Tensor, stats: PreprocessStats | None = None) -> torch.Tensor:
    mn = rgb_min_tensor(device=y.device, dtype=y.dtype, stats=stats)
    mx = rgb_max_tensor(device=y.device, dtype=y.dtype, stats=stats)
    if y.ndim == 4:
        mn = mn.unsqueeze(0)
        mx = mx.unsqueeze(0)
    return torch.clamp((y - mn) / (mx - mn + 1e-8), 0.0, 1.0)


def denormalize_rgb_tensor(y: torch.Tensor, stats: PreprocessStats | None = None) -> torch.Tensor:
    mn = rgb_min_tensor(device=y.device, dtype=y.dtype, stats=stats)
    mx = rgb_max_tensor(device=y.device, dtype=y.dtype, stats=stats)
    if y.ndim == 4:
        mn = mn.unsqueeze(0)
        mx = mx.unsqueeze(0)
    return y * (mx - mn + 1e-8) + mn


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


def stretch_for_display(img: np.ndarray, low=2, high=98) -> np.ndarray:
    """Percentile stretch for display only; never use as model input."""
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if finite.sum() == 0:
        return np.zeros_like(img)
    p_low, p_high = np.percentile(img[finite], [low, high])
    return np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)


def rgb_chw_to_display(rgb_chw: np.ndarray) -> np.ndarray:
    """Convert CHW RGB to display HWC with display-only percentile stretching."""
    rgb_hwc = np.moveaxis(rgb_chw.astype(np.float32), 0, -1)
    return stretch_for_display(rgb_hwc)


def show_sr_sample(ds, idx=0):
    import matplotlib.pyplot as plt

    x, y, name = ds[idx]
    x_raw = denormalize_tir_tensor(x)[0].numpy()
    y_raw = denormalize_tir_tensor(y)[0].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(x_raw, cmap="inferno")
    axes[0].set_title(f"Input TIR 200m {x_raw.shape}")
    axes[1].imshow(y_raw, cmap="inferno")
    axes[1].set_title(f"Target TIR 100m {y_raw.shape}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name)
    return fig


def show_color_sample(ds, idx=0):
    import matplotlib.pyplot as plt

    x, y, name = ds[idx]
    x_raw = denormalize_tir_tensor(x)[0].numpy()
    y_raw = denormalize_rgb_tensor(y).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(x_raw, cmap="inferno")
    axes[0].set_title("Input TIR 100m")
    axes[1].imshow(rgb_chw_to_display(y_raw))
    axes[1].set_title("Target RGB display only")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name)
    return fig


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    mse = max(float(mse), 1e-12)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def save_checkpoint(model, path, model_name, task, history, extra=None, stats=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "task": task,
        "history": history,
        "preprocess_stats": _stats(stats).to_dict(),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_model_checkpoint(model, path, strict=True, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_sr(model, loader, stats=None):
    stats = _stats(stats)
    model.eval()
    total_abs_norm = 0.0
    total_sq_norm = 0.0
    total_pixels = 0
    for x, y, names in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        diff = pred - y
        total_abs_norm += diff.abs().sum().item()
        total_sq_norm += (diff**2).sum().item()
        total_pixels += y.numel()
    l1_norm = total_abs_norm / total_pixels
    mse_norm = total_sq_norm / total_pixels
    mae_kelvin = l1_norm * stats.tir_std
    rmse_kelvin = math.sqrt(mse_norm) * stats.tir_std
    return {
        "l1_normalized": l1_norm,
        "mse_normalized": mse_norm,
        "mae_kelvin": mae_kelvin,
        "rmse_kelvin": rmse_kelvin,
        "psnr_kelvin_range40": psnr_from_mse(rmse_kelvin**2, data_range=40.0),
    }


@torch.no_grad()
def evaluate_color(model, loader, stats=None):
    stats = _stats(stats)
    model.eval()
    total_abs_norm = 0.0
    total_sq_norm = 0.0
    total_abs_orig = 0.0
    total_sq_orig = 0.0
    total_pixels = 0
    for x, y, names in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        diff = pred - y
        total_abs_norm += diff.abs().sum().item()
        total_sq_norm += (diff**2).sum().item()
        total_pixels += y.numel()
        pred_orig = denormalize_rgb_tensor(pred, stats)
        y_orig = denormalize_rgb_tensor(y, stats)
        diff_orig = pred_orig - y_orig
        total_abs_orig += diff_orig.abs().sum().item()
        total_sq_orig += (diff_orig**2).sum().item()
    mse_norm = total_sq_norm / total_pixels
    return {
        "l1_rgb_normalized": total_abs_norm / total_pixels,
        "mse_rgb_normalized": mse_norm,
        "psnr_rgb_normalized": psnr_from_mse(mse_norm, data_range=1.0),
        "l1_rgb_original_scale": total_abs_orig / total_pixels,
        "mse_rgb_original_scale": total_sq_orig / total_pixels,
    }


def train_image_regression(
    model,
    train_loader,
    val_loader,
    task,
    epochs,
    lr,
    save_name,
    model_name,
    save_dir=CHECKPOINT_DIR,
    grad_clip=1.0,
    stats=None,
):
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.L1Loss()
    use_amp = DEVICE_TYPE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val = float("inf")
    save_path = os.path.join(save_dir, save_name)
    history = {"train_l1": [], "val_l1": [], "best_val_l1": None}

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        for x, y, names in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_sum += loss.item() * x.size(0)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, y, names in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                val_sum += criterion(model(x), y).item() * x.size(0)
        train_l1 = train_sum / len(train_loader.dataset)
        val_l1 = val_sum / len(val_loader.dataset)
        history["train_l1"].append(train_l1)
        history["val_l1"].append(val_l1)
        if val_l1 < best_val:
            best_val = val_l1
            history["best_val_l1"] = best_val
            save_checkpoint(
                model,
                save_path,
                model_name,
                task,
                history,
                extra={"best_val_l1": best_val},
                stats=stats,
            )
    return history, save_path


def visualize_sr_prediction(model, ds, idx=0):
    import matplotlib.pyplot as plt

    model.eval()
    x, y, name = ds[idx]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(DEVICE)).cpu()[0]
    x_raw = denormalize_tir_tensor(x)[0].numpy()
    y_raw = denormalize_tir_tensor(y)[0].numpy()
    p_raw = denormalize_tir_tensor(pred)[0].numpy()
    abs_error = np.abs(p_raw - y_raw)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [x_raw, y_raw, p_raw, abs_error],
        ["Input 200m", "Target 100m", "Predicted 100m", "Abs error"],
    ):
        ax.imshow(img, cmap="inferno" if title != "Abs error" else "magma")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(name)
    return fig


def visualize_color_prediction(model, ds, idx=0, title="Color Model"):
    import matplotlib.pyplot as plt

    model.eval()
    x, y, name = ds[idx]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(DEVICE)).cpu()[0]
    x_raw = denormalize_tir_tensor(x)[0].numpy()
    y_raw = denormalize_rgb_tensor(y).numpy()
    p_raw = denormalize_rgb_tensor(pred).numpy()
    err = np.abs(p_raw - y_raw).mean(axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(x_raw, cmap="inferno")
    axes[0].set_title("Input TIR")
    axes[1].imshow(rgb_chw_to_display(y_raw))
    axes[1].set_title("Target display only")
    axes[2].imshow(rgb_chw_to_display(p_raw))
    axes[2].set_title(f"{title} display only")
    axes[3].imshow(err, cmap="magma")
    axes[3].set_title("Mean RGB abs error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name)
    return fig


def save_gan_checkpoint(G, D, path, history, best_val_l1, stats=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict(),
            "model_name": "Pix2Pix_ColorUNet_Generator_PatchDiscriminator",
            "task": "colorization_gan",
            "history": history,
            "best_val_l1": best_val_l1,
            "preprocess_stats": _stats(stats).to_dict(),
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
    )


def train_pix2pix(
    G,
    D,
    train_loader,
    val_loader,
    epochs,
    lr=2e-4,
    lambda_l1=100.0,
    save_name=COLOR_GAN_CHECKPOINT,
    save_dir=CHECKPOINT_DIR,
    stats=None,
):
    G, D = G.to(DEVICE), D.to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    history = {"G_loss": [], "D_loss": [], "val_l1": []}
    best_val = float("inf")
    save_path = os.path.join(save_dir, save_name)
    for _epoch in range(1, epochs + 1):
        G.train()
        D.train()
        g_sum, d_sum = 0.0, 0.0
        for tir, real_rgb, names in train_loader:
            tir, real_rgb = tir.to(DEVICE), real_rgb.to(DEVICE)
            with torch.no_grad():
                fake_rgb = G(tir)
            d_loss = 0.5 * (
                bce(D(tir, real_rgb), torch.ones_like(D(tir, real_rgb)))
                + bce(D(tir, fake_rgb.detach()), torch.zeros_like(D(tir, fake_rgb.detach())))
            )
            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

            fake_rgb = G(tir)
            g_loss = bce(D(tir, fake_rgb), torch.ones_like(D(tir, fake_rgb))) + lambda_l1 * l1(
                fake_rgb, real_rgb
            )
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()
            g_sum += g_loss.item() * tir.size(0)
            d_sum += d_loss.item() * tir.size(0)

        G.eval()
        val_l1_sum = 0.0
        with torch.no_grad():
            for tir, real_rgb, names in val_loader:
                tir, real_rgb = tir.to(DEVICE), real_rgb.to(DEVICE)
                val_l1_sum += l1(G(tir), real_rgb).item() * tir.size(0)
        val_l1 = val_l1_sum / len(val_loader.dataset)
        history["G_loss"].append(g_sum / len(train_loader.dataset))
        history["D_loss"].append(d_sum / len(train_loader.dataset))
        history["val_l1"].append(val_l1)
        if val_l1 < best_val:
            best_val = val_l1
            save_gan_checkpoint(G, D, save_path, history, best_val, stats=stats)
    return history, save_path


def predict_sr_from_raw_array(sr_model, tir_200m_hw, stats=None):
    stats = _stats(stats)
    if tir_200m_hw.shape != (256, 256):
        raise ValueError(f"Expected 256x256 input, got {tir_200m_hw.shape}")
    sr_model.eval()
    x = torch.from_numpy(tir_200m_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    x = normalize_tir_tensor(x, stats).to(DEVICE)
    with torch.no_grad():
        pred_norm = sr_model(x).cpu()[0]
    pred_raw = denormalize_tir_tensor(pred_norm, stats)[0].numpy()
    return pred_raw.astype(np.float32)


def colorize_512_tir_by_tiles(color_model, tir_100m_512_hw, stats=None):
    stats = _stats(stats)
    if tir_100m_512_hw.shape != (512, 512):
        raise ValueError(f"Expected 512x512, got {tir_100m_512_hw.shape}")
    color_model.eval()
    out_rgb_norm = np.zeros((3, 512, 512), dtype=np.float32)
    tiles = [
        (0, 256, 0, 256),
        (0, 256, 256, 512),
        (256, 512, 0, 256),
        (256, 512, 256, 512),
    ]
    with torch.no_grad():
        for r1, r2, c1, c2 in tiles:
            tile = tir_100m_512_hw[r1:r2, c1:c2].astype(np.float32)
            x = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
            x = normalize_tir_tensor(x, stats).to(DEVICE)
            out_rgb_norm[:, r1:r2, c1:c2] = color_model(x).cpu()[0].numpy()
    out_rgb_raw = denormalize_rgb_tensor(torch.from_numpy(out_rgb_norm), stats).numpy()
    return out_rgb_raw.astype(np.float32)


def save_final_outputs(out_prefix, pred_tir_100m_512, pred_rgb_chw):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    tir_path = out_prefix + "_pred_tir100m_512.npy"
    rgb_path = out_prefix + "_pred_rgb_chw_original_scale.npy"
    np.save(tir_path, pred_tir_100m_512.astype(np.float32))
    np.save(rgb_path, pred_rgb_chw.astype(np.float32))
    tif_path = out_prefix + "_pred_bgr_chw.tif"
    try:
        import tifffile

        tifffile.imwrite(tif_path, pred_rgb_chw[[2, 1, 0], :, :].astype(np.float32))
    except Exception:
        tif_path = None
    preview_path = out_prefix + "_preview.png"
    rgb_disp = rgb_chw_to_display(pred_rgb_chw)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_disp)
    plt.axis("off")
    plt.title("Preview only: stretched for display")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=160)
    plt.close()
    return {"tir_npy": tir_path, "rgb_npy": rgb_path, "tif": tif_path, "preview": preview_path}


def load_best_color_model(choice="cnn", save_dir=CHECKPOINT_DIR):
    if choice == "cnn":
        return load_model_checkpoint(ColorUNet(base=32), os.path.join(save_dir, COLOR_CNN_CHECKPOINT))
    if choice == "gan":
        model = ColorUNet(base=32).to(DEVICE)
        ckpt = torch.load(os.path.join(save_dir, COLOR_GAN_CHECKPOINT), map_location=DEVICE)
        model.load_state_dict(ckpt["G_state_dict"])
        model.eval()
        return model
    if choice == "transformer":
        return load_model_checkpoint(
            TinyViTColorNet(dim=128, depth=4, heads=4, patch=16),
            os.path.join(save_dir, COLOR_VIT_CHECKPOINT),
        )
    raise ValueError("choice must be 'cnn', 'gan', or 'transformer'")


def run_batch_inference_on_folder(
    input_folder,
    output_folder,
    color_choice="cnn",
    max_files=None,
    save_dir=CHECKPOINT_DIR,
):
    os.makedirs(output_folder, exist_ok=True)
    sr_model = load_model_checkpoint(
        SimpleSRNet(channels=64, num_blocks=6),
        os.path.join(save_dir, SR_CNN_CHECKPOINT),
    )
    color_model = load_best_color_model(color_choice, save_dir=save_dir)
    files = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
    if max_files is not None:
        files = files[:max_files]
    records = []
    for path in files:
        raw = safe_load_npy(path)
        if raw.shape != (256, 256):
            records.append({"input": path, "skipped": True, "reason": f"bad shape {raw.shape}"})
            continue
        pred_tir = predict_sr_from_raw_array(sr_model, raw)
        pred_rgb = colorize_512_tir_by_tiles(color_model, pred_tir)
        prefix = os.path.join(output_folder, os.path.basename(path).replace(".npy", f"_{color_choice}"))
        saved = save_final_outputs(prefix, pred_tir, pred_rgb)
        records.append({"input": path, "output_prefix": prefix, **saved})
    manifest_path = os.path.join(output_folder, "inference_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    return records


__all__ = [
    "LandsatSRDataset",
    "LandsatColorDataset",
    "ResidualBlock",
    "SimpleSRNet",
    "ConvBlock",
    "ColorUNet",
    "PatchDiscriminator",
    "TinyViTColorNet",
    "list_npy",
    "count_npy",
    "summarize_array",
    "check_pair",
    "safe_load_npy",
    "compute_tir_mean_std_from_train",
    "compute_rgb_min_max_from_train",
    "normalize_tir_tensor",
    "denormalize_tir_tensor",
    "rgb_min_tensor",
    "rgb_max_tensor",
    "normalize_rgb_tensor",
    "denormalize_rgb_tensor",
    "ensure_rgb_chw",
    "make_loader",
    "stretch_for_display",
    "rgb_chw_to_display",
    "show_sr_sample",
    "show_color_sample",
    "count_params",
    "psnr_from_mse",
    "save_checkpoint",
    "load_model_checkpoint",
    "evaluate_sr",
    "evaluate_color",
    "train_image_regression",
    "visualize_sr_prediction",
    "visualize_color_prediction",
    "save_gan_checkpoint",
    "train_pix2pix",
    "predict_sr_from_raw_array",
    "colorize_512_tir_by_tiles",
    "save_final_outputs",
    "load_best_color_model",
    "run_batch_inference_on_folder",
]
