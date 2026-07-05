"""Train the TinyViT colorization comparison model."""

import argparse
import os
import time

import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.config import (
    DATASET_ROOT,
    DEVICE,
    LOG_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
    UNET_USE_TANH,
    USE_AMP,
    VIT_BATCH_SIZE,
    VIT_EPOCHS,
    VIT_LOSS_WEIGHTS,
    VIT_LR,
)
from src.data.color_dataset import ColorizationDataset
from src.models.losses import L1Loss, SSIMLoss
from src.models.tiny_vit import TinyViTColorNet
from src.utils import (
    AverageMeter,
    create_color_comparison,
    denormalize_rgb,
    load_checkpoint,
    save_checkpoint,
    setup_logger,
    tensor_to_numpy,
)


@torch.no_grad()
def validate(model, val_loader, device, epoch, save_dir=None):
    model.eval()
    stats = getattr(val_loader.dataset, "stats", None)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for i, (tir, rgb_gt, names) in enumerate(val_loader):
        tir = tir.to(device)
        rgb_gt = rgb_gt.to(device)

        rgb_pred = model(tir)
        pred_01 = denormalize_rgb(rgb_pred, stats).clamp(0, 1)
        gt_01 = denormalize_rgb(rgb_gt, stats).clamp(0, 1)

        pred_np = tensor_to_numpy(pred_01)
        gt_np = tensor_to_numpy(gt_01)

        for j in range(pred_np.shape[0]):
            pred_hwc = pred_np[j].transpose(1, 2, 0)
            gt_hwc = gt_np[j].transpose(1, 2, 0)
            psnr_meter.update(psnr(gt_hwc, pred_hwc, data_range=1.0))
            ssim_meter.update(ssim(gt_hwc, pred_hwc, data_range=1.0, channel_axis=2))

        if i == 0 and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tir_np = tensor_to_numpy(tir)
            create_color_comparison(
                stats.tir_to_display_array(tir_np[0, 0]),
                pred_np[0],
                gt_np[0],
                save_path=os.path.join(save_dir, f"tiny_vit_val_epoch{epoch:03d}.png"),
                title=f"Epoch {epoch} - PSNR {psnr_meter.avg:.2f} dB",
            )

    model.train()
    return psnr_meter.avg, ssim_meter.avg


def train(args):
    logger = setup_logger("TinyViT-Color", LOG_DIR)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Dataset root: {DATASET_ROOT}")

    train_ds = ColorizationDataset(DATASET_ROOT, "train", augment=True)
    val_ds = ColorizationDataset(DATASET_ROOT, "val", augment=False)
    stats = train_ds.stats

    train_loader = DataLoader(
        train_ds,
        batch_size=VIT_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VIT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = TinyViTColorNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=VIT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=VIT_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler(enabled=USE_AMP)

    l1_fn = L1Loss().to(DEVICE)
    ssim_fn = SSIMLoss().to(DEVICE)
    weights = VIT_LOSS_WEIGHTS

    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,}")
    logger.info(f"Loss weights: {weights}")

    save_dir = os.path.join(LOG_DIR, "tiny_vit_color_vis")

    for epoch in range(start_epoch, VIT_EPOCHS):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()

        for tir, rgb_gt, _ in train_loader:
            tir = tir.to(DEVICE)
            rgb_gt = rgb_gt.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                rgb_pred = model(tir)
                pred_01 = denormalize_rgb(rgb_pred, stats) if UNET_USE_TANH else rgb_pred
                gt_01 = denormalize_rgb(rgb_gt, stats) if UNET_USE_TANH else rgb_gt
                loss = (
                    weights["l1"] * l1_fn(rgb_pred, rgb_gt)
                    + weights["ssim"] * ssim_fn(pred_01, gt_01)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), n=tir.shape[0])

        scheduler.step()
        val_psnr, val_ssim = validate(model, val_loader, DEVICE, epoch, save_dir)
        logger.info(
            f"Epoch {epoch:03d}/{VIT_EPOCHS} | "
            f"Loss {loss_meter.avg:.6f} | "
            f"Val PSNR {val_psnr:.2f} dB | Val SSIM {val_ssim:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | {time.time() - t0:.1f}s"
        )

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "preprocess_stats": stats.to_dict(),
        }
        save_checkpoint(state, "color_tiny_transformer_latest.pth")
        if is_best:
            save_checkpoint(state, "color_tiny_transformer_best.pth")
            logger.info(f"New best PSNR: {best_psnr:.2f} dB")

    logger.info(f"TinyViT training complete. Best PSNR: {best_psnr:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Train TinyViT TIR colorization model")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint filename to resume from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
