"""
Colorization Training Script — Pix2Pix cGAN.

Loss = L_cGAN + 100 × L_L1 + 10 × L_perceptual + 5 × L_SSIM

Generator is updated twice per discriminator update for stability.

Usage
-----
    python -m src.train_colorization
    python -m src.train_colorization --resume color_latest.pth
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.config import (
    DATASET_ROOT,
    CHECKPOINT_DIR,
    LOG_DIR,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    USE_AMP,
    COLOR_BATCH_SIZE,
    COLOR_LR_G,
    COLOR_LR_D,
    COLOR_BETA1,
    COLOR_BETA2,
    COLOR_EPOCHS,
    COLOR_G_UPDATES_PER_D,
    COLOR_LOSS_WEIGHTS,
    UNET_USE_TANH,
)
from src.data.color_dataset import ColorizationDataset
from src.models.unet import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.models.losses import L1Loss, PerceptualLoss, GANLoss, SSIMLoss
from src.utils import (
    setup_logger,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    tensor_to_numpy,
    denormalize_rgb,
    create_color_comparison,
)

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ────────────────────────────────────────────────────────
#  Validation
# ────────────────────────────────────────────────────────

@torch.no_grad()
def validate(generator, val_loader, device, epoch, save_dir=None):
    generator.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for i, (tir, rgb_gt, names) in enumerate(val_loader):
        tir = tir.to(device)
        rgb_gt = rgb_gt.to(device)

        rgb_pred = generator(tir)

        # Denormalise to [0, 1] for metrics
        rgb_pred_01 = denormalize_rgb(rgb_pred).clamp(0, 1)
        rgb_gt_01 = denormalize_rgb(rgb_gt).clamp(0, 1)

        pred_np = tensor_to_numpy(rgb_pred_01)
        gt_np = tensor_to_numpy(rgb_gt_01)

        for j in range(pred_np.shape[0]):
            # Channels-last for skimage  (H, W, 3)
            p_hwc = pred_np[j].transpose(1, 2, 0)
            g_hwc = gt_np[j].transpose(1, 2, 0)
            p_val = psnr(g_hwc, p_hwc, data_range=1.0)
            s_val = ssim(g_hwc, p_hwc, data_range=1.0, channel_axis=2)
            psnr_meter.update(p_val)
            ssim_meter.update(s_val)

        # Save first batch visual
        if i == 0 and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tir_np = tensor_to_numpy(tir)
            create_color_comparison(
                tir_np[0, 0],
                pred_np[0],
                gt_np[0],
                save_path=os.path.join(save_dir, f"color_val_epoch{epoch:03d}.png"),
                title=f"Epoch {epoch} — PSNR {psnr_meter.avg:.2f} dB",
            )

    generator.train()
    return psnr_meter.avg, ssim_meter.avg


# ────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────

def train(args):
    logger = setup_logger("Colorization", LOG_DIR)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Dataset root: {DATASET_ROOT}")

    # Data
    train_ds = ColorizationDataset(DATASET_ROOT, "train", augment=True)
    val_ds = ColorizationDataset(DATASET_ROOT, "val", augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=COLOR_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=COLOR_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Models
    generator = UNetGenerator().to(DEVICE)
    discriminator = PatchGANDiscriminator().to(DEVICE)  # in_ch=4 (1 TIR + 3 RGB)

    opt_G = torch.optim.Adam(
        generator.parameters(), lr=COLOR_LR_G, betas=(COLOR_BETA1, COLOR_BETA2)
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=COLOR_LR_D, betas=(COLOR_BETA1, COLOR_BETA2)
    )
    scaler = GradScaler(enabled=USE_AMP)

    # Losses
    l1_fn = L1Loss().to(DEVICE)
    percep_fn = PerceptualLoss().to(DEVICE)
    gan_fn = GANLoss(mode="vanilla").to(DEVICE)
    ssim_fn = SSIMLoss().to(DEVICE)
    w = COLOR_LOSS_WEIGHTS

    # Resume
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=DEVICE)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_G.load_state_dict(ckpt["optimizer_G"])
        opt_D.load_state_dict(ckpt["optimizer_D"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    logger.info(f"Generator params: {g_params:,} | Discriminator params: {d_params:,}")
    logger.info(f"Training: {len(train_ds)} samples | Validation: {len(val_ds)} samples")
    logger.info(f"Loss weights: {w}")

    save_dir = os.path.join(LOG_DIR, "color_vis")

    for epoch in range(start_epoch, COLOR_EPOCHS):
        generator.train()
        discriminator.train()
        g_meter = AverageMeter()
        d_meter = AverageMeter()
        t0 = time.time()

        for tir, rgb_gt, _ in train_loader:
            tir = tir.to(DEVICE)
            rgb_gt = rgb_gt.to(DEVICE)

            # ── Discriminator update ──────────────────
            opt_D.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                rgb_pred = generator(tir).detach()
                real_pair = torch.cat([tir, rgb_gt], dim=1)   # [B, 4, 256, 256]
                fake_pair = torch.cat([tir, rgb_pred], dim=1)
                loss_D = 0.5 * (
                    gan_fn(discriminator(real_pair), True) +
                    gan_fn(discriminator(fake_pair), False)
                )
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            d_meter.update(loss_D.item())

            # ── Generator update (x G_UPDATES_PER_D) ─
            for _ in range(COLOR_G_UPDATES_PER_D):
                opt_G.zero_grad(set_to_none=True)
                with autocast(enabled=USE_AMP):
                    rgb_pred = generator(tir)
                    fake_pair = torch.cat([tir, rgb_pred], dim=1)

                    # For perceptual / SSIM, work in [0,1] space
                    pred_01 = denormalize_rgb(rgb_pred) if UNET_USE_TANH else rgb_pred
                    gt_01 = denormalize_rgb(rgb_gt) if UNET_USE_TANH else rgb_gt

                    loss_G = (
                        w["adversarial"] * gan_fn(discriminator(fake_pair), True) +
                        w["l1"] * l1_fn(rgb_pred, rgb_gt) +
                        w["perceptual"] * percep_fn(pred_01, gt_01) +
                        w["ssim"] * ssim_fn(pred_01, gt_01)
                    )
                scaler.scale(loss_G).backward()
                scaler.step(opt_G)
                scaler.update()
            g_meter.update(loss_G.item())

        elapsed = time.time() - t0

        # Validate
        val_psnr, val_ssim = validate(generator, val_loader, DEVICE, epoch, save_dir)

        logger.info(
            f"Epoch {epoch:03d}/{COLOR_EPOCHS} | "
            f"G {g_meter.avg:.6f} | D {d_meter.avg:.6f} | "
            f"Val PSNR {val_psnr:.2f} dB | Val SSIM {val_ssim:.4f} | "
            f"{elapsed:.1f}s"
        )

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        state = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_G": opt_G.state_dict(),
            "optimizer_D": opt_D.state_dict(),
            "best_psnr": best_psnr,
        }
        save_checkpoint(state, "color_latest.pth")
        if is_best:
            save_checkpoint(state, "color_best.pth")
            logger.info(f"  ★ New best PSNR: {best_psnr:.2f} dB")

    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f} dB")


# ────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TIR Colorization model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint filename to resume from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
