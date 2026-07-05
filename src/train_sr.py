"""
Super-Resolution Training Script.

Two-stage training:
  Stage 1 — Pixel-only (L1 loss)            ~100 epochs
  Stage 2 — L1 + Perceptual + GAN + Physics  ~100 epochs

Usage
-----
    python -m src.train_sr                       # Stage 1 from scratch
    python -m src.train_sr --stage 2             # Stage 2 (loads best Stage 1 ckpt)
    python -m src.train_sr --resume sr_stage1_latest.pth
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
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
    SR_BATCH_SIZE,
    SR_LR_G,
    SR_BETA1,
    SR_BETA2,
    SR_STAGE1_EPOCHS,
    SR_STAGE2_EPOCHS,
    SR_STAGE1_LOSS_WEIGHTS,
    SR_STAGE2_LOSS_WEIGHTS,
    SR_PATIENCE,
)
from src.data.sr_dataset import SRDataset
from src.models.rrdb import RRDBNet
from src.models.discriminator import PatchGANDiscriminator
from src.models.losses import L1Loss, PerceptualLoss, GANLoss, PhysicsLoss
from src.utils import (
    setup_logger,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    tensor_to_numpy,
    create_sr_comparison,
)

# Evaluation metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ────────────────────────────────────────────────────────
#  Validation
# ────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, epoch, save_dir=None):
    """Run validation and return average PSNR/SSIM."""
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    stats = getattr(val_loader.dataset, "stats", None)

    for i, (lr, hr, names) in enumerate(val_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        sr = model(lr)

        # Per-sample metrics
        sr_kelvin = stats.denormalize_tir_tensor(sr)
        hr_kelvin = stats.denormalize_tir_tensor(hr)
        sr_np = tensor_to_numpy(sr_kelvin)
        hr_np = tensor_to_numpy(hr_kelvin)

        for j in range(sr_np.shape[0]):
            p = psnr(hr_np[j, 0], sr_np[j, 0], data_range=stats.tir_range)
            s = ssim(hr_np[j, 0], sr_np[j, 0], data_range=stats.tir_range)
            psnr_meter.update(p)
            ssim_meter.update(s)

        # Save first batch comparison
        if i == 0 and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            lr_np = tensor_to_numpy(lr)
            lr_display = stats.tir_to_display_array(lr_np[0, 0])
            sr_display = stats.tir_to_display_array(tensor_to_numpy(sr)[0, 0])
            hr_display = stats.tir_to_display_array(tensor_to_numpy(hr)[0, 0])
            create_sr_comparison(
                lr_display, sr_display, hr_display,
                save_path=os.path.join(save_dir, f"sr_val_epoch{epoch:03d}.png"),
                title=f"Epoch {epoch} — PSNR {psnr_meter.avg:.2f} dB",
            )

    model.train()
    return psnr_meter.avg, ssim_meter.avg


# ────────────────────────────────────────────────────────
#  Stage 1 — Pixel-only training
# ────────────────────────────────────────────────────────

def train_stage1(args):
    logger = setup_logger("SR-Stage1", LOG_DIR)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Dataset root: {DATASET_ROOT}")

    # Data
    train_ds = SRDataset(DATASET_ROOT, "train", augment=True)
    val_ds = SRDataset(DATASET_ROOT, "val", augment=False)
    stats = train_ds.stats
    train_loader = DataLoader(
        train_ds, batch_size=SR_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=SR_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Model
    model = RRDBNet().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SR_LR_G, betas=(SR_BETA1, SR_BETA2)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SR_STAGE1_EPOCHS, eta_min=1e-7
    )
    scaler = GradScaler(enabled=USE_AMP)

    # Loss
    l1_loss = L1Loss().to(DEVICE)
    weights = SR_STAGE1_LOSS_WEIGHTS

    # Resume
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    patience_counter = 0
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Generator parameters: {total_params:,}")
    logger.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    save_dir = os.path.join(LOG_DIR, "sr_stage1_vis")

    for epoch in range(start_epoch, SR_STAGE1_EPOCHS):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()

        for lr, hr, _ in train_loader:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=USE_AMP):
                sr = model(lr)
                loss = weights["l1"] * l1_loss(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item())

        scheduler.step()
        elapsed = time.time() - t0

        # Validate
        val_psnr, val_ssim = validate(model, val_loader, DEVICE, epoch, save_dir)

        logger.info(
            f"Epoch {epoch:03d}/{SR_STAGE1_EPOCHS} | "
            f"Loss {loss_meter.avg:.6f} | "
            f"Val PSNR {val_psnr:.2f} dB | Val SSIM {val_ssim:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        # Checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "preprocess_stats": train_ds.stats.to_dict(),
        }
        save_checkpoint(state, "sr_stage1_latest.pth")
        if is_best:
            save_checkpoint(state, "sr_stage1_best.pth")
            logger.info(f"  ★ New best PSNR: {best_psnr:.2f} dB")

        if patience_counter >= SR_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch} (patience={SR_PATIENCE})")
            break

    logger.info(f"Stage 1 complete. Best PSNR: {best_psnr:.2f} dB")


# ────────────────────────────────────────────────────────
#  Stage 2 — Perceptual + GAN + Physics
# ────────────────────────────────────────────────────────

def train_stage2(args):
    logger = setup_logger("SR-Stage2", LOG_DIR)
    logger.info(f"Device: {DEVICE}")

    # Data
    train_ds = SRDataset(DATASET_ROOT, "train", augment=True)
    val_ds = SRDataset(DATASET_ROOT, "val", augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=SR_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=SR_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Models
    generator = RRDBNet().to(DEVICE)
    discriminator = PatchGANDiscriminator(in_ch=2).to(DEVICE)  # 1ch LR + 1ch HR/SR

    # Load Stage 1 best checkpoint into generator
    stage1_ckpt = args.resume or "sr_stage1_best.pth"
    ckpt = load_checkpoint(stage1_ckpt, map_location=DEVICE)
    generator.load_state_dict(ckpt["model"])
    logger.info(f"Loaded Stage 1 weights from {stage1_ckpt}")

    # Optimizers
    opt_G = torch.optim.Adam(generator.parameters(), lr=SR_LR_G, betas=(SR_BETA1, SR_BETA2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=SR_LR_G, betas=(SR_BETA1, SR_BETA2))
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=SR_STAGE2_EPOCHS, eta_min=1e-7)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=SR_STAGE2_EPOCHS, eta_min=1e-7)
    scaler = GradScaler(enabled=USE_AMP)

    # Losses
    l1_loss = L1Loss().to(DEVICE)
    percep_loss = PerceptualLoss().to(DEVICE)
    gan_loss = GANLoss(mode="vanilla").to(DEVICE)
    physics_loss = PhysicsLoss().to(DEVICE)
    w = SR_STAGE2_LOSS_WEIGHTS

    best_psnr = ckpt.get("best_psnr", 0.0)
    save_dir = os.path.join(LOG_DIR, "sr_stage2_vis")

    for epoch in range(SR_STAGE2_EPOCHS):
        generator.train()
        discriminator.train()
        g_meter = AverageMeter()
        d_meter = AverageMeter()
        t0 = time.time()

        for lr, hr, _ in train_loader:
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)

            # Upsample LR for discriminator conditioning
            lr_up = F.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)

            # ── Discriminator update ──────────────────
            opt_D.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                sr = generator(lr).detach()
                real_pair = torch.cat([lr_up, hr], dim=1)
                fake_pair = torch.cat([lr_up, sr], dim=1)
                loss_D = 0.5 * (
                    gan_loss(discriminator(real_pair), True) +
                    gan_loss(discriminator(fake_pair), False)
                )
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            d_meter.update(loss_D.item())

            # ── Generator update ──────────────────────
            opt_G.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                sr = generator(lr)
                fake_pair = torch.cat([lr_up, sr], dim=1)
                sr_display = stats.tir_to_display_tensor(sr)
                hr_display = stats.tir_to_display_tensor(hr)

                loss_G = (
                    w["l1"] * l1_loss(sr, hr) +
                    w["perceptual"] * percep_loss(sr_display, hr_display) +
                    w["adversarial"] * gan_loss(discriminator(fake_pair), True) +
                    w["physics"] * physics_loss(sr, lr)
                )
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()
            g_meter.update(loss_G.item())

        sched_G.step()
        sched_D.step()
        elapsed = time.time() - t0

        # Validate
        val_psnr, val_ssim = validate(generator, val_loader, DEVICE, epoch, save_dir)

        logger.info(
            f"Epoch {epoch:03d}/{SR_STAGE2_EPOCHS} | "
            f"G {g_meter.avg:.6f} | D {d_meter.avg:.6f} | "
            f"Val PSNR {val_psnr:.2f} dB | Val SSIM {val_ssim:.4f} | "
            f"{elapsed:.1f}s"
        )

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        state = {
            "epoch": epoch,
            "model": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_G": opt_G.state_dict(),
            "optimizer_D": opt_D.state_dict(),
            "best_psnr": best_psnr,
            "preprocess_stats": stats.to_dict(),
        }
        save_checkpoint(state, "sr_stage2_latest.pth")
        if is_best:
            save_checkpoint(state, "sr_stage2_best.pth")
            logger.info(f"  ★ New best PSNR: {best_psnr:.2f} dB")

    logger.info(f"Stage 2 complete. Best PSNR: {best_psnr:.2f} dB")


# ────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TIR Super-Resolution model")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage (1=pixel-only, 2=percep+GAN)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint filename to resume from")
    args = parser.parse_args()

    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)


if __name__ == "__main__":
    main()
