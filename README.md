# TIR Super-Resolution & Colorization Pipeline

**SAC / ISRO — Bharatiya Antariksh Hackathon 2026**

Two-stage deep learning pipeline for Landsat 9 Thermal Infrared imagery:
1. **Super-Resolution** — TIR 200m → 100m (×2 upscale) using an RRDB/ESRGAN generator
2. **Colorization** — TIR 100m → RGB 100m using a Pix2Pix cGAN (U-Net + PatchGAN)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set dataset path

Edit `src/config.py` and set `DATASET_ROOT` to point to your local dataset folder containing `sr/`, `colorization/`, and `metadata/` subdirectories.

### 3. Train Super-Resolution

```bash
# Stage 1 — Pixel-only L1 loss (~100 epochs)
python -m src.train_sr --stage 1

# Stage 2 — L1 + Perceptual + GAN + Physics (~100 epochs)
python -m src.train_sr --stage 2
```

### 4. Train Colorization

```bash
python -m src.train_colorization
```

### 5. Evaluate

```bash
python -m src.evaluate --task both
```

### 6. Run Inference

```bash
python -m src.infer --input path/to/scene_B10.tif --output output/
```

---

## Project Structure

```
src/
├── config.py              — Central configuration (paths, hyperparameters)
├── data/
│   ├── sr_dataset.py      — SR dataset loader (TIR 200m → 100m)
│   └── color_dataset.py   — Colorization dataset loader (TIR → RGB)
├── models/
│   ├── rrdb.py            — RRDB generator (ESRGAN-style)
│   ├── unet.py            — U-Net generator (Pix2Pix)
│   ├── discriminator.py   — PatchGAN discriminator
│   └── losses.py          — L1, VGG perceptual, GAN, SSIM, physics losses
├── train_sr.py            — Two-stage SR training
├── train_colorization.py  — Pix2Pix cGAN training
├── evaluate.py            — PSNR, SSIM, FID evaluation
├── infer.py               — End-to-end tiled inference + GeoTIFF output
└── utils.py               — Checkpointing, logging, visualization
```

---

## Dataset Format

```
dataset_current_repo_format/
├── sr/
│   ├── train/
│   │   ├── tir_200m/   ← .npy (256×256)
│   │   └── tir_100m/   ← .npy (512×512)
│   ├── val/
│   └── test/
├── colorization/
│   ├── train/
│   │   ├── tir_100m/   ← .npy (256×256)
│   │   └── rgb_100m/   ← .npy (3×256×256, R,G,B)
│   ├── val/
│   └── test/
└── metadata/
    ├── patch_metadata.csv
    └── scene_summary.csv
```

---

## Model Architectures

### Super-Resolution — RRDB (ESRGAN-style)
- 23 Residual-in-Residual Dense Blocks
- PixelShuffle ×2 upsampling
- Physics constraint: AvgPool(SR) ≈ LR (preserves radiometric fidelity)

### Colorization — Pix2Pix cGAN
- U-Net generator: 8-layer encoder/decoder with skip connections
- PatchGAN discriminator (70×70 receptive field)
- Instance normalisation, spectral normalisation
- Multi-loss: GAN + 100×L1 + 10×perceptual + 5×SSIM

---

## Inference Output

```
output/
└── model_outputs/
    ├── tir_superresolved_100m/
    │   └── <product_id>.tif      ← Single-band TIR (Kelvin)
    └── colorized_tir_100m/
        └── <product_id>.tif      ← 3-band BGR GeoTIFF
```

> **Note:** Colorized output uses BGR band order per challenge specification.

---

## Evaluation Targets

| Metric | Task | Target |
|--------|------|--------|
| PSNR | Super-Resolution | >30 dB |
| PSNR | Colorization | >25 dB |
| SSIM | Super-Resolution | >0.85 |
| FID | Colorization | <20 |
| Inference | Per tile | <100 ms |

---

## References

- [ESRGAN](https://arxiv.org/abs/1809.00219) — Wang et al., ECCV 2018 Workshops
- [Pix2Pix](https://arxiv.org/abs/1611.07004) — Isola et al., CVPR 2017
- [Challenge Repository](https://github.com/jugal-sac/IR-colorization-BAH2026)
- Landsat 9 Collection 2 Level-2 via Google Earth Engine
