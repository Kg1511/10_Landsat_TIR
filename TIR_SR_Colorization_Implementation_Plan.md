# SAC / ISRO — TIR Super-Resolution & Colorization Pipeline
### Full Implementation Plan · Landsat 9 Band B10 → High-res Colorized RGB

---

## Phase 0 — Environment Setup & Data Acquisition
**Category: Foundation**

### Step 1 — Python Environment
- Python 3.10+
- Install: `gdal`, `rasterio`, `tifffile`, `opencv-python`, `numpy`, `torch`, `torchvision`, `scikit-image`, `earthpy`
- GPU (CUDA 11.8+) strongly recommended

### Step 2 — Landsat 9 Data Download (USGS EarthExplorer)
- Download Landsat 9 Collection 2 Level-2 scenes
- Required bands: `B2` (Blue 30m), `B3` (Green 30m), `B4` (Red 30m), `B10` (TIR-1 100m, resampled to 30m by USGS)
- Target: 50–200+ geographically diverse scenes (urban, rural, coast, forest) for robust generalization

### Step 3 — Data Inventory & QA
- Filter scenes: cloud cover <10%, QA_PIXEL band pass, at least 4 seasons represented
- Organize into `raw/`, `processed/`, `patches/` directory structure

### Key Parameters

| | |
|---|---|
| **Input resolution** | TIR B10 — native 100m · RGB B2/B3/B4 — native 30m |
| **Target output** | TIR SR @ 100m (2× upscale from 200m) · Colorized RGB @ 100m |

---

## Phase 1 — Data Preprocessing & Patch Generation
**Category: Data**

### Step 1 — Merge RGB Bands
- Use GDAL / rasterio to stack B2, B3, B4 into a single 3-channel GeoTIFF
- Preserve CRS and geotransform metadata
- Apply Top-of-Atmosphere (ToA) reflectance scaling using Landsat 9 scale/offset factors from MTL metadata file

### Step 2 — Coregistration & Reprojection
- Reproject all bands to a common CRS (UTM zone of scene)
- Resample RGB to 100m grid using bilinear interpolation so TIR and RGB are spatially aligned pixel-to-pixel
- **Critical:** misalignment will poison training pairs

### Step 3 — Simulate Degraded TIR Inputs
**Super-resolution training pairs:**
- Take the native 100m TIR as HR ground truth
- Apply Gaussian blur (σ=1.5–2.0) + bilinear downscale by 2× → 200m LR input
- This gives paired (LR@200m, HR@100m) samples

**Colorization training pairs:**
- HR TIR at 100m as input; co-registered RGB at 100m as target
- Pair these spatially aligned crops

### Step 4 — Patch Extraction
- Extract overlapping tiles: `64×64` or `128×128` pixels at 100m resolution with 50% stride overlap
- Discard patches with >30% NaN/nodata
- Yields ~50,000–500,000 patches per task depending on scene count
- Save as `.npz` or HDF5 for fast loading

### Step 5 — Normalization
- Per-band min-max normalization to [0, 1] using dataset-wide percentile statistics (2nd–98th percentile robust scaling)
- Store normalization stats for inference-time denormalization
- For TIR, convert DN to brightness temperature (Kelvin) using Landsat calibration constants before normalizing

### Step 6 — Train / Val / Test Split
- **Scene-level split** (not patch-level) to prevent data leakage
- 70% train / 15% val / 15% test
- Ensure geographic diversity across splits
- Never mix patches from the same scene across splits

### Step 7 — Augmentation
- Random horizontal/vertical flip, 90° rotation, random crop
- Do **NOT** apply color jitter to TIR (temperature values are physically meaningful)
- Apply consistent transforms to input-target pairs simultaneously

---

## Phase 2 — Super-Resolution Model (TIR 200m → 100m)
**Category: SR Model**

### Architecture — RRDB-based SR (ESRGAN-style)
- Use a Residual-in-Residual Dense Block (RRDB) generator
- Input: single-channel TIR at 200m → Output: single-channel TIR at 100m
- Modify standard ESRGAN: reduce channels to 32/64 (satellite images are simpler than natural photos), single input channel, ~23 RRDB blocks

**Architecture flow:**
```
LR TIR 200m (1ch) → Conv3×3 → 23× RRDB → PixelShuffle ×2 → Conv3×3 → HR TIR 100m
```

### Alternative — SwinIR (Vision Transformer SR)
- For potentially better perceptual quality
- SwinIR with Swin Transformer blocks for classical SR (×2 scale)
- Heavier to train but captures long-range spatial dependencies — beneficial for large uniform regions (water bodies, agricultural fields) common in TIR imagery

### Loss Function (SR)
- **Stage 1 — Pixel loss only:** `L1 loss` (more robust than MSE for satellite data). Train to convergence (~100 epochs)
- **Stage 2 — Perceptual + GAN:** Add VGG perceptual loss (features from pretrained VGG19) + adversarial loss from PatchGAN discriminator
- Loss ratio: `L_total = L1 + 0.1×L_percep + 0.01×L_adv`

### Training Config
- Batch size 16–32
- Adam optimizer (lr=2e-4, β1=0.9, β2=0.99), cosine LR decay
- Train on 128×128 patches
- Mixed precision (FP16) for memory efficiency
- Early stopping on val PSNR with patience=20

### Physics-Informed Enhancement (Bonus)
- Add a physics constraint loss: the super-resolved TIR output, when spatially averaged (pool 2×2), must match the original LR TIR input pixel values
- Formula: `L_physics = MSE(AvgPool2d(SR_output, 2), LR_input)`
- Preserves radiometric fidelity and prevents hallucinated temperature values — critical for Earth observation applications

**Tags:** RRDB / ESRGAN · SwinIR (alt) · Physics constraint · L1 + perceptual + GAN

---

## Phase 3 — Colorization Model (TIR → RGB Translation)
**Category: Colorization**

### Architecture — Pix2Pix (cGAN Image-to-Image)
- Conditional GAN with U-Net generator (skip connections critical for preserving spatial structure) and PatchGAN discriminator
- Input: 1-channel SR TIR at 100m → Output: 3-channel RGB at 100m
- U-Net encoder: [64, 128, 256, 512, 512, 512, 512, 512] channels; decoder mirrors with skip connections

**Generator flow:**
```
SR TIR (1ch) → U-Net Encoder → Bottleneck → U-Net Decoder + skips → RGB (3ch)
```
**Discriminator:**
```
RGB + TIR (concatenated) → PatchGAN (real/fake)
```

### Alternative — Diffusion-based (ControlNet-style)
- Use a latent diffusion model conditioned on TIR input via a ControlNet adapter
- More compute-intensive but produces higher perceptual quality and handles multimodal outputs better
- Recommended if hardware allows: use Stable Diffusion v2 as base + train ControlNet adapter on TIR→RGB pairs

### Loss Function (Colorization)
| Loss term | Role |
|---|---|
| `L_cGAN` | Adversarial loss (discriminator binary cross-entropy) |
| `L_L1 × 100` | Pixel loss — controls mode collapse |
| `L_percep × 10` | VGG perceptual loss — texture/structure consistency |
| `L_SSIM × 5` | Structural similarity loss |
| **Total** | `L = L_cGAN + 100×L_L1 + 10×L_percep + 5×L_SSIM` |

### Training Config
- Batch size 8–16
- Adam (lr=2e-4 for G, 2e-4 for D), alternating G/D updates
- Train G twice per D update (stabilizes training)
- 200 epochs
- Apply spectral normalization to discriminator layers
- Use instance normalization in generator (better for image translation than batch norm)

### Hallucination Prevention (Qualitative Requirement)
- The discriminator's PatchGAN loss penalizes locally incoherent predictions
- Constrain generator output range via Tanh activation + clamp
- During inference: if SR output has temperature anomaly >3σ from scene mean in a patch, flag for visual review

**Tags:** Pix2Pix cGAN · Diffusion (alt) · U-Net + PatchGAN · Anti-hallucination flag

---

## Phase 4 — Physics-Informed Modeling (Bonus)
**Category: Bonus**

### Energy Conservation Constraint
- The SR model's output must satisfy: spatial integral of upscaled TIR ≈ spatial integral of LR TIR (energy is conserved at larger scale)
- Differentiable loss term: `L_energy = |mean(SR) - mean(upsample(LR))|`

### Land Surface Emissivity Prior
- Use NDVI from RGB: `(B3−B4)/(B3+B4)` as an auxiliary input to the colorization model
- NDVI strongly correlates with land surface temperature (LST) — vegetation is cooler, urban surfaces hotter
- Concatenate NDVI as an extra input channel to the U-Net
- This is physically grounded supervision

### Atmospheric Correction Consistency
- Landsat 9 Level-2 data already has atmospheric correction applied
- Verify that model outputs are consistent with expected LST ranges for each land cover type:
  - Water: 280–305 K
  - Urban: 295–320 K
  - Vegetation: 285–310 K
- Implement a soft range constraint loss during training

### Radiometric Flux Preservation
- For colorization: color appearance of each pixel should be physically consistent with its thermal signature
- Use a physics-informed auxiliary head that predicts approximate emissivity from the generated RGB output
- Penalize divergence from the TIR input — acts as a regularizer preventing spectrally absurd colorizations

**Tags:** Energy conservation loss · NDVI auxiliary input · LST range constraint · Emissivity regularizer

---

## Phase 5 — Evaluation & Metrics
**Category: Eval**

### PSNR — Peak Signal-to-Noise Ratio
- **SR:** compare SR output vs native 100m TIR ground truth
- **Colorization:** compare generated RGB vs co-registered Landsat RGB
- Target: PSNR >30 dB for SR, >25 dB for colorization
- Implementation: `skimage.metrics.peak_signal_noise_ratio`

### SSIM — Structural Similarity Index
- Measures luminance, contrast, and structure preservation
- More perceptually meaningful than PSNR
- Compute per-channel for colorization
- Target: SSIM >0.85 for SR
- Implementation: `skimage.metrics.structural_similarity` with `multichannel=True` for RGB

### FID — Fréchet Inception Distance
- For colorization quality — measures distributional similarity between generated and real RGB patches using Inception-v3 features
- Lower is better (target <50, ideally <20 for high-quality colorization)
- Requires ~50k patches minimum for stable estimate
- Implementation: `pytorch-fid` library

### Inference Time Per Tile
- Measure end-to-end latency for a 128×128 patch: (1) SR model forward pass, (2) colorization model forward pass
- Benchmark on target hardware (GPU T4 or equivalent)
- Goal: <100ms per tile
- Use `torch.cuda.synchronize()` for accurate CUDA timing
- Report mean ± std over 100 runs

### Qualitative Visual Inspection
Human review checklist:
1. No spatial artifacts / ringing / checkerboard at tile boundaries
2. No "hallucinated" features (fabricated roads, buildings, water bodies that don't exist in TIR)
3. Color plausibility for land cover type
4. Thermal gradients smoothly preserved

Build a visual comparison dashboard (matplotlib grid or Streamlit app) showing LR → SR → colorized triplets.

### Evaluation Targets Summary

| Metric | Task | Target |
|---|---|---|
| PSNR | Super-resolution | >30 dB |
| PSNR | Colorization | >25 dB |
| SSIM | Super-resolution | >0.85 |
| FID | Colorization | <20 |
| Inference time | Per tile (128×128) | <100 ms |

---

## Phase 6 — End-to-End Inference Pipeline
**Category: Pipeline**

### Scene-Level Tiling & Reconstruction
- Tile the input TIR scene into overlapping patches (128×128 with 16px overlap)
- Process each tile through SR → Colorization pipeline
- Reconstruct full scene using Gaussian-weighted blending in overlap regions to eliminate tile seams

### Geospatial Output
- Write output as Cloud-Optimized GeoTIFF (COG) preserving original CRS, geotransform, and nodata mask
- Use rasterio to write with proper projection and metadata
- Output files:
  - `TIR_SR_100m.tif` — super-resolved single-band TIR
  - `TIR_colorized_RGB_100m.tif` — 3-band colorized output

### Model Export & Deployment
- Export models to `TorchScript` or `ONNX` for portable inference without full PyTorch install
- Optionally quantize to INT8 (post-training quantization) for faster CPU inference
- Wrap in a CLI tool:
  ```bash
  python infer.py --input scene_B10.tif --output output/
  ```

### Full Pipeline Flow
```
Input TIR @200m → Tile (128×128) → SR model → TIR @100m → Colorization model → RGB @100m → Stitch + GeoTIFF
```

---

## Tech Stack Summary

| Category | Tools |
|---|---|
| **Libraries** | PyTorch, GDAL, Rasterio, OpenCV, tifffile, scikit-image |
| **SR Models** | ESRGAN (RRDB) / SwinIR |
| **Colorization Models** | Pix2Pix (cGAN) / ControlNet (Diffusion) |
| **Physics Losses** | Energy conservation, NDVI prior, LST range constraint, emissivity regularizer |
| **Evaluation** | PSNR, SSIM, FID, Inference time, Visual inspection |
| **Output Format** | Cloud-Optimized GeoTIFF (COG), TorchScript / ONNX |
| **Data Source** | Landsat 9 USGS Collection 2 Level-2 (B2, B3, B4, B10) |
