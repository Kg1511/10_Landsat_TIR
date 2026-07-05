# Codex Update Prompt — Align Repo With Correct Landsat-9 Notebook

## Main instruction for Codex

You are updating this GitHub repository:

```text
https://github.com/Kg1511/10_Landsat_TIR
```

Use the uploaded notebook as the **source of truth**:

```text
landsat9_correct_full_model_notebook.ipynb
```

The previous repo update was useful, but it appears the notebook itself was **not directly used**. Your job is to audit the repo against the notebook and patch anything that is missing, mismatched, or only partially implemented.

Do **not** blindly rewrite the whole project. Compare first, then update only the missing or incorrect parts.

---

## Current context

A previous Codex run already added or modified these kinds of files:

```text
src/preprocessing.py
src/data/sr_dataset.py
src/data/color_dataset.py
src/train_sr.py
src/train_colorization.py
src/evaluate.py
src/infer.py
src/models/tiny_vit.py
src/train_transformer_colorization.py
src/dataset_sanity.py
src/export_models.py
README.md
docs/
verify_models.py
```

It also created branches like:

```text
feature/preprocessing-stats
feature/project-completion
```

and likely pushed `main` and the feature branches to GitHub.

Still, the repo must now be checked against the exact notebook logic below.

---

## Non-negotiable organizer constraint

The project must clearly follow this rule:

```text
Use original Landsat-9 sensor arrays only.
Do not train on pseudo-colored, stretched, matplotlib-rendered, PNG, or colormap images.
```

Required behavior:

- Model input comes from `.npy` arrays directly.
- TIR arrays are numerically normalized using train-set statistics.
- RGB/reference arrays are numerically scaled using train-set statistics.
- Display stretching, matplotlib, colormaps, preview PNGs, and percentile scaling are only for human visualization.
- No visualization output is ever used as model input.
- The README and PPT/submission notes must explicitly mention this.

---

## Correct notebook pipeline to match

The correct notebook implements this order:

```text
1. Setup
2. Dataset sanity check
3. Compute/load preprocessing stats
4. Preprocessing functions
5. Dataset classes and dataloaders
6. Visualization helpers
7. CNN Super-Resolution model
8. CNN U-Net Colorization model
9. Shared training/evaluation utilities
10. Train CNN SR
11. Visualize SR prediction
12. Train CNN U-Net Colorization
13. Visualize colorization
14. Pix2Pix GAN Colorization
15. Train Pix2Pix GAN
16. Tiny Transformer Colorization
17. Train Tiny Transformer
18. Compare saved models
19. Final two-stage inference
20. Batch inference for common dataset
```

The default safe final model is:

```text
CNN SR + CNN U-Net Colorization
```

GAN and Transformer are comparative/extension models, not the safest default.

---

## Dataset format that must be supported

The dataset root must support this layout:

```text
dataset_current_repo_format/
  sr/
    train/
      tir_200m/*.npy      # input, shape (256, 256)
      tir_100m/*.npy      # target, shape (512, 512)
    val/
      tir_200m/*.npy
      tir_100m/*.npy
    test/
      tir_200m/*.npy
      tir_100m/*.npy

  colorization/
    train/
      tir_100m/*.npy      # input, shape (256, 256)
      rgb_100m/*.npy      # target, shape (3, 256, 256) or HWC convertable
    val/
      tir_100m/*.npy
      rgb_100m/*.npy
    test/
      tir_100m/*.npy
      rgb_100m/*.npy
```

Required shape checks:

```text
SR input:        (1, 256, 256)
SR target:       (1, 512, 512)
Color input:     (1, 256, 256)
Color target:    (3, 256, 256)
```

---

## Preprocessing stats — must match notebook

The notebook computes and saves:

```text
preprocess_stats.json
```

Required contents:

```json
{
  "tir_mean": "...",
  "tir_std": "...",
  "rgb_min": [ "...", "...", "..." ],
  "rgb_max": [ "...", "...", "..." ]
}
```

### TIR stats source

Compute TIR mean/std from **training data only**:

```text
sr/train/tir_200m
sr/train/tir_100m
colorization/train/tir_100m
```

Use original `.npy` values, finite values only.

### RGB stats source

Compute RGB min/max from **training data only**:

```text
colorization/train/rgb_100m
```

Support both CHW and HWC RGB input. Convert to CHW internally.

### Required formulas

```python
normalize_tir = (x - TIR_MEAN) / (TIR_STD + 1e-8)
denormalize_tir = x * (TIR_STD + 1e-8) + TIR_MEAN

normalize_rgb = clamp((rgb - RGB_MIN) / (RGB_MAX - RGB_MIN + 1e-8), 0, 1)
denormalize_rgb = rgb_norm * (RGB_MAX - RGB_MIN + 1e-8) + RGB_MIN
```

Important: do **not** replace TIR preprocessing with fixed Kelvin min-max scaling unless the notebook is also updated. The correct notebook uses train-set mean/std for TIR.

---

## Dataset loading behavior

Check/update dataset classes so they match the notebook behavior:

### `LandsatSRDataset`

Input:

```text
sr/<split>/tir_200m/*.npy
```

Target:

```text
sr/<split>/tir_100m/*.npy
```

Required behavior:

- Match files by basename.
- Load raw `.npy` using `np.load(...).astype(np.float32)`.
- Replace non-finite values safely.
- Convert to tensors.
- Add channel dimension.
- Normalize both input and target TIR using TIR mean/std.
- Augmentation must apply the same random flips to input and target.
- Return:

```python
x, y, name
```

where:

```text
x shape = (1, 256, 256)
y shape = (1, 512, 512)
```

### `LandsatColorDataset`

Input:

```text
colorization/<split>/tir_100m/*.npy
```

Target:

```text
colorization/<split>/rgb_100m/*.npy
```

Required behavior:

- Match files by basename.
- Load TIR as raw `.npy`, normalize with TIR mean/std.
- Load RGB as original `.npy`, convert HWC to CHW if needed.
- Normalize RGB using train RGB min/max.
- Same random flips for TIR and RGB.
- Return:

```python
x, y, name
```

where:

```text
x shape = (1, 256, 256)
y shape = (3, 256, 256)
```

---

## Visualization must stay display-only

Implement or verify these helpers:

```python
stretch_for_display(img, low=2, high=98)
rgb_chw_to_display(rgb_chw)
show_sr_sample(...)
show_color_sample(...)
visualize_sr_prediction(...)
visualize_color_prediction(...)
```

Rules:

- These helpers may use percentile stretch, `matplotlib`, `inferno`, PNG previews, etc.
- They must never feed stretched/display data back into training, validation, evaluation, or inference.
- Add comments/docstrings making this explicit.

---

## Model A — CNN Super-Resolution

The notebook’s SR model is not a heavy RRDB by default. It uses a simple residual CNN:

```text
TIR 200m 256×256
    -> bicubic upsample to 512×512
    -> CNN residual correction
    -> TIR 100m 512×512
```

Required class name from notebook:

```python
SimpleSRNet
```

Expected structure:

```python
class ResidualBlock(nn.Module):
    conv1: channels -> channels
    conv2: channels -> channels
    ReLU
    residual return identity + out

class SimpleSRNet(nn.Module):
    head: Conv2d(1, channels, 3, padding=1)
    body: num_blocks ResidualBlock
    tail: Conv2d(channels, 1, 3, padding=1)

    forward(x):
        x_up = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        feat = ReLU(head(x_up))
        feat = body(feat)
        residual = tail(feat)
        return x_up + residual
```

Default:

```python
SimpleSRNet(channels=64, num_blocks=6)
```

### Important audit point

If the repo currently uses an RRDB/ESRGAN-style SR model as the default, do one of these:

1. Add `SimpleSRNet` exactly and make it the safe/default notebook-aligned SR model, or
2. Keep RRDB only as an optional improved model, but make README and CLI clear that the notebook-correct baseline is `SimpleSRNet`.

Do not silently replace the notebook’s baseline with a different architecture without documenting it.

---

## Model B — CNN U-Net Colorization

Notebook objective:

```text
TIR 100m 256×256 -> RGB 100m 3×256×256
```

Required class name:

```python
ColorUNet
```

Expected behavior:

- Encoder-decoder U-Net with skip connections.
- Input channels = 1.
- Output channels = 3.
- Final activation = `sigmoid`, so RGB output is normalized `[0, 1]`.

Default:

```python
ColorUNet(base=32)
```

---

## Shared training/evaluation utilities

The notebook uses simple, stable training first.

Required functions/behavior:

```python
psnr_from_mse(mse, data_range=1.0)
save_checkpoint(...)
load_model_checkpoint(...)
evaluate_sr(...)
evaluate_color(...)
train_image_regression(...)
```

### Checkpoint contents

Every checkpoint should include:

```python
{
    "model_state_dict": model.state_dict(),
    "model_name": "...",
    "task": "...",
    "history": history,
    "preprocess_stats": PREPROCESS_STATS,
    "created_time": "..."
}
```

For GAN checkpoints, include:

```python
{
    "G_state_dict": G.state_dict(),
    "D_state_dict": D.state_dict(),
    "model_name": "Pix2Pix_ColorUNet_Generator_PatchDiscriminator",
    "task": "colorization_gan",
    "history": history,
    "best_val_l1": best_val_l1,
    "preprocess_stats": PREPROCESS_STATS,
    "created_time": "..."
}
```

### Training default

For `SimpleSRNet`, `ColorUNet`, and `TinyViTColorNet`, the notebook uses:

```text
AdamW
L1 loss
weight_decay = 1e-5
AMP when CUDA is available
gradient clipping = 1.0
best checkpoint based on val L1
```

Do not make VGG/perceptual/SSIM/physics losses mandatory for the default notebook-aligned path. Extra losses can exist as optional experiments, but the safe default should match the notebook.

### SR evaluation

Required SR metrics:

```text
l1_normalized
mse_normalized
mae_kelvin
rmse_kelvin
psnr_kelvin_range40
```

The notebook computes:

```python
mae_kelvin = l1_norm * TIR_STD
rmse_kelvin = sqrt(mse_norm) * TIR_STD
psnr_kelvin = psnr_from_mse(rmse_kelvin ** 2, data_range=40.0)
```

### Color evaluation

Required color metrics:

```text
l1_rgb_normalized
mse_rgb_normalized
psnr_rgb_normalized
l1_rgb_original_scale
mse_rgb_original_scale
```

---

## Model C — Pix2Pix GAN Colorization

GAN is optional/comparison. The CNN U-Net remains the safest final colorization model.

Required architecture:

```text
Generator: ColorUNet
Discriminator: PatchDiscriminator
Discriminator input: concatenate TIR + RGB => 4 channels
```

Required GAN loss:

```text
D loss = 0.5 * (BCE(real, 1) + BCE(fake, 0))
G loss = BCE(fake, 1) + 100 * L1(fake_rgb, real_rgb)
```

Expected defaults:

```python
PatchDiscriminator(in_channels=4, base=64)
lambda_l1 = 100.0
lr = 2e-4
Adam betas = (0.5, 0.999)
```

Important:

- Warm-start GAN generator from the trained CNN U-Net checkpoint if available.
- Save only the best GAN generator/discriminator checkpoint based on val L1.
- Evaluation should evaluate the generator only.

---

## Model D — Tiny Transformer Colorization

Required class name:

```python
TinyViTColorNet
```

Notebook architecture:

```text
TIR -> patch embedding -> transformer encoder -> CNN decoder -> RGB
```

Expected defaults:

```python
TinyViTColorNet(dim=128, depth=4, heads=4, patch=16)
```

Input/output:

```text
input:  (B, 1, 256, 256)
output: (B, 3, 256, 256)
```

Implementation details to verify:

- Patch embedding: `Conv2d(1, dim, kernel_size=patch, stride=patch)`
- `num_tokens = (256 // patch) * (256 // patch)`
- Learnable positional embedding.
- `nn.TransformerEncoderLayer(..., batch_first=True, norm_first=True)`
- Decoder upsamples back to 256×256.
- Final output uses `Sigmoid`.

---

## Model comparison

Implement a comparison script/function that evaluates saved checkpoints and writes:

```text
model_comparison_metrics.json
```

It should compare:

```text
SR CNN
Color CNN U-Net
Color Pix2Pix GAN
Color Tiny Transformer
```

Only include a model if its checkpoint exists.

---

## Final two-stage inference — most important

This must match the notebook exactly.

Pipeline:

```text
Raw TIR 200m 256×256
        ↓
SimpleSRNet / CNN SR
        ↓
Predicted TIR 100m 512×512
        ↓
Split into four 256×256 tiles
        ↓
Colorization model
        ↓
Merge tiles
        ↓
RGB-like 512×512 output
```

Required functions:

```python
predict_sr_from_raw_array(sr_model, tir_200m_hw)
colorize_512_tir_by_tiles(color_model, tir_100m_512_hw)
save_final_outputs(out_prefix, pred_tir_100m_512, pred_rgb_chw)
load_best_color_model(choice="cnn")
```

### `predict_sr_from_raw_array`

Required:

- Input shape must be `(256, 256)`.
- Load raw original TIR values.
- Normalize using TIR mean/std.
- Run SR model.
- Denormalize output back to original TIR physical scale.
- Return `(512, 512)` float32.

### `colorize_512_tir_by_tiles`

Required:

- Input shape must be `(512, 512)`.
- Split into exactly four tiles:

```python
(0, 256, 0, 256)
(0, 256, 256, 512)
(256, 512, 0, 256)
(256, 512, 256, 512)
```

- Normalize each TIR tile using TIR stats.
- Run color model on each tile.
- Merge normalized RGB tiles into `(3, 512, 512)`.
- Denormalize RGB using RGB train min/max.
- Return original-scale RGB CHW float32.

### `save_final_outputs`

Required outputs:

```text
*_pred_tir100m_512.npy
*_pred_rgb_chw_original_scale.npy
*_pred_bgr_chw.tif
*_preview.png
```

Important:

- TIFF should save BGR CHW if the final project convention requires BGR.
- Preview PNG is display-only and should be clearly named/handled as preview.

### `load_best_color_model`

Supported choices:

```text
cnn
gan
transformer
```

Default should be:

```text
cnn
```

---

## Batch inference for common/finale dataset

Implement or verify:

```python
run_batch_inference_on_folder(input_folder, output_folder, color_choice="cnn", max_files=None)
```

Expected input folder:

```text
common_dataset/
  sample_001.npy
  sample_002.npy
```

Each file:

```text
shape = (256, 256)
raw original TIR sensor values
```

Required behavior:

- Skip or warn on wrong shape.
- Run final two-stage inference.
- Save outputs for each file.
- Write:

```text
inference_manifest.json
```

The manifest should contain input path and all saved output paths.

---

## README must include exact run order

Add or verify README commands for:

```text
1. Dataset sanity check
2. Compute preprocessing stats
3. Train SR CNN
4. Train Color CNN U-Net
5. Optional train Pix2Pix GAN
6. Optional train Tiny Transformer
7. Evaluate/compare checkpoints
8. Run final two-stage inference
9. Run batch inference for common dataset
```

Also include a quick-run configuration:

```python
SR_EPOCHS = 5
COLOR_CNN_EPOCHS = 5
GAN_EPOCHS = 2
VIT_EPOCHS = 2
```

and final/better training suggestion:

```python
SR_EPOCHS = 20
COLOR_CNN_EPOCHS = 25
GAN_EPOCHS = 10
VIT_EPOCHS = 15
```

If the repo is script-based rather than notebook-based, equivalent CLI flags are fine.

---

## Specific things to check because previous update may differ

Check these carefully:

### 1. Did the repo default to RRDB instead of `SimpleSRNet`?

If yes, add `SimpleSRNet` and make it the notebook-correct baseline/default.

### 2. Did the repo make VGG/perceptual/physics/SSIM losses mandatory?

If yes, make them optional. The notebook default is simple L1 for SR/CNN color/Transformer and Pix2Pix GAN loss for GAN.

### 3. Are TIR stats mean/std, not min/max?

The correct notebook uses train-set mean/std for TIR.

### 4. Are RGB stats per-band min/max from training RGB only?

Must be per-band min/max, supporting CHW and HWC.

### 5. Is final inference tile-based after SR?

Colorization model expects 256×256 TIR, so the 512×512 SR result must be split into four tiles and merged.

### 6. Does batch inference write a manifest?

Required:

```text
inference_manifest.json
```

### 7. Are preview PNGs clearly marked display-only?

They must not be confused with training input/output.

### 8. Are checkpoints carrying preprocessing stats?

Every checkpoint should contain `preprocess_stats`.

### 9. Are model comparison outputs saved?

Required:

```text
model_comparison_metrics.json
```

### 10. Were full forward/model tests skipped before?

Previous log said only lightweight checks/syntax compilation ran in an environment without PyTorch. Now run actual model shape tests if PyTorch is available.

---

## Required verification before committing

Run as much of this as the environment supports:

```bash
git status --short --branch
python -m compileall src verify_models.py
python verify_models.py
python -m src.dataset_sanity --root <DATASET_ROOT>
```

If dataset is available, also run a tiny smoke test:

```bash
python -m src.preprocessing --root <DATASET_ROOT> --output <SAVE_DIR>/preprocess_stats.json
python -m src.train_sr --epochs 1
python -m src.train_colorization --epochs 1
python -m src.evaluate --task all
python -m src.infer --input <one_raw_256x256_tir.npy> --color-model cnn
```

If CLI names differ, adapt commands but perform equivalent checks.

---

## Git workflow

Use a safe branch:

```bash
git checkout main
git pull origin main
git checkout -b fix/notebook-alignment
```

After updates:

```bash
git add .
git commit -m "fix: align project with correct full notebook pipeline"
git push -u origin fix/notebook-alignment
```

If everything passes and direct push to `main` is expected:

```bash
git checkout main
git merge --no-ff fix/notebook-alignment -m "merge: notebook alignment fixes"
git push origin main
```

Do not expose tokens, credentials, or local auth folders. Ensure `.gh/`, `.env`, checkpoints, datasets, and generated outputs are ignored unless intentionally versioned.

---

## Final response expected from Codex

After finishing, summarize:

```text
- Which notebook requirements were already present
- Which files were changed
- Which mismatches were fixed
- Which checks passed
- Which checks could not run and why
- Whether main/branch was pushed
```

Also explicitly answer:

```text
The repo is now aligned with landsat9_correct_full_model_notebook.ipynb.
```

Only say that if you actually compared and patched the repo against the notebook.

---

## Extra strict notebook filename/function alignment addendum

Codex should also verify these exact notebook-level names and paths where practical. If the repository uses different names, either add compatibility aliases or clearly document the mapping in the README.

### Exact notebook checkpoint filenames

The correct notebook saves/loads these checkpoint names under `SAVE_DIR`:

```text
sr_cnn_residual_original_sensor_values.pth
color_cnn_unet_original_sensor_values.pth
color_pix2pix_original_sensor_values.pth
color_tiny_transformer_original_sensor_values.pth
model_comparison_metrics.json
```

Do not replace these with unrelated default names unless backward-compatible aliases or README mapping are added.

### Exact notebook function/class coverage checklist

The repo should contain equivalent implementations for all of these notebook-defined items:

```text
Classes:
LandsatSRDataset
LandsatColorDataset
ResidualBlock
SimpleSRNet
ConvBlock
ColorUNet
PatchDiscriminator
TinyViTColorNet

Functions:
list_npy
count_npy
summarize_array
check_pair
safe_load_npy
compute_tir_mean_std_from_train
compute_rgb_min_max_from_train
normalize_tir_tensor
denormalize_tir_tensor
rgb_min_tensor
rgb_max_tensor
normalize_rgb_tensor
denormalize_rgb_tensor
ensure_rgb_chw
make_loader
stretch_for_display
rgb_chw_to_display
show_sr_sample
show_color_sample
count_params
psnr_from_mse
save_checkpoint
load_model_checkpoint
evaluate_sr
evaluate_color
train_image_regression
visualize_sr_prediction
visualize_color_prediction
save_gan_checkpoint
train_pix2pix
predict_sr_from_raw_array
colorize_512_tir_by_tiles
save_final_outputs
load_best_color_model
run_batch_inference_on_folder
```

The function names do not need to be identical if the repo is modularized, but behavior must be equivalent and discoverable from the README.

### Kaggle/Drive path handling

The notebook is written to work with paths such as:

```text
/content/drive/MyDrive/landsat_india_200/dataset_current_repo_format
/content/drive/MyDrive/landsat_india_200/checkpoints_correct_full
./checkpoints_correct_full
```

The script repo should not hardcode only one environment. It should support CLI/config overrides for dataset root, save directory, and output directory.

### Batch sizes and debug/final epoch presets

Notebook defaults include:

```text
BATCH_SIZE_SR = 4
BATCH_SIZE_COLOR = 4
```

The README/config should expose quick debug and final training presets. The earlier section already lists the main quick/final values; if the notebook comments also mention ultra-fast debug values like `SR_EPOCHS=2`, `COLOR_CNN_EPOCHS=2`, `GAN_EPOCHS=1`, `VIT_EPOCHS=1`, those can be documented as smoke-test settings.

### Final submission notes

Ensure README/PPT notes explicitly state:

```text
- Raw Landsat-9 TIR sensor arrays are used for learning.
- Previews/colormaps are display-only.
- Final inference is two-stage: SR first, then tile-wise colorization.
- CNN SR + CNN U-Net is the safest default final model.
- Pix2Pix GAN and Tiny Transformer are comparison/extension models.
```
