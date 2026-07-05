# Landsat-9 TIR Super-Resolution and Colorization

Notebook-aligned implementation for raw Landsat-9 thermal infrared
super-resolution and colorization.

Source of truth:

```text
notebooks/landsat9_correct_full_model_notebook.ipynb
docs/codex_landsat9_notebook_alignment_update.md
```

## Non-Negotiable Rule

The models learn from original Landsat-9 sensor arrays only.

- Raw `.npy` TIR/RGB arrays are used for training and inference.
- TIR is normalized with train-set mean/std.
- RGB is normalized with train-set per-band min/max.
- PNGs, colormaps, matplotlib figures, percentile stretching, and previews are display-only.
- Display outputs are never fed back into a model.

## Safe Default Pipeline

The safe final model is:

```text
SimpleSRNet CNN SR + ColorUNet CNN colorization
```

Pix2Pix GAN and Tiny Transformer are comparison/extension models.

Final inference is always two-stage:

```text
Raw TIR 200m 256x256
  -> SimpleSRNet
  -> predicted TIR 100m 512x512
  -> split into four 256x256 tiles
  -> ColorUNet / GAN / Transformer colorizer
  -> RGB-like 512x512 output
```

## Dataset Layout

```text
dataset_current_repo_format/
  sr/train/tir_200m/*.npy
  sr/train/tir_100m/*.npy
  sr/val/tir_200m/*.npy
  sr/val/tir_100m/*.npy
  sr/test/tir_200m/*.npy
  sr/test/tir_100m/*.npy

  colorization/train/tir_100m/*.npy
  colorization/train/rgb_100m/*.npy
  colorization/val/tir_100m/*.npy
  colorization/val/rgb_100m/*.npy
  colorization/test/tir_100m/*.npy
  colorization/test/rgb_100m/*.npy
```

Supported shapes:

```text
SR input:      (256, 256) -> tensor (1, 256, 256)
SR target:     (512, 512) -> tensor (1, 512, 512)
Color input:   (256, 256) -> tensor (1, 256, 256)
Color target:  (3, 256, 256) or HWC RGB -> tensor (3, 256, 256)
```

## Setup

```bash
pip install -r requirements.txt
```

Set `DATASET_ROOT` in `src/config.py`, or pass `--dataset-root` / `--root` where available.

## Deployable Dashboard

A real API-backed dashboard is included at `apps/dashboard/`.

```bash
pip install -r apps/dashboard/backend/requirements.txt
uvicorn apps.dashboard.backend.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The dashboard validates raw `.npy` TIR uploads with shape `(256, 256)`, runs the
same final two-stage inference path used by the notebook-aligned scripts, shows
Raw TIR / SR TIR / RGB-like output panels, and provides downloads for:

```text
*_pred_tir100m_512.npy
*_pred_rgb_chw_original_scale.npy
*_pred_bgr_chw.tif
*_preview.png
inference_manifest.json
```

Production dashboard inference requires these files under `SAVE_DIR`:

```text
preprocess_stats.json
sr_cnn_residual_original_sensor_values.pth
color_cnn_unet_original_sensor_values.pth
```

GAN and Transformer dashboard modes additionally need their matching comparison
checkpoints. See `apps/dashboard/README.md` for environment variables, demo
mode, Docker, and deployment notes.

## Exact Run Order

1. Dataset sanity check and preprocessing stats:

```bash
python -m src.dataset_sanity --root dataset_current_repo_format --write-stats
```

2. Train SR CNN baseline:

```bash
python -m src.train_sr --dataset-root dataset_current_repo_format --epochs 5
```

3. Train Color CNN U-Net baseline:

```bash
python -m src.train_colorization --dataset-root dataset_current_repo_format --mode cnn --epochs 5
```

4. Optional Pix2Pix GAN comparison:

```bash
python -m src.train_colorization --dataset-root dataset_current_repo_format --mode gan --epochs 2
```

5. Optional Tiny Transformer comparison:

```bash
python -m src.train_transformer_colorization --dataset-root dataset_current_repo_format --epochs 2
```

6. Evaluate/compare checkpoints:

```bash
python -m src.evaluate --task all --root dataset_current_repo_format
python -m src.compare_models --dataset-root dataset_current_repo_format
```

7. Run final two-stage inference on one raw 256x256 TIR `.npy`:

```bash
python -m src.infer --input sample_001.npy --output output/final_two_stage_outputs --color-model cnn
```

8. Run batch inference for common/finale dataset:

```bash
python -m src.infer --input common_dataset --output output/common_outputs --batch --color-model cnn
```

Batch inference writes `inference_manifest.json`.

## Training Presets

Smoke test:

```python
SR_EPOCHS = 2
COLOR_CNN_EPOCHS = 2
GAN_EPOCHS = 1
VIT_EPOCHS = 1
```

Quick run:

```python
SR_EPOCHS = 5
COLOR_CNN_EPOCHS = 5
GAN_EPOCHS = 2
VIT_EPOCHS = 2
```

Better/final run:

```python
SR_EPOCHS = 20
COLOR_CNN_EPOCHS = 25
GAN_EPOCHS = 10
VIT_EPOCHS = 15
```

## Notebook Checkpoint Names

Saved under `checkpoints/` by default:

```text
sr_cnn_residual_original_sensor_values.pth
color_cnn_unet_original_sensor_values.pth
color_pix2pix_original_sensor_values.pth
color_tiny_transformer_original_sensor_values.pth
model_comparison_metrics.json
preprocess_stats.json
```

## Final Output Files

Single or batch inference saves:

```text
*_pred_tir100m_512.npy
*_pred_rgb_chw_original_scale.npy
*_pred_bgr_chw.tif
*_preview.png
```

The preview PNG is stretched for human display only. The BGR TIFF is produced
for final project convention compatibility.

## Notebook-Aligned Names

The repo exposes the notebook names in `src.notebook_pipeline` and model modules:

```text
LandsatSRDataset, LandsatColorDataset
ResidualBlock, SimpleSRNet
ConvBlock, ColorUNet
PatchDiscriminator
TinyViTColorNet
predict_sr_from_raw_array
colorize_512_tir_by_tiles
save_final_outputs
load_best_color_model
run_batch_inference_on_folder
```
