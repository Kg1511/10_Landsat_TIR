# Landsat-9 TIR Super-Resolution and Colorization

SAC / ISRO Bharatiya Antariksh Hackathon project for raw thermal infrared
enhancement and colorization.

The pipeline uses original `.npy` or GeoTIFF sensor values as model input. Plot
stretching, colormaps, screenshots, and other visualization outputs are kept out
of training and inference.

## What This Project Does

1. Super-resolves Landsat-9 TIR 200 m patches to TIR 100 m output.
2. Colorizes TIR 100 m patches into RGB-like 100 m output.
3. Saves and reuses `checkpoints/preprocess_stats.json` for reproducible model
   preprocessing.
4. Supports RRDB super-resolution, Pix2Pix/U-Net colorization, and TinyViT
   colorization comparison.
5. Exports final arrays and challenge-style BGR GeoTIFF outputs.

## Project Layout

```text
src/
  config.py                       Central paths and hyperparameters
  preprocessing.py                Train-set stats and normalization helpers
  dataset_sanity.py               Dataset pair, shape, and stats checks
  train_sr.py                     RRDB super-resolution training
  train_colorization.py           Pix2Pix/U-Net colorization training
  train_transformer_colorization.py TinyViT colorization training
  evaluate.py                     SR, U-Net, and TinyViT evaluation
  infer.py                        End-to-end SR + colorization inference
  export_models.py                TorchScript / ONNX export
  data/                           Dataset loaders
  models/                         RRDB, U-Net, PatchGAN, TinyViT, losses
docs/
  ISRO_Landsat9_SR_Colorization_Project_Documentation.md
```

## Dataset Format

Set `DATASET_ROOT` in `src/config.py`, or place the dataset at:

```text
dataset_current_repo_format/
  sr/
    train/tir_200m/*.npy
    train/tir_100m/*.npy
    val/tir_200m/*.npy
    val/tir_100m/*.npy
    test/tir_200m/*.npy
    test/tir_100m/*.npy
  colorization/
    train/tir_100m/*.npy
    train/rgb_100m/*.npy
    val/tir_100m/*.npy
    val/rgb_100m/*.npy
    test/tir_100m/*.npy
    test/rgb_100m/*.npy
```

Expected shapes:

```text
SR input:       (256, 256)
SR target:      (512, 512)
Color input:    (256, 256)
Color target:   (3, 256, 256)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

First validate the dataset and create preprocessing stats:

```bash
python -m src.dataset_sanity --write-stats
```

This writes:

```text
checkpoints/preprocess_stats.json
output/dataset_sanity_report.json
```

Train the SR model:

```bash
python -m src.train_sr --stage 1
python -m src.train_sr --stage 2
```

Train colorization models:

```bash
python -m src.train_colorization
python -m src.train_transformer_colorization
```

Evaluate:

```bash
python -m src.evaluate --task both
```

Run final inference:

```bash
python -m src.infer --input path/to/tir_200m.tif --output output
```

Use TinyViT instead of U-Net colorization:

```bash
python -m src.infer --input path/to/tir_200m.tif --output output --color_model tiny_vit --color_ckpt color_tiny_transformer_best.pth
```

Export trained models:

```bash
python -m src.export_models --task sr --checkpoint sr_stage2_best.pth
python -m src.export_models --task color --checkpoint color_best.pth
python -m src.export_models --task vit --checkpoint color_tiny_transformer_best.pth
```

## Outputs

Inference writes:

```text
output/model_outputs/tir_superresolved_100m/<product_id>.tif
output/model_outputs/colorized_tir_100m/<product_id>.tif
output/model_outputs/arrays/<product_id>_pred_tir100m.npy
output/model_outputs/arrays/<product_id>_pred_rgb_chw.npy
```

The colorized GeoTIFF is saved in BGR band order for the challenge output
format. The `.npy` RGB array remains channel-first RGB.

## Raw Sensor Value Rule

Training and inference use original arrays only. Visualization helpers can make
human-readable figures, but those figures are never passed back into the model.
The saved preprocessing stats define the numeric transform used across training,
evaluation, and final inference.
