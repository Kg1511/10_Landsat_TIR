# TIR Super-Resolution and Colorization Dataset Documentation

**Project:** Infrared Image Colorization and Enhancement for Improved Object Interpretation  
**Dataset region:** India  
**Dataset source:** Landsat 9 Collection 2 Level-2 imagery through Google Earth Engine  
**Current prepared dataset status:** Repo-compatible dataset created and verified on 30 exported scenes  
**Current paired samples:** 660 total paired samples  
**Prepared for:** IDE/model-development handoff

---

## 1. Purpose of This Dataset

This dataset is prepared for a two-stage machine learning pipeline:

1. **Thermal Infrared Super-Resolution**
   - Input: low-resolution TIR at 200 m
   - Target: high-resolution TIR at 100 m

2. **Thermal Infrared Colorization**
   - Input: TIR at 100 m
   - Target: RGB image at 100 m

The final inference pipeline expected by the challenge is:

```text
TIR 200m input
    ↓
Super-resolution model
    ↓
TIR 100m output
    ↓
Colorization model
    ↓
RGB/colorized TIR 100m output
```

This dataset is built to train those two model stages separately but consistently.

---

## 2. Challenge / Repository Requirements Followed

Reference repository:

```text
https://github.com/jugal-sac/IR-colorization-BAH2026/tree/main
```

The repository baseline specifies the following key requirements:

### 2.1 Required Landsat Bands

The required Landsat 9 bands are:

```text
B10 = Thermal Infrared band
B2  = Blue band
B3  = Green band
B4  = Red band
```

### 2.2 Super-Resolution Pair Format

The baseline dataset format for super-resolution is:

```text
Input:  256 × 256 TIR at 200m
Target: 512 × 512 TIR at 100m
```

This means one pixel in the 200m image corresponds to a `2 × 2` block in the 100m image.

### 2.3 Colorization Pair Format

The baseline dataset format for colorization is:

```text
Input:  256 × 256 TIR at 100m
Target: 256 × 256 RGB at 100m
```

### 2.4 Training File Format

Training files are saved as:

```text
.npy
```

The repository warns that `.png` files are only for visualization/verification and should not be used for model training because PNG loses radiometric precision.

### 2.5 Final Output Requirement

During inference/final submission, outputs must be saved as:

```text
output/
└── model_outputs/
    ├── tir_superresolved_100m/
    │   └── <product_id>.tif
    └── colorized_tir_100m/
        └── <product_id>.tif
```

For the colorized output TIFF, channel order must be:

```text
Layer 1 = Blue
Layer 2 = Green
Layer 3 = Red
```

Note that this final output band order is BGR, even though training arrays are currently stored as RGB.

---

## 3. Dataset Source

### 3.1 Satellite Data

Source collection used:

```text
LANDSAT/LC09/C02/T1_L2
```

This is Landsat 9 Collection 2 Level-2 data accessed through Google Earth Engine.

### 3.2 Region

The region used for dataset selection is:

```text
India
```

The India boundary was taken from:

```text
USDOS/LSIB_SIMPLE/2017
```

### 3.3 Time Range

Scenes were selected from:

```text
2022-01-01 to 2025-01-01
```

### 3.4 Cloud Filter

Initial scene filter:

```text
CLOUD_COVER < 10
```

Later, pixel-level quality filtering was also applied using Landsat QA masking.

---

## 4. Google Earth Engine Setup Used

The Earth Engine project used was:

```text
tir-project-499614
```

Initialization command used in Colab:

```python
import ee

ee.Authenticate(force=True)
ee.Initialize(project="tir-project-499614")
```

The Earth Engine connection was verified successfully by querying the Landsat 9 collection.

---

## 5. Scene Export Strategy

### 5.1 Initial Slow Approach Rejected

The first attempted approach was patch-first directly through Earth Engine:

```text
Colab → Earth Engine → sample one patch
Colab → Earth Engine → sample one patch
...
```

This was too slow because each patch requested data separately from Earth Engine.

### 5.2 Final Faster Approach

The chosen approach is:

```text
Earth Engine:
    Export cleaned 100m scenes as GeoTIFF

Colab/local IDE:
    Read GeoTIFFs locally
    Extract patches locally
    Save final .npy training dataset
```

This is much faster because Earth Engine is used only for scene export, while patch extraction is done locally.

### 5.3 Exported Scene Format

Each exported `processed100m_*.tif` file contains 4 bands:

```text
Band 1 = R
Band 2 = G
Band 3 = B
Band 4 = TIR
```

The exported files are already:

- cloud-masked
- quality-masked
- scaled to physical/usable values
- resampled to 100m
- stored as GeoTIFF

### 5.4 Important Scene Selection Correction

The first export used:

```python
.filterBounds(india)
```

This allowed scenes that barely touched India. When clipped/intersected with India, many files became too narrow, for example:

```text
149 × 66
140 × 64
1311 × 439
```

These cannot produce `512 × 512` patches.

The corrected strategy is:

1. Select scenes whose **centroid lies inside India**.
2. Export using the **full Landsat scene footprint**, not the small India intersection.

Corrected idea:

```python
def add_centroid_inside_india(img):
    centroid = img.geometry().centroid(100)
    inside = india.contains(centroid, ee.ErrorMargin(100))
    return img.set("centroid_inside_india", inside)
```

Then filter:

```python
.filter(ee.Filter.eq("centroid_inside_india", True))
```

And export with:

```python
region = img_raw.geometry()
```

not:

```python
region = img_raw.geometry().intersection(india, ee.ErrorMargin(1))
```

---

## 6. Earth Engine Preprocessing Logic

Each Landsat scene was processed as follows.

### 6.1 QA Pixel Mask

The `QA_PIXEL` band was used to remove bad pixels.

Masked conditions:

```text
Bit 0 = fill
Bit 1 = dilated cloud
Bit 2 = cirrus
Bit 3 = cloud
Bit 4 = cloud shadow
Bit 5 = snow
```

Mask logic:

```python
qa = img.select("QA_PIXEL")

mask = (
    qa.bitwiseAnd(1 << 0).eq(0)
    .And(qa.bitwiseAnd(1 << 1).eq(0))
    .And(qa.bitwiseAnd(1 << 2).eq(0))
    .And(qa.bitwiseAnd(1 << 3).eq(0))
    .And(qa.bitwiseAnd(1 << 4).eq(0))
    .And(qa.bitwiseAnd(1 << 5).eq(0))
)
```

### 6.2 Landsat Scaling Factors

Optical bands were scaled as:

```python
reflectance = raw_value * 0.0000275 - 0.2
```

Thermal band was scaled as:

```python
temperature_kelvin = raw_value * 0.00341802 + 149.0
```

### 6.3 Band Construction

```python
blue  = img.select("SR_B2").multiply(0.0000275).add(-0.2).rename("B")
green = img.select("SR_B3").multiply(0.0000275).add(-0.2).rename("G")
red   = img.select("SR_B4").multiply(0.0000275).add(-0.2).rename("R")
tir   = img.select("ST_B10").multiply(0.00341802).add(149.0).rename("TIR")
```

Export band stack:

```python
out = red.addBands(green).addBands(blue).addBands(tir)
```

Therefore exported GeoTIFF band order is:

```text
R, G, B, TIR
```

### 6.4 Physical Value Filtering

Pixels were additionally filtered by valid value ranges:

```text
R, G, B: 0 to 1
TIR:     250 K to 350 K
```

This removes obviously invalid reflectance or temperature values.

### 6.5 Resampling to 100m

All exported bands were resampled to a common 100m grid:

```python
out100 = out.resample("bilinear").reproject(crs=proj, scale=100)
```

---

## 7. Current Dataset State

### 7.1 Exported Scenes

Currently exported and locally detected scenes:

```text
processed100m_000 to processed100m_029
```

Total exported scenes available:

```text
30 scenes
```

### 7.2 Current Patch Counts

Current generated repo-format paired dataset:

```text
Total paired samples = 660
```

Breakdown:

```text
TASK: sr
  train
    tir_200m: 390
    tir_100m: 390
  val
    tir_200m: 120
    tir_100m: 120
  test
    tir_200m: 150
    tir_100m: 150

TASK: colorization
  train
    tir_100m: 390
    rgb_100m: 390
  val
    tir_100m: 120
    rgb_100m: 120
  test
    tir_100m: 150
    rgb_100m: 150
```

The counts match across paired folders, which confirms that patch pairing is consistent.

### 7.3 Current Dataset Usefulness

The current 660-sample dataset is suitable for:

- validating dataset loaders
- testing IDE setup
- sanity-checking model code
- running quick baseline training
- confirming loss decreases
- debugging tensor shapes

It is not yet ideal as the final strong training dataset. For final training, aim for at least:

```text
3,000 to 5,000 paired samples minimum
```

Better target:

```text
8,000 to 10,000 paired samples
```

---

## 8. Final Dataset Directory Structure

The current dataset is saved locally as:

```text
/content/landsat_india_200_work/dataset_current/
```

A copy should be stored in Drive as:

```text
/content/drive/MyDrive/landsat_india_200/dataset_current_repo_format/
```

Expected folder structure:

```text
dataset_current_repo_format/
├── sr/
│   ├── train/
│   │   ├── tir_200m/
│   │   │   └── <patch_id>.npy
│   │   └── tir_100m/
│   │       └── <patch_id>.npy
│   ├── val/
│   │   ├── tir_200m/
│   │   └── tir_100m/
│   └── test/
│       ├── tir_200m/
│       └── tir_100m/
│
├── colorization/
│   ├── train/
│   │   ├── tir_100m/
│   │   │   └── <patch_id>.npy
│   │   └── rgb_100m/
│   │       └── <patch_id>.npy
│   ├── val/
│   │   ├── tir_100m/
│   │   └── rgb_100m/
│   └── test/
│       ├── tir_100m/
│       └── rgb_100m/
│
└── metadata/
    ├── patch_metadata.csv
    └── scene_summary.csv
```

---

## 9. Patch Shapes and Meaning

### 9.1 Super-Resolution Dataset

Input file:

```text
sr/<split>/tir_200m/<patch_id>.npy
```

Shape:

```text
(256, 256)
```

Meaning:

```text
Thermal Infrared input at 200m resolution
```

Target file:

```text
sr/<split>/tir_100m/<patch_id>.npy
```

Shape:

```text
(512, 512)
```

Meaning:

```text
Thermal Infrared target at 100m resolution
```

Relationship:

```text
1 pixel in TIR_200m corresponds to 2 × 2 pixels in TIR_100m.
```

### 9.2 Colorization Dataset

Input file:

```text
colorization/<split>/tir_100m/<patch_id>.npy
```

Shape:

```text
(256, 256)
```

Meaning:

```text
Thermal Infrared input at 100m resolution
```

Target file:

```text
colorization/<split>/rgb_100m/<patch_id>.npy
```

Shape:

```text
(3, 256, 256)
```

Meaning:

```text
RGB target at 100m resolution
```

Channel order in training arrays:

```text
Channel 0 = Red
Channel 1 = Green
Channel 2 = Blue
```

Important final-output conversion:

```text
Training RGB order: R, G, B
Final output TIFF order required by repo: B, G, R
```

---

## 10. Patch Extraction Logic

### 10.1 Base Patch Window

For every accepted scene, a random `512 × 512` window is sampled from the 100m grid.

This gives:

```text
TIR 100m patch: 512 × 512
RGB 100m patch: 3 × 512 × 512
```

### 10.2 SR Pair Creation

The `512 × 512` TIR 100m patch is used as the SR target.

The SR input is created by downsampling the `512 × 512` TIR 100m patch by a factor of 2 using mean pooling:

```text
512 × 512 TIR 100m
    ↓ 2× mean pooling
256 × 256 TIR 200m
```

This guarantees exact alignment between SR input and SR target.

### 10.3 Colorization Pair Creation

The center `256 × 256` crop is taken from the same valid `512 × 512` area:

```text
TIR 100m color input:  256 × 256
RGB 100m color target: 3 × 256 × 256
```

This guarantees the colorization input and target are spatially aligned.

---

## 11. Patch Quality Filters

A patch is saved only if it passes all quality checks.

### 11.1 Shape Checks

The scene must be at least:

```text
512 × 512
```

Otherwise it is skipped.

Required patch shapes:

```text
rgb512:  (3, 512, 512)
tir512:  (512, 512)
sr_tir200: (256, 256)
color_tir100: (256, 256)
color_rgb100: (3, 256, 256)
```

### 11.2 Valid Pixel Checks

Pixels must be finite:

```python
np.isfinite(...)
```

Bad values or masked pixels should not appear in saved patches.

### 11.3 Reflectance Range Check

RGB values must satisfy:

```text
0 < R <= 1
0 < G <= 1
0 < B <= 1
```

### 11.4 Temperature Range Check

TIR values must satisfy:

```text
250 K <= TIR <= 350 K
```

### 11.5 Valid Ratio Check

At least 95% of pixels must be valid:

```python
VALID_THRESHOLD = 0.95
```

### 11.6 Non-Flat Patch Check

Reject patches with too little variation:

```python
MIN_RGB_STD = 0.005
MIN_TIR_STD = 0.10
```

This prevents black, blank, uniform, or useless patches from entering the training dataset.

---

## 12. Metadata Files

### 12.1 patch_metadata.csv

Path:

```text
metadata/patch_metadata.csv
```

Columns:

```text
patch_id
scene_index
product_id
split
row_100m
col_100m
sr_tir200_path
sr_tir100_path
color_tir100_path
color_rgb100_path
```

Purpose:

- maps every patch to its source scene
- stores split assignment
- stores original crop location
- stores relative paths for both tasks

### 12.2 scene_summary.csv

Path:

```text
metadata/scene_summary.csv
```

Columns:

```text
scene_index
product_id
split
height
width
saved_patches
status
```

Possible `status` values:

```text
ok
no_good_patches
too_small
bad_band_count
```

Purpose:

- tells which scenes contributed patches
- tells which scenes were skipped
- helps decide whether more scenes need to be exported

---

## 13. Current Important Observations

### 13.1 Some Scenes Are Too Small

Some early exported scenes were too narrow after clipping or selection, for example:

```text
1311 × 439
1236 × 415
149 × 66
```

These cannot produce `512 × 512` SR target patches and are skipped.

### 13.2 Some Large Scenes May Still Produce Zero Patches

Example observed:

```text
Shape: 1868 × 1083
Saved patches: 0
Attempts: 5000
```

Possible reasons:

- too many masked pixels
- clouds/cloud shadows
- invalid reflectance/temperature values
- strict quality thresholds
- mostly water/low texture region

This is acceptable. Bad scenes should be skipped rather than polluting the dataset.

### 13.3 Successful Scenes Produce Patches Quickly

Example observed:

```text
Shape: 2101 × 1735
Saved patches: 30
Attempts: 46
```

This indicates a clean scene with many valid patch locations.

---

## 14. Loading Dataset in PyTorch

### 14.1 Super-Resolution Dataset Loader

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TIRSuperResolutionDataset(Dataset):
    def __init__(self, root, split="train", normalize=True):
        self.lr_dir = os.path.join(root, "sr", split, "tir_200m")
        self.hr_dir = os.path.join(root, "sr", split, "tir_100m")
        self.files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith(".npy")])
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)  # 256x256
        hr = np.load(os.path.join(self.hr_dir, fname)).astype(np.float32)  # 512x512

        if self.normalize:
            # Kelvin range used during filtering
            lr = (lr - 250.0) / 100.0
            hr = (hr - 250.0) / 100.0
            lr = np.clip(lr, 0, 1)
            hr = np.clip(hr, 0, 1)

        lr = torch.from_numpy(lr).unsqueeze(0)  # 1x256x256
        hr = torch.from_numpy(hr).unsqueeze(0)  # 1x512x512

        return lr, hr, fname
```

### 14.2 Colorization Dataset Loader

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TIRColorizationDataset(Dataset):
    def __init__(self, root, split="train", normalize=True):
        self.tir_dir = os.path.join(root, "colorization", split, "tir_100m")
        self.rgb_dir = os.path.join(root, "colorization", split, "rgb_100m")
        self.files = sorted([f for f in os.listdir(self.tir_dir) if f.endswith(".npy")])
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        tir = np.load(os.path.join(self.tir_dir, fname)).astype(np.float32)  # 256x256
        rgb = np.load(os.path.join(self.rgb_dir, fname)).astype(np.float32)  # 3x256x256, R,G,B

        if self.normalize:
            tir = (tir - 250.0) / 100.0
            tir = np.clip(tir, 0, 1)
            rgb = np.clip(rgb, 0, 1)

        tir = torch.from_numpy(tir).unsqueeze(0)  # 1x256x256
        rgb = torch.from_numpy(rgb)              # 3x256x256

        return tir, rgb, fname
```

---

## 15. Recommended Normalization

### 15.1 TIR Normalization

TIR values are in Kelvin.

Recommended normalization for training:

```python
tir_norm = (tir - 250.0) / 100.0
```

This maps:

```text
250 K → 0
350 K → 1
```

Clip after normalization:

```python
tir_norm = np.clip(tir_norm, 0, 1)
```

### 15.2 RGB Normalization

RGB reflectance values are already scaled approximately to:

```text
0 to 1
```

Use:

```python
rgb = np.clip(rgb, 0, 1)
```

For GANs or models using `tanh`, convert to `[-1, 1]`:

```python
x = x * 2.0 - 1.0
```

---

## 16. Model Development Recommendations

### 16.1 Start With SR Model First

Train the SR model before colorization.

Input:

```text
sr/train/tir_200m/*.npy
```

Target:

```text
sr/train/tir_100m/*.npy
```

Recommended first baseline:

```text
EDSR-lite or simple UNet upsampler
```

Loss:

```text
L1 loss or SmoothL1 loss
```

Only after this works, try:

```text
ESRGAN / SRGAN style model
```

### 16.2 Then Train Colorization Model

Input:

```text
colorization/train/tir_100m/*.npy
```

Target:

```text
colorization/train/rgb_100m/*.npy
```

Recommended first baseline:

```text
UNet
```

Then try:

```text
Pix2Pix
```

Loss:

```text
L1 RGB loss
```

Optional later:

```text
GAN loss
perceptual loss
SSIM loss
```

### 16.3 Avoid Training GAN First

Do not start directly with GAN-only training.

Recommended sequence:

```text
1. Train with pixel loss first
2. Confirm outputs are stable
3. Add adversarial/perceptual losses later
```

---

## 17. Recommended Development Order in IDE

Use this order after moving dataset into your IDE environment:

```text
1. Write config.py
2. Write dataset loaders
3. Load one SR batch and print shapes
4. Load one colorization batch and print shapes
5. Visualize samples from dataloader
6. Train a small SR baseline for 2-5 epochs
7. Train a small colorization baseline for 2-5 epochs
8. Save sample predictions
9. Add validation metrics
10. Scale model complexity
```

---

## 18. Shape Sanity Checks

Before training, verify these with a script:

```python
import os
import numpy as np

root = "dataset_current_repo_format"

for split in ["train", "val", "test"]:
    sr_lr_dir = os.path.join(root, "sr", split, "tir_200m")
    sr_hr_dir = os.path.join(root, "sr", split, "tir_100m")

    for f in os.listdir(sr_lr_dir)[:10]:
        lr = np.load(os.path.join(sr_lr_dir, f))
        hr = np.load(os.path.join(sr_hr_dir, f))
        assert lr.shape == (256, 256)
        assert hr.shape == (512, 512)

    c_tir_dir = os.path.join(root, "colorization", split, "tir_100m")
    c_rgb_dir = os.path.join(root, "colorization", split, "rgb_100m")

    for f in os.listdir(c_tir_dir)[:10]:
        tir = np.load(os.path.join(c_tir_dir, f))
        rgb = np.load(os.path.join(c_rgb_dir, f))
        assert tir.shape == (256, 256)
        assert rgb.shape == (3, 256, 256)

print("Shape checks passed.")
```

---

## 19. Visual Verification Code

Use this to visualize a sample:

```python
import os
import numpy as np
import matplotlib.pyplot as plt

root = "dataset_current_repo_format"
split = "train"

files = sorted(os.listdir(os.path.join(root, "sr", split, "tir_200m")))
fname = files[0]

sr_tir200 = np.load(os.path.join(root, "sr", split, "tir_200m", fname))
sr_tir100 = np.load(os.path.join(root, "sr", split, "tir_100m", fname))
color_tir100 = np.load(os.path.join(root, "colorization", split, "tir_100m", fname))
color_rgb100 = np.load(os.path.join(root, "colorization", split, "rgb_100m", fname))

rgb_disp = np.moveaxis(color_rgb100, 0, -1)
valid = np.isfinite(rgb_disp).all(axis=2)
vals = rgb_disp[valid]
p2, p98 = np.percentile(vals, [2, 98])
rgb_disp = np.clip((rgb_disp - p2) / (p98 - p2), 0, 1)

fig, ax = plt.subplots(1, 4, figsize=(18, 5))

ax[0].imshow(sr_tir200, cmap="gray")
ax[0].set_title("SR Input: TIR 200m")
ax[0].axis("off")

ax[1].imshow(sr_tir100, cmap="gray")
ax[1].set_title("SR Target: TIR 100m")
ax[1].axis("off")

ax[2].imshow(color_tir100, cmap="gray")
ax[2].set_title("Color Input: TIR 100m")
ax[2].axis("off")

ax[3].imshow(rgb_disp)
ax[3].set_title("Color Target: RGB 100m")
ax[3].axis("off")

plt.tight_layout()
plt.show()
```

---

## 20. Final Output TIFF Guidance

During inference, the model will output:

1. Super-resolved TIR at 100m
2. Colorized RGB at 100m

### 20.1 TIR Output

Save as single-band GeoTIFF:

```text
output/model_outputs/tir_superresolved_100m/<product_id>.tif
```

### 20.2 Colorized Output

Model likely outputs RGB order:

```text
R, G, B
```

But final required TIFF order is:

```text
B, G, R
```

So convert before saving:

```python
rgb = model_output_rgb  # shape: 3,H,W in R,G,B
bgr = rgb[[2, 1, 0], :, :]
```

Save `bgr` as 3-band GeoTIFF.

---

## 21. Important Limitations of Current Dataset

### 21.1 Dataset Size Is Still Small

Current sample count:

```text
660 paired samples
```

This is enough to start model development but not enough for a strong final model.

### 21.2 Scene Split Needs Final Cleanup Later

Current split is based on scene/file rank across available processed scenes.

For the final dataset, recommended approach:

1. Export many scenes.
2. Run patch extraction.
3. Keep only scenes with `saved_patches > 0`.
4. Split accepted scenes into:

```text
70% train
15% val
15% test
```

This prevents leakage and avoids empty/bad scenes affecting split balance.

### 21.3 India-Wide Generalization Still Needs More Scenes

Current 30-scene dataset may not cover all Indian terrain types.

For stronger performance, include more scenes across:

- urban areas
- farmland
- forests
- dry/arid regions
- coastal regions
- mountains/hills
- water boundaries

---

## 22. Recommended Next Dataset Expansion

Continue exporting scenes in batches:

```text
030–050
050–070
070–090
090–110
...
```

After each batch:

```text
1. Wait for Earth Engine tasks to complete
2. Copy new GeoTIFFs locally
3. Run local patch extraction
4. Check patch counts
5. Check scene_summary.csv
6. Continue until enough paired samples are collected
```

Target final count:

```text
minimum: 3,000–5,000 paired samples
better:  8,000–10,000 paired samples
```

---

## 23. Files to Move Into IDE Environment

Copy this folder into your IDE project:

```text
dataset_current_repo_format/
```

Also copy this documentation file.

Recommended IDE project structure:

```text
project_root/
├── dataset_current_repo_format/
├── docs/
│   └── TIR_SR_Colorization_Dataset_Documentation.md
├── src/
│   ├── datasets/
│   │   ├── sr_dataset.py
│   │   └── color_dataset.py
│   ├── models/
│   ├── train_sr.py
│   ├── train_colorization.py
│   └── infer_pipeline.py
├── outputs/
├── checkpoints/
└── README.md
```

---

## 24. Critical Do-Not-Forget Points

1. Do not train on PNG files.
2. Use `.npy` files for training.
3. SR model input shape is `1 × 256 × 256`.
4. SR model target shape is `1 × 512 × 512`.
5. Colorization model input shape is `1 × 256 × 256`.
6. Colorization target shape is `3 × 256 × 256` in RGB order.
7. Final colorized output TIFF must be saved in BGR order.
8. Keep patch IDs aligned across corresponding folders.
9. Use scene-level split, not patch-level split, for the final dataset.
10. Current dataset is correct for development but should be expanded for final training.

---

## 25. Current Status Summary

```text
Earth Engine authentication: complete
Google Cloud project: tir-project-499614
Region: India
Source: Landsat 9 Collection 2 Level-2
Export format: processed 100m GeoTIFFs
Exported scenes: 30
Repo-format patch dataset: created
Current paired samples: 660
SR train/val/test: 390 / 120 / 150
Colorization train/val/test: 390 / 120 / 150
Visual sample check: passed
Dataset ready for IDE model-development: yes
Dataset final-size ready: not yet, expansion recommended
```

---

## 26. References

- Challenge repository: https://github.com/jugal-sac/IR-colorization-BAH2026/tree/main
- Google Earth Engine Landsat 9 Collection 2 Level-2 catalog: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
- Google Earth Engine Export.image.toDrive documentation: https://developers.google.com/earth-engine/apidocs/export-image-todrive
- USGS EarthExplorer: https://earthexplorer.usgs.gov
