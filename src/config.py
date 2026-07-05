"""
Central configuration for TIR Super-Resolution & Colorization pipeline.

All hyperparameters, paths, and constants are defined here so that
every training / evaluation / inference script imports from one place.
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths  (update DATASET_ROOT to your local copy)
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Point this to the folder that contains  sr/  colorization/  metadata/
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_current_repo_format")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")

# Submission output layout required by the challenge
SR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "model_outputs", "tir_superresolved_100m")
COLOR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "model_outputs", "colorized_tir_100m")

for _d in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR, SR_OUTPUT_DIR, COLOR_OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0
PIN_MEMORY = DEVICE.type == "cuda"

# ──────────────────────────────────────────────
# Data constants  (from dataset documentation)
# ──────────────────────────────────────────────
# TIR values are in Kelvin;  valid range enforced during patch creation
TIR_MIN = 250.0  # K
TIR_MAX = 350.0  # K
TIR_RANGE = TIR_MAX - TIR_MIN  # 100 K

# RGB reflectance already in [0, 1]
RGB_MIN = 0.0
RGB_MAX = 1.0

# Patch sizes
SR_LR_SIZE = 256   # TIR 200 m  input
SR_HR_SIZE = 512   # TIR 100 m  target
SR_SCALE = 2       # upscale factor

COLOR_SIZE = 256   # both TIR input and RGB target

# ──────────────────────────────────────────────
# Super-Resolution model hyper-parameters
# ──────────────────────────────────────────────
SR_IN_CHANNELS = 1
SR_OUT_CHANNELS = 1
SR_N_FEAT = 64          # first-conv feature width
SR_N_RRDB = 23          # number of RRDB blocks
SR_GC = 32              # growth channels inside dense blocks
SR_RESIDUAL_SCALING = 0.2

# ──────────────────────────────────────────────
# Colorization model hyper-parameters
# ──────────────────────────────────────────────
COLOR_IN_CHANNELS = 1
COLOR_OUT_CHANNELS = 3
UNET_FEATURES = [64, 128, 256, 512, 512, 512, 512, 512]
UNET_DROPOUT = 0.5      # applied in first 3 decoder blocks
UNET_USE_TANH = True     # output in [-1, 1], rescaled to [0, 1] at inference

# PatchGAN discriminator
DISC_IN_CHANNELS = COLOR_IN_CHANNELS + COLOR_OUT_CHANNELS  # 4: TIR + RGB
DISC_FEATURES = [64, 128, 256, 512]

# ──────────────────────────────────────────────
# Training — Super-Resolution
# ──────────────────────────────────────────────
SR_BATCH_SIZE = 8
SR_LR_G = 2e-4
SR_BETA1 = 0.9
SR_BETA2 = 0.99

# Stage 1: pixel-loss only
SR_STAGE1_EPOCHS = 100
SR_STAGE1_LOSS_WEIGHTS = {
    "l1": 1.0,
}

# Stage 2: pixel + perceptual + adversarial + physics
SR_STAGE2_EPOCHS = 100
SR_STAGE2_LOSS_WEIGHTS = {
    "l1": 1.0,
    "perceptual": 0.1,
    "adversarial": 0.01,
    "physics": 0.1,
}

SR_PATIENCE = 20  # early-stopping patience (epochs)

# ──────────────────────────────────────────────
# Training — Colorization
# ──────────────────────────────────────────────
COLOR_BATCH_SIZE = 4
COLOR_LR_G = 2e-4
COLOR_LR_D = 2e-4
COLOR_BETA1 = 0.5
COLOR_BETA2 = 0.999
COLOR_EPOCHS = 200
COLOR_G_UPDATES_PER_D = 2  # train G twice per D update

COLOR_LOSS_WEIGHTS = {
    "adversarial": 1.0,
    "l1": 100.0,
    "perceptual": 10.0,
    "ssim": 5.0,
}

# ──────────────────────────────────────────────
# Mixed precision
# ──────────────────────────────────────────────
USE_AMP = DEVICE.type == "cuda"

# ──────────────────────────────────────────────
# Evaluation targets  (for reference / logging)
# ──────────────────────────────────────────────
EVAL_TARGETS = {
    "sr_psnr_db": 30.0,
    "sr_ssim": 0.85,
    "color_psnr_db": 25.0,
    "color_fid": 20.0,
    "inference_ms_per_tile": 100.0,
}
