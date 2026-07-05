"""
Loss functions for TIR Super-Resolution & Colorization.

Provides
--------
- L1Loss          : pixel-level L1 reconstruction loss
- PerceptualLoss  : VGG19 feature-matching loss
- GANLoss         : vanilla / LSGAN adversarial loss
- SSIMLoss        : differentiable SSIM via Gaussian window
- PhysicsLoss     : radiometric consistency (SR only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# ────────────────────────────────────────────────────────
#  L1 Loss (basic pixel reconstruction)
# ────────────────────────────────────────────────────────

class L1Loss(nn.Module):
    """Standard L1 (MAE) loss."""

    def forward(self, pred, target):
        return F.l1_loss(pred, target)


# ────────────────────────────────────────────────────────
#  Perceptual Loss  (VGG19 feature matching)
# ────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """VGG19 perceptual loss.

    Extracts features from conv1_2, conv2_2, conv3_3, conv4_3 of a frozen
    VGG19 and computes the L1 distance between feature maps.

    For single-channel inputs the tensor is repeated to 3 channels before
    being fed into VGG.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Feature extraction slices
        self.slice1 = nn.Sequential(*list(vgg)[:4])    # conv1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9])   # conv2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:18])  # conv3_3
        self.slice4 = nn.Sequential(*list(vgg)[18:27]) # conv4_3

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalise(self, x):
        """Repeat to 3ch if needed and apply ImageNet normalisation."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = self._normalise(pred)
        target = self._normalise(target)

        loss = 0.0
        x, y = pred, target
        for layer in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)

        return loss


# ────────────────────────────────────────────────────────
#  GAN Loss
# ────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """Adversarial loss.  Supports 'vanilla' (BCE) and 'lsgan' (MSE) modes."""

    def __init__(self, mode: str = "vanilla"):
        super().__init__()
        if mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")

    def forward(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss_fn(pred, target)


# ────────────────────────────────────────────────────────
#  SSIM Loss
# ────────────────────────────────────────────────────────

def _gaussian_window(size: int, sigma: float):
    """Create 1-D Gaussian window and outer-product to 2-D."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(1) @ g.unsqueeze(0)  # outer product → (size, size)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss.

    Returns  1 - SSIM  so that lower = better, consistent with other losses.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        window = _gaussian_window(window_size, sigma)
        # Shape: [1, 1, ws, ws] — will be expanded for multi-channel
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        C = pred.shape[1]  # channels
        window = self.window.expand(C, -1, -1, -1)

        mu_p = F.conv2d(pred, window, padding=self.window_size // 2, groups=C)
        mu_t = F.conv2d(target, window, padding=self.window_size // 2, groups=C)

        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_p_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=C) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=C) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=C) - mu_pt

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
            (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
        )

        return 1.0 - ssim_map.mean()


# ────────────────────────────────────────────────────────
#  Physics Constraint Loss  (SR only)
# ────────────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """Radiometric fidelity constraint.

    The super-resolved output, when spatially averaged by 2×2 pooling,
    must match the original low-resolution input.

        L_physics = MSE( AvgPool2d(SR_out, 2) , LR_input )

    This prevents the SR model from hallucinating temperature values.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, sr_output, lr_input):
        """
        Parameters
        ----------
        sr_output : [B, 1, 512, 512]  (normalised HR prediction)
        lr_input  : [B, 1, 256, 256]  (normalised LR input)
        """
        downsampled = self.pool(sr_output)
        return F.mse_loss(downsampled, lr_input)
