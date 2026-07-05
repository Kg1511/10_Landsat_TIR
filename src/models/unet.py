"""
U-Net Generator for Pix2Pix Colorization (TIR → RGB).

Architecture
------------
TIR (1ch, 256×256)
  → Encoder: 8 down-blocks  [64, 128, 256, 512, 512, 512, 512, 512]
  → Decoder: 8 up-blocks    with skip connections from encoder
  → Tanh activation
  → RGB (3ch, 256×256)

Uses instance normalisation and dropout in the first 3 decoder blocks.

References
----------
- Isola et al., "Image-to-Image Translation with Conditional Adversarial
  Networks", CVPR 2017  (Pix2Pix).
"""

import torch
import torch.nn as nn

from src.config import (
    COLOR_IN_CHANNELS,
    COLOR_OUT_CHANNELS,
    UNET_FEATURES,
    UNET_DROPOUT,
    UNET_USE_TANH,
)


class ConvBlock(nn.Module):
    """Notebook U-Net block: Conv-BN-ReLU twice."""

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ColorUNet(nn.Module):
    """Notebook-aligned CNN U-Net colorizer with sigmoid RGB output."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ────────────────────────────────────────────────────────
#  Encoder / Decoder blocks
# ────────────────────────────────────────────────────────

class UNetEncoderBlock(nn.Module):
    """Conv4×4 stride-2 → InstanceNorm → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """ConvTranspose4×4 stride-2 → InstanceNorm → (Dropout) → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(UNET_DROPOUT))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ────────────────────────────────────────────────────────
#  Full U-Net generator
# ────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """Pix2Pix-style U-Net generator with skip connections.

    Input  : [B, 1, 256, 256]   (normalised TIR)
    Output : [B, 3, 256, 256]   (RGB, Tanh range [-1,1])
    """

    def __init__(
        self,
        in_ch: int = COLOR_IN_CHANNELS,
        out_ch: int = COLOR_OUT_CHANNELS,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = list(UNET_FEATURES)

        # ── Encoder ───────────────────────────────
        self.enc1 = UNetEncoderBlock(in_ch, features[0], use_norm=False)   # 256→128
        self.enc2 = UNetEncoderBlock(features[0], features[1])              # 128→64
        self.enc3 = UNetEncoderBlock(features[1], features[2])              # 64→32
        self.enc4 = UNetEncoderBlock(features[2], features[3])              # 32→16
        self.enc5 = UNetEncoderBlock(features[3], features[4])              # 16→8
        self.enc6 = UNetEncoderBlock(features[4], features[5])              # 8→4
        self.enc7 = UNetEncoderBlock(features[5], features[6])              # 4→2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[6], features[7], 4, stride=2, padding=1, bias=False),  # 2→1
            nn.ReLU(inplace=True),
        )

        # ── Decoder ───────────────────────────────
        # Each decoder input = upsample + skip concatenation
        self.dec1 = UNetDecoderBlock(features[7], features[6], use_dropout=True)      # 1→2
        self.dec2 = UNetDecoderBlock(features[6] * 2, features[5], use_dropout=True)  # 2→4
        self.dec3 = UNetDecoderBlock(features[5] * 2, features[4], use_dropout=True)  # 4→8
        self.dec4 = UNetDecoderBlock(features[4] * 2, features[3])                    # 8→16
        self.dec5 = UNetDecoderBlock(features[3] * 2, features[2])                    # 16→32
        self.dec6 = UNetDecoderBlock(features[2] * 2, features[1])                    # 32→64
        self.dec7 = UNetDecoderBlock(features[1] * 2, features[0])                    # 64→128

        # Final upsample  128→256
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features[0] * 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh() if UNET_USE_TANH else nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, 1, 256, 256]

        Returns
        -------
        out : [B, 3, 256, 256]  in [-1, 1] if Tanh, [0, 1] if Sigmoid
        """
        # Encoder (save activations for skip connections)
        e1 = self.enc1(x)          # [B, 64,  128, 128]
        e2 = self.enc2(e1)         # [B, 128,  64,  64]
        e3 = self.enc3(e2)         # [B, 256,  32,  32]
        e4 = self.enc4(e3)         # [B, 512,  16,  16]
        e5 = self.enc5(e4)         # [B, 512,   8,   8]
        e6 = self.enc6(e5)         # [B, 512,   4,   4]
        e7 = self.enc7(e6)         # [B, 512,   2,   2]

        # Bottleneck
        b = self.bottleneck(e7)    # [B, 512,   1,   1]

        # Decoder with skip connections
        d1 = self.dec1(b)                          # [B, 512,   2,   2]
        d2 = self.dec2(torch.cat([d1, e7], 1))     # [B, 512,   4,   4]
        d3 = self.dec3(torch.cat([d2, e6], 1))     # [B, 512,   8,   8]
        d4 = self.dec4(torch.cat([d3, e5], 1))     # [B, 512,  16,  16]
        d5 = self.dec5(torch.cat([d4, e4], 1))     # [B, 256,  32,  32]
        d6 = self.dec6(torch.cat([d5, e3], 1))     # [B, 128,  64,  64]
        d7 = self.dec7(torch.cat([d6, e2], 1))     # [B,  64, 128, 128]

        out = self.final(torch.cat([d7, e1], 1))   # [B,   3, 256, 256]
        return out
