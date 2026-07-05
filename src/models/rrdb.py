"""
RRDB-based Super-Resolution Generator (ESRGAN-style).

Architecture
------------
LR TIR (1ch, 256×256)
  → Conv3×3 (→ n_feat)
  → 23 × RRDB  (Residual-in-Residual Dense Block)
  → Conv3×3
  → PixelShuffle ×2 upsampling
  → Conv3×3
  → Conv3×3
  → HR TIR (1ch, 512×512)

References
----------
- Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial
  Networks", ECCV 2018 Workshops.
"""

import torch
import torch.nn as nn

from src.config import (
    SR_IN_CHANNELS,
    SR_OUT_CHANNELS,
    SR_N_FEAT,
    SR_N_RRDB,
    SR_GC,
    SR_RESIDUAL_SCALING,
)


# ────────────────────────────────────────────────────────
#  Building blocks
# ────────────────────────────────────────────────────────

class DenseBlock(nn.Module):
    """Five-layer dense block with LeakyReLU.

    Each layer receives the concatenation of all preceding features.
    """

    def __init__(self, n_feat: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feat, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feat + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(n_feat + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(n_feat + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(n_feat + 4 * gc, n_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Kaiming init
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Last conv scaled down
        self.conv5.weight.data *= 0.1

    def forward(self, x):
        c1 = self.lrelu(self.conv1(x))
        c2 = self.lrelu(self.conv2(torch.cat([x, c1], 1)))
        c3 = self.lrelu(self.conv3(torch.cat([x, c1, c2], 1)))
        c4 = self.lrelu(self.conv4(torch.cat([x, c1, c2, c3], 1)))
        c5 = self.conv5(torch.cat([x, c1, c2, c3, c4], 1))
        return c5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block.

    Three DenseBlocks with residual scaling β.
    """

    def __init__(self, n_feat: int = 64, gc: int = 32, beta: float = 0.2):
        super().__init__()
        self.db1 = DenseBlock(n_feat, gc)
        self.db2 = DenseBlock(n_feat, gc)
        self.db3 = DenseBlock(n_feat, gc)
        self.beta = beta

    def forward(self, x):
        out = self.db1(x) * self.beta + x
        out = self.db2(out) * self.beta + out
        out = self.db3(out) * self.beta + out
        return out * self.beta + x


# ────────────────────────────────────────────────────────
#  Full generator
# ────────────────────────────────────────────────────────

class RRDBNet(nn.Module):
    """RRDB-based super-resolution network (×2 upscale).

    Parameters
    ----------
    in_ch   : input channels  (default 1 for single-band TIR)
    out_ch  : output channels (default 1)
    n_feat  : base feature width
    n_rrdb  : number of RRDB blocks
    gc      : growth channels inside dense blocks
    """

    def __init__(
        self,
        in_ch: int = SR_IN_CHANNELS,
        out_ch: int = SR_OUT_CHANNELS,
        n_feat: int = SR_N_FEAT,
        n_rrdb: int = SR_N_RRDB,
        gc: int = SR_GC,
    ):
        super().__init__()

        # First convolution
        self.conv_first = nn.Conv2d(in_ch, n_feat, 3, 1, 1)

        # RRDB trunk
        self.trunk = nn.Sequential(
            *[RRDB(n_feat, gc, beta=SR_RESIDUAL_SCALING) for _ in range(n_rrdb)]
        )
        self.trunk_conv = nn.Conv2d(n_feat, n_feat, 3, 1, 1)

        # PixelShuffle ×2 upsampling
        self.upconv = nn.Conv2d(n_feat, n_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # Final output convolutions
        self.conv_hr = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(n_feat, out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Init
        for m in [self.conv_first, self.trunk_conv, self.upconv, self.conv_hr, self.conv_last]:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, 1, 256, 256]  (normalised LR TIR)

        Returns
        -------
        out : [B, 1, 512, 512]  (normalised HR TIR)
        """
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk  # global residual

        # Upsample ×2
        feat = self.lrelu(self.pixel_shuffle(self.upconv(feat)))

        # Output
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
