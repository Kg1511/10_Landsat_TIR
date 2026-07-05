"""Notebook-aligned residual CNN super-resolution baseline."""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block used by the notebook SR baseline."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class SimpleSRNet(nn.Module):
    """Bicubic upsample plus CNN residual correction.

    Input:  [B, 1, 256, 256]
    Output: [B, 1, 512, 512]
    """

    def __init__(self, channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        feat = F.relu(self.head(x_up))
        feat = self.body(feat)
        residual = self.tail(feat)
        return x_up + residual
