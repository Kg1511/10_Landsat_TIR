"""Notebook-aligned Tiny Transformer colorization model."""

import torch
import torch.nn as nn


class TinyViTColorNet(nn.Module):
    """Tiny ViT encoder plus CNN decoder for TIR to RGB colorization.

    Input:  [B, 1, 256, 256]
    Output: [B, 3, 256, 256], normalized RGB in [0, 1]
    """

    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4, patch: int = 16):
        super().__init__()
        self.dim = dim
        self.patch = patch

        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.num_tokens = (256 // patch) * (256 // patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.decoder = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.patch_embed(x)
        batch, channels, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.transformer(tokens)
        feat = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.decoder(feat)
