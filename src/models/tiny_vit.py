"""Tiny Vision Transformer colorization model."""

import torch
import torch.nn as nn

from src.config import (
    COLOR_IN_CHANNELS,
    COLOR_OUT_CHANNELS,
    COLOR_SIZE,
    UNET_USE_TANH,
    VIT_DEPTH,
    VIT_EMBED_DIM,
    VIT_HEADS,
    VIT_MLP_RATIO,
    VIT_PATCH_SIZE,
)


class TinyViTColorNet(nn.Module):
    """Small ViT encoder with a convolutional decoder for TIR to RGB.

    Input:  [B, 1, 256, 256]
    Output: [B, 3, 256, 256]
    """

    def __init__(
        self,
        in_ch: int = COLOR_IN_CHANNELS,
        out_ch: int = COLOR_OUT_CHANNELS,
        image_size: int = COLOR_SIZE,
        patch_size: int = VIT_PATCH_SIZE,
        embed_dim: int = VIT_EMBED_DIM,
        depth: int = VIT_DEPTH,
        heads: int = VIT_HEADS,
        mlp_ratio: float = VIT_MLP_RATIO,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size * self.grid_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.Tanh() if UNET_USE_TANH else nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        tokens = self.patch_embed(x)
        b, _, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.norm(self.encoder(tokens))
        features = tokens.transpose(1, 2).reshape(b, self.embed_dim, h, w)
        return self.decoder(features)
