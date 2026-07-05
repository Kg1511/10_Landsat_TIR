"""
PatchGAN Discriminator.

Used by both the SR GAN stage and the colorization cGAN.
For colorization the input is the concatenation of TIR + RGB (4 channels).
For SR the input is the concatenation of LR (upsampled) + HR candidate (2 channels).

The discriminator classifies 70×70 overlapping patches as real/fake.
Spectral normalisation is applied for training stability.

References
----------
- Isola et al., "Image-to-Image Translation with Conditional Adversarial
  Networks", CVPR 2017.
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.config import DISC_IN_CHANNELS, DISC_FEATURES


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator with spectral normalisation.

    Parameters
    ----------
    in_ch    : total input channels  (e.g. 4 = 1 TIR + 3 RGB for colorization)
    features : list of channel widths for hidden layers
    """

    def __init__(
        self,
        in_ch: int = DISC_IN_CHANNELS,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = list(DISC_FEATURES)

        layers = []

        # First layer — no normalisation
        layers.append(
            spectral_norm(nn.Conv2d(in_ch, features[0], 4, stride=2, padding=1))
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers — spectral norm + instance norm
        prev_ch = features[0]
        for feat in features[1:]:
            layers.append(
                spectral_norm(nn.Conv2d(prev_ch, feat, 4, stride=2, padding=1, bias=False))
            )
            layers.append(nn.InstanceNorm2d(feat, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_ch = feat

        # Patch-level classification head
        layers.append(
            spectral_norm(nn.Conv2d(prev_ch, 1, 4, stride=1, padding=1))
        )

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : [B, in_ch, H, W]   (concatenation of condition + image)

        Returns
        -------
        out : [B, 1, H', W']   patch-level real/fake scores
        """
        return self.model(x)
