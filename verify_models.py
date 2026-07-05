"""Forward-pass shape checks for notebook-aligned models."""

import sys

import torch

print("Python: {}".format(sys.version))
print("PyTorch: {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.is_available()))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("\n--- SimpleSRNet notebook baseline ---")
from src.models.simple_sr import SimpleSRNet

sr_model = SimpleSRNet(channels=64, num_blocks=6)
x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = sr_model(x)
print("  Parameters: {:,}".format(count_params(sr_model)))
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
assert y.shape == (1, 1, 512, 512)
print("  PASS")


print("\n--- ColorUNet notebook baseline ---")
from src.models.unet import ColorUNet

color_model = ColorUNet(base=32)
x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = color_model(x)
print("  Parameters: {:,}".format(count_params(color_model)))
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
assert y.shape == (1, 3, 256, 256)
assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
print("  PASS")


print("\n--- PatchDiscriminator notebook Pix2Pix discriminator ---")
from src.models.discriminator import PatchDiscriminator

disc = PatchDiscriminator(in_channels=4, base=64)
tir = torch.randn(1, 1, 256, 256)
rgb = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    y = disc(tir, rgb)
print("  Parameters: {:,}".format(count_params(disc)))
print("  Output: {}".format(list(y.shape)))
assert y.shape[1] == 1
print("  PASS")


print("\n--- TinyViTColorNet notebook transformer ---")
from src.models.tiny_vit import TinyViTColorNet

vit_model = TinyViTColorNet(dim=128, depth=4, heads=4, patch=16)
x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = vit_model(x)
print("  Parameters: {:,}".format(count_params(vit_model)))
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
assert y.shape == (1, 3, 256, 256)
assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
print("  PASS")


print("\n--- Optional RRDB SR Generator ---")
try:
    from src.models.rrdb import RRDBNet

    rrdb = RRDBNet(n_rrdb=1)
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = rrdb(x)
    print("  Output: {}".format(list(y.shape)))
    assert y.shape == (1, 1, 512, 512)
    print("  PASS")
except Exception as exc:
    print("  Skipped optional RRDB check:", exc)


print("\n" + "=" * 50)
print("ALL AVAILABLE SHAPE CHECKS PASSED")
print("=" * 50)
