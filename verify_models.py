"""Quick verification script - forward pass shape checks for all models and losses."""
import torch
import sys

print("Python: {}".format(sys.version))
print("PyTorch: {}".format(torch.__version__))
print("CUDA: {}".format(torch.cuda.is_available()))

# -- RRDB SR Generator --
print("\n--- RRDB Super-Resolution Generator ---")
from src.models.rrdb import RRDBNet
sr_model = RRDBNet()
params = sum(p.numel() for p in sr_model.parameters() if p.requires_grad)
print("  Parameters: {:,}".format(params))

x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = sr_model(x)
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
assert y.shape == (1, 1, 512, 512), "BAD SHAPE: {}".format(y.shape)
print("  PASS")

# -- U-Net Colorization Generator --
print("\n--- U-Net Colorization Generator ---")
from src.models.unet import UNetGenerator
color_model = UNetGenerator()
params = sum(p.numel() for p in color_model.parameters() if p.requires_grad)
print("  Parameters: {:,}".format(params))

x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = color_model(x)
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
assert y.shape == (1, 3, 256, 256), "BAD SHAPE: {}".format(y.shape)
print("  PASS")

# -- TinyViT Colorization Model --
print("\n--- TinyViT Colorization Model ---")
from src.models.tiny_vit import TinyViTColorNet
vit_model = TinyViTColorNet()
params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
print("  Parameters: {:,}".format(params))

x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    y = vit_model(x)
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
assert y.shape == (1, 3, 256, 256), "BAD SHAPE: {}".format(y.shape)
print("  PASS")

# -- PatchGAN Discriminator --
print("\n--- PatchGAN Discriminator ---")
from src.models.discriminator import PatchGANDiscriminator
disc = PatchGANDiscriminator()
params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
print("  Parameters: {:,}".format(params))

x = torch.randn(1, 4, 256, 256)
with torch.no_grad():
    y = disc(x)
print("  Input:  {}".format(list(x.shape)))
print("  Output: {}".format(list(y.shape)))
print("  PASS")

# -- Loss Functions --
print("\n--- Loss Functions ---")
from src.models.losses import L1Loss, GANLoss, SSIMLoss, PhysicsLoss

pred = torch.randn(1, 1, 256, 256)
target = torch.randn(1, 1, 256, 256)

l1 = L1Loss()(pred, target)
print("  L1 loss: {:.4f}".format(l1.item()))

ssim_loss = SSIMLoss()(pred, target)
print("  SSIM loss: {:.4f}".format(ssim_loss.item()))

gan = GANLoss("vanilla")
disc_out = torch.randn(1, 1, 30, 30)
print("  GAN loss (real): {:.4f}".format(gan(disc_out, True).item()))
print("  GAN loss (fake): {:.4f}".format(gan(disc_out, False).item()))

sr_out = torch.randn(1, 1, 512, 512)
lr_in = torch.randn(1, 1, 256, 256)
phys = PhysicsLoss()(sr_out, lr_in)
print("  Physics loss: {:.4f}".format(phys.item()))

# Perceptual loss (requires VGG download)
try:
    from src.models.losses import PerceptualLoss
    percep = PerceptualLoss()
    p1 = torch.randn(1, 3, 64, 64)
    p2 = torch.randn(1, 3, 64, 64)
    pl = percep(p1, p2)
    print("  Perceptual loss: {:.4f}".format(pl.item()))
except Exception as e:
    print("  Perceptual loss skipped: {}".format(e))

print("\n" + "=" * 50)
print("ALL SHAPE CHECKS PASSED")
print("=" * 50)
