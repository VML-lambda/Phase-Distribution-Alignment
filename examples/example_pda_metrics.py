"""
Example: PDA-SSIM and PDA-PSNR Usage

Demonstrates how to use the final proposed metrics from the ACM MM 2025 paper
"Phase Distribution Matters: On the Importance of Phase Distribution Alignment (PDA)
in Holographic Applications."

Phase images are stored as uint8 (0–255) where 0 ↔ -π and 255 ↔ +π.
"""

import numpy as np
from metrics import Metric_SSIM, Metric_PSNR


# ---------------------------------------------------------------------------
# Create synthetic phase images for demonstration
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
H, W = 256, 256

# Reference phase image (random, uint8 [0, 255])
img_ref = rng.integers(0, 256, size=(H, W), dtype=np.uint8)

# Distorted: add Gaussian noise and apply modulo-256 wrap (phase is periodic)
noise = rng.normal(0, 15, size=(H, W))
img_dist = ((img_ref.astype(np.float64) + noise) % 256).astype(np.uint8)

# Globally phase-shifted distortion (tests PDA alignment)
shift = 64
img_shifted = ((img_ref.astype(np.float64) + shift) % 256).astype(np.uint8)


# ---------------------------------------------------------------------------
# Standard (non-phase-aware) metrics
# ---------------------------------------------------------------------------

ssim_orig = Metric_SSIM['original'](img_ref.astype(np.float64),
                               img_dist.astype(np.float64),
                               data_range=255)
_, psnr_orig, _ = Metric_PSNR['original'](img_ref.astype(np.float64) / 255.,
                                            img_dist.astype(np.float64) / 255.)

print('=== Standard metrics (noise distortion) ===')
print(f'  SSIM  : {ssim_orig:.4f}')
print(f'  PSNR  : {psnr_orig:.2f} dB')


# ---------------------------------------------------------------------------
# PDA-SSIM / PDA-PSNR (proposed)
# ---------------------------------------------------------------------------

# Noise distortion
ssim_pda_noise = Metric_SSIM['pda'](img_ref.astype(np.float64),
                                     img_dist.astype(np.float64))
_, psnr_pda_noise, _ = Metric_PSNR['pda'](img_ref.astype(np.float64),
                                            img_dist.astype(np.float64))

print('\n=== PDA metrics (noise distortion) ===')
print(f'  PDA-SSIM : {ssim_pda_noise:.4f}')
print(f'  PDA-PSNR : {psnr_pda_noise:.2f} dB')

# Global phase shift — standard SSIM fails, PDA corrects the offset
ssim_orig_shift = Metric_SSIM['original'](img_ref.astype(np.float64),
                                     img_shifted.astype(np.float64),
                                     data_range=255)
ssim_pda_shift  = Metric_SSIM['pda'](img_ref.astype(np.float64),
                                      img_shifted.astype(np.float64))
_, psnr_orig_shift, _ = Metric_PSNR['original'](img_ref.astype(np.float64) / 255.,
                                                  img_shifted.astype(np.float64) / 255.)
_, psnr_pda_shift, _  = Metric_PSNR['pda'](img_ref.astype(np.float64),
                                             img_shifted.astype(np.float64))

print('\n=== Phase-shifted distortion (shift=64 → identical holographic content) ===')
print(f'  Standard SSIM : {ssim_orig_shift:.4f}  ← penalises equivalent phase patterns')
print(f'  PDA-SSIM      : {ssim_pda_shift:.4f}   ← correctly identifies near-identical content')
print(f'  Standard PSNR : {psnr_orig_shift:.2f} dB')
print(f'  PDA-PSNR      : {psnr_pda_shift:.2f} dB')


# ---------------------------------------------------------------------------
# Full output (with component maps)
# ---------------------------------------------------------------------------

result = Metric_SSIM['pda'](img_ref.astype(np.float64),
                             img_dist.astype(np.float64),
                             full=True)
pda_score, Lu, Co, St, S = result

print(f'\n=== PDA-SSIM full output ===')
print(f'  Score    : {pda_score:.4f}')
print(f'  Lu map   : min={Lu.min():.3f}  max={Lu.max():.3f}')
print(f'  Co map   : min={Co.min():.3f}  max={Co.max():.3f}')
print(f'  St map   : min={St.min():.3f}  max={St.max():.3f}')
print(f'  SSIM map : min={S.min():.3f}   max={S.max():.3f}')
