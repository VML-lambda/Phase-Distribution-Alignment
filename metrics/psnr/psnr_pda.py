"""
PDA-PSNR — Phase Distribution Alignment PSNR (ACM MM 2025)

Phase-aware PSNR with double mean removal and β-correction.

Algorithm (same PDA preprocessing as PDA-SSIM):
  1. Shift both images to zero-center: (x - 128) → [-128, 127]
  2. Double mean removal with symmetric wrap
  3. β-correction: if mean(|x|) > 64, apply additional 128-shift
  4. Relative alignment: shift img2 toward img1 using mean error phase
  5. Normalize to [0, 1] then convert to radian [-π, π]
  6. Decompose into (cos+1, sin+1) and accumulate squared errors
  7. PIXEL_MAX = 2√2;  return (rmse, psnr, error_map)

Reference
---------
Phase Distribution Matters: On the Importance of Phase Distribution Alignment
(PDA) in Holographic Applications. ACM MM, 2025.
"""

import math
import numpy as np

__all__ = ['pda_psnr']


def _polar_to_rect(mag, ang):
    return mag * np.cos(ang), mag * np.sin(ang)


def pda_psnr(img1, img2):
    """PDA-PSNR: phase-aware PSNR with distribution alignment.

    Parameters
    ----------
    img1, img2 : ndarray (uint8 or float, values in [0, 255])
        Phase images stored as 8-bit pixel values (0–255 ↔ −π to π).

    Returns
    -------
    rmse  : float  — root mean squared error
    psnr  : float  — PDA-PSNR score in dB (100 if MSE == 0)
    error : ndarray — per-pixel squared error map (before mean)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Step 1: center to zero
    img1 = img1 - 128
    img2 = img2 - 128

    # Step 2: double mean removal (symmetric wrap to [-128, 127])
    img1 = (img1 - np.mean(img1) + 128) % 256 - 128
    img2 = (img2 - np.mean(img2) + 128) % 256 - 128
    img1 = (img1 - np.mean(img1) + 128) % 256 - 128
    img2 = (img2 - np.mean(img2) + 128) % 256 - 128

    # Step 3: β-correction (resolve ±π ambiguity)
    beta1 = 128 if np.mean(np.abs(img1)) > 64 else 0
    img1 = (img1 - beta1 + 128) % 256 - 128
    beta2 = 128 if np.mean(np.abs(img2)) > 64 else 0
    img2 = (img2 - beta2 + 128) % 256 - 128

    # Step 4: align img2 to img1 (remove relative mean offset)
    error_phase = (img2 - img1 + 128) % 256 - 128
    img2 = (img2 - np.mean(error_phase) + 128) % 256 - 128

    # Step 5: normalize to [0, 1] then convert to radian [-π, π]
    img1 = (img1 + 128) / 255.
    img2 = (img2 + 128) / 255.
    img1 = (1 - img1) * 2 * np.pi - np.pi
    img2 = (1 - img2) * 2 * np.pi - np.pi

    # Step 6: complex (cos, sin) decomposition
    im1 = _polar_to_rect(np.ones_like(img1), img1)
    im2 = _polar_to_rect(np.ones_like(img2), img2)

    PIXEL_MAX = 2 * math.sqrt(2)
    error = np.zeros_like(img1)
    for i in range(2):
        a = im1[i] + 1
        b = im2[i] + 1
        error += (a - b) ** 2

    mse = np.mean(error)
    if mse == 0:
        return 0, 100, error
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(PIXEL_MAX / rmse)
    return rmse, psnr, error
