"""
C-PSNR — 256-period Circular PSNR

Operates in the uint8 [0, 255] domain with global mean-centering modulo 256
(+128 offset) before computing the circular pixel-wise distance.
Paired with ssim_circular (C-SSIM) as the 256-period circular baseline.
"""

import math
import numpy as np

__all__ = ['psnr_circular']

_PERIOD = 256
_HALF   = 128


def psnr_circular(img1, img2):
    """C-PSNR: mean-centered (256, +128 offset) + circular distance.

    Both images are globally mean-centered modulo 256 (with +128 offset)
    before computing the circular pixel-wise distance.

    Parameters
    ----------
    img1, img2 : ndarray (uint8, values in [0, 255])

    Returns
    -------
    rmse, psnr (dB), error_map
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    img1 = (img1 - np.mean(img1) + _HALF) % _PERIOD
    img2 = (img2 - np.mean(img2) + _HALF) % _PERIOD

    error = np.abs(img1 - img2)
    error = np.where(error > _HALF, _PERIOD - error, error)
    error = error / 255.
    mse = np.mean(error ** 2)
    if mse == 0:
        return 0, 100, error
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(1.0 / rmse)
    return rmse, psnr, error
