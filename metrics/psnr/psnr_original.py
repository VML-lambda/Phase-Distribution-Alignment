"""
Standard PSNR for phase images.

cal_psnr — Standard MSE-based PSNR (PIXEL_MAX = 1, inputs normalized to [0,1])
"""

import math
import numpy as np

__all__ = ['cal_psnr']


def cal_psnr(img1, img2):
    """Standard PSNR.  Inputs should be normalized to [0, 1].

    Returns
    -------
    rmse  : float
    psnr  : float  (dB); 100 if MSE == 0
    error : ndarray  (absolute error map)
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    error = np.abs(img1 - img2)
    mse = np.mean(error ** 2)
    if mse == 0:
        return 0, 100, error
    PIXEL_MAX = 1.0
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(PIXEL_MAX / rmse)
    return rmse, psnr, error
