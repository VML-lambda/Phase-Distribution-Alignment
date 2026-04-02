"""
PDA-SSIM — Phase Distribution Alignment SSIM (ACM MM 2025)

Phase-aware SSIM with double mean removal and β-correction.

Algorithm (two-input alignment):
  1. Shift both images to zero-center: (x - 128) → [-128, 127]
  2. Double mean removal with symmetric wrap: removes global phase offset
  3. β-correction: if the mean absolute deviation > 64 (half of half-period),
     apply an additional 128-shift to resolve the ±π ambiguity
  4. Relative alignment: shift im2 toward im1 using their mean difference
  5. Normalize to [0,1] then convert to radian [-π, π]
  6. Decompose into (cos, sin) and compute SSIM on each component
  7. Aggregate: return sqrt(mean(SSIM_cos² + SSIM_sin²) / 2)

Reference
---------
Phase Distribution Matters: On the Importance of Phase Distribution Alignment (PDA)
in Holographic Applications. ACM MM, 2025.
"""

from warnings import warn
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.dtype import dtype_range
from skimage.util.arraycrop import crop
from skimage._shared.utils import check_shape_equality

__all__ = ['structural_similarity']


def structural_similarity(im1, im2,
                          *,
                          win_size=None, gradient=False, data_range=None,
                          multichannel=False, gaussian_weights=False,
                          full=False, **kwargs):
    """PDA-SSIM: phase-aware SSIM with distribution alignment.

    Parameters
    ----------
    im1, im2 : ndarray (uint8 or float, values in [0, 255])
        Phase images stored as 8-bit pixel values (0-255 ↔ -π to π).
    win_size : int or None
        Sliding window side length (must be odd). Default 7 (or Gaussian-derived).
    gradient : bool
        If True, return gradient w.r.t. im2.
    data_range : float
        Image value range (default 2, matching the [-1,1] normalized range).
    multichannel : bool
        If True, process last dimension as channels.
    gaussian_weights : bool
        If True, use Gaussian-weighted window (sigma=1.5).
    full : bool
        If True, return full SSIM map and component maps (Lu, Co, St).

    Returns
    -------
    mssim : float  — PDA-SSIM score ∈ [0, 1]
    Lu, Co, St, S : ndarray  — component maps (only when full=True)
    """
    check_shape_equality(im1, im2)

    if multichannel:
        args = dict(win_size=win_size, gradient=gradient, data_range=data_range,
                    multichannel=False, gaussian_weights=gaussian_weights, full=full)
        args.update(kwargs)
        nch = im1.shape[-1]
        mssim = np.empty(nch)
        if full:
            S = np.empty(im1.shape)
        for ch in range(nch):
            ch_result = structural_similarity(im1[..., ch], im2[..., ch], **args)
            if full:
                mssim[..., ch], S[..., ch] = ch_result[0], ch_result[-1]
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        return (mssim, S) if full else mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0: raise ValueError("K1 must be positive")
    if K2 < 0: raise ValueError("K2 must be positive")
    if sigma < 0: raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            r = int(truncate * sigma + 0.5)
            win_size = 2 * r + 1
        else:
            win_size = 7

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError("win_size exceeds image extent.")
    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        data_range = 2  # normalized [-1, 1] range after PDA alignment

    ndim = im1.ndim
    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    NP = win_size ** ndim
    cov_norm = NP / (NP - 1) if use_sample_covariance else 1.0

    # ------------------------------------------------------------------
    # PDA preprocessing (Phase Distribution Alignment)
    # ------------------------------------------------------------------
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    # Step 1: center to zero
    im1 = im1 - 128
    im2 = im2 - 128

    # Step 2: double mean removal (symmetric wrap to [-128, 127])
    im1 = (im1 - np.mean(im1) + 128) % 256 - 128
    im2 = (im2 - np.mean(im2) + 128) % 256 - 128
    im1 = (im1 - np.mean(im1) + 128) % 256 - 128
    im2 = (im2 - np.mean(im2) + 128) % 256 - 128

    # Step 3: β-correction (resolve ±π ambiguity)
    beta1 = 128 if np.mean(np.abs(im1)) > 64 else 0
    im1 = (im1 - beta1 + 128) % 256 - 128
    beta2 = 128 if np.mean(np.abs(im2)) > 64 else 0
    im2 = (im2 - beta2 + 128) % 256 - 128

    # Step 4: align im2 to im1 (remove relative mean offset)
    error_phase = (im2 - im1 + 128) % 256 - 128
    sum_val = np.mean(error_phase)
    im2 = (im2 - sum_val + 128) % 256 - 128

    # Step 5: normalize to [0, 1] then convert to radian
    im1 = (im1 + 128) / 255.
    im2 = (im2 + 128) / 255.
    im1 = (1 - im1) * 2 * np.pi - np.pi
    im2 = (1 - im2) * 2 * np.pi - np.pi

    # Step 6: complex (cos, sin) decomposition
    img1 = [np.cos(im1), np.sin(im1)]
    img2 = [np.cos(im2), np.sin(im2)]

    # Step 7: compute SSIM on each component and aggregate
    mssim = 0.0
    R = data_range
    C1, C2 = (K1 * R) ** 2, (K2 * R) ** 2
    C3 = C2 / 2

    for i in range(2):
        im1 = img1[i] + 1
        im2 = img2[i] + 1

        ux  = filter_func(im1, **filter_args)
        uy  = filter_func(im2, **filter_args)
        vx  = im1 - ux
        vy  = im2 - uy
        vxy = cov_norm * filter_func(vx * vy, **filter_args)
        vx  = cov_norm * filter_func(vx * vx, **filter_args)
        vy  = cov_norm * filter_func(vy * vy, **filter_args)

        A1, A2 = 2 * ux * uy + C1, 2 * vxy + C2
        B1, B2 = ux ** 2 + uy ** 2 + C1, vx + vy + C2

        Lu = A1 / B1
        sx, sy = np.sqrt(np.maximum(vx, 0)), np.sqrt(np.maximum(vy, 0))
        Co = (2 * sx * sy + C2) / B2
        St = (vxy + C3) / (sx * sy + C3)
        S  = Lu * Co * St

        pad = (win_size - 1) // 2
        mssim += (crop(S, pad).mean()) ** 2

    result = np.sqrt(mssim / 2)

    if full:
        return result, Lu, Co, St, S
    return result
