"""
C-SSIM — 256-period Circular SSIM

Operates in the uint8 [0, 255] domain and applies circular wrapping
to the local residuals (vx, vy) to handle the 256-period phase ambiguity.
No global mean centering is applied.  Returns S = (A1 * A2) / D (combined SSIM).
"""

from warnings import warn
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.arraycrop import crop
from skimage._shared.utils import check_shape_equality

__all__ = ['ssim_circular']


def _common_setup(im1, im2, win_size, gaussian_weights, data_range, kwargs):
    """Parse kwargs and build filter function."""
    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        truncate = 3.5
        r = int(truncate * sigma + 0.5)
        win_size = win_size if win_size is not None else 2 * r + 1
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate}
    else:
        win_size = win_size if win_size is not None else 7
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    if data_range is None:
        data_range = 255

    NP = win_size ** im1.ndim
    cov_norm = NP / (NP - 1) if use_sample_covariance else 1.0
    return K1, K2, data_range, filter_func, filter_args, cov_norm, win_size


def ssim_circular(im1, im2, *,
                  win_size=None, gradient=False, data_range=None,
                  multichannel=False, gaussian_weights=False, full=False, **kwargs):
    """C-SSIM: 256-period circular SSIM — variance wrapping only.

    Standard SSIM formula applied in the [0, 255] pixel domain, with local
    residuals (vx, vy) circularly wrapped to the half-period interval [-128, 127].
    No global mean centering is applied.  Returns S = (A1 * A2) / D.

    Parameters
    ----------
    im1, im2 : ndarray (uint8 or float, values in [0, 255])
    """
    check_shape_equality(im1, im2)

    if multichannel:
        args = dict(win_size=win_size, gradient=gradient, data_range=data_range,
                    multichannel=False, gaussian_weights=gaussian_weights, full=full)
        args.update(kwargs)
        nch = im1.shape[-1]
        mssim = np.empty(nch)
        if full:
            S_out = np.empty(im1.shape)
        for ch in range(nch):
            res = ssim_circular(im1[..., ch], im2[..., ch], **args)
            if full:
                mssim[ch], S_out[..., ch] = res[0], res[-1]
            else:
                mssim[ch] = res
        return (mssim.mean(), S_out) if full else mssim.mean()

    K1, K2, data_range, filter_func, filter_args, cov_norm, win_size = \
        _common_setup(im1, im2, win_size, gaussian_weights, data_range, kwargs)

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError("win_size exceeds image extent.")
    if win_size % 2 == 0:
        raise ValueError('Window size must be odd.')

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)
    vx = im1 - ux
    vy = im2 - uy

    # Circular variance wrapping (period = 256, half = 128)
    vx = np.where(vx > 128, vx - 256, vx)
    vx = np.where(vx < -128, vx + 256, vx)
    vy = np.where(vy > 128, vy - 256, vy)
    vy = np.where(vy < -128, vy + 256, vy)

    vxy = cov_norm * filter_func(vx * vy, **filter_args)
    vx  = cov_norm * filter_func(vx * vx, **filter_args)
    vy  = cov_norm * filter_func(vy * vy, **filter_args)

    R  = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux ** 2 + uy ** 2 + C1
    B2 = vx + vy + C2
    D  = B1 * B2
    S  = (A1 * A2) / D  # combined SSIM

    Lu = A1 / B1
    sx = np.sqrt(np.maximum(vx, 0))
    sy = np.sqrt(np.maximum(vy, 0))
    Co = (2 * sx * sy + C2) / B2
    C3 = C2 / 2
    St = (vxy + C3) / (sx * sy + C3)

    pad = (win_size - 1) // 2
    mssim = crop(S, pad).mean()

    if full:
        return mssim, Lu, Co, St, S
    return mssim
