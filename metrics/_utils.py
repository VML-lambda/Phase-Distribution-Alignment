"""
Shared utility functions for PDA holographic metrics.

polar_to_rect — Convert polar (magnitude, angle) to (real, imaginary)
rect_to_polar — Convert (real, imaginary) to polar (magnitude, angle)
kl            — KL divergence between two distributions (epsilon-smoothed)
phasemap_8bit — Convert radian phase map to uint8 for display/save
"""

import numpy as np

__all__ = ['polar_to_rect', 'rect_to_polar', 'kl', 'phasemap_8bit']


def polar_to_rect(mag, ang):
    """Convert polar complex representation to rectangular (real, imag).

    Works for both numpy arrays and torch tensors via duck typing.

    Parameters
    ----------
    mag : array-like  — magnitude (≥ 0)
    ang : array-like  — phase angle in radians

    Returns
    -------
    real, imag : same type as inputs
    """
    return mag * np.cos(ang), mag * np.sin(ang)


def rect_to_polar(real, imag):
    """Convert rectangular (real, imag) to polar (magnitude, angle).

    Parameters
    ----------
    real, imag : ndarray

    Returns
    -------
    mag : ndarray  — magnitude
    ang : ndarray  — angle in radians (from np.arctan2)
    """
    mag = np.sqrt(real ** 2 + imag ** 2)
    ang = np.arctan2(imag, real)
    return mag, ang


def kl(p, q, epsilon=1e-5):
    """Epsilon-smoothed KL divergence KL(p ‖ q).

    Parameters
    ----------
    p, q    : array-like  — probability distributions (need not sum to 1)
    epsilon : float       — small value added to avoid log(0)

    Returns
    -------
    float
    """
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    return float(np.sum(p * np.log(p / q)))


def phasemap_8bit(phase_rad, inverted=True):
    """Convert a radian phase map to a uint8 image for visualization.

    Parameters
    ----------
    phase_rad : ndarray  — phase values in [-π, π]
    inverted  : bool     — if True, maps π → 0 and -π → 255 (display convention)

    Returns
    -------
    ndarray (uint8)
    """
    phase_norm = (phase_rad + np.pi) / (2 * np.pi)  # [0, 1]
    if inverted:
        phase_norm = 1.0 - phase_norm
    return (phase_norm * 255).clip(0, 255).astype(np.uint8)
