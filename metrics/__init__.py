"""
PDA Holographic Metrics Package

Provides SSIM and PSNR variants for Phase-only Hologram (PoH) quality assessment,
designed to handle the 2π periodicity ambiguity of phase images.

Reference
---------
Phase Distribution Matters: On the Importance of Phase Distribution Alignment (PDA)
in Holographic Applications. ACM MM, 2025.

Quick Start
-----------
>>> from metrics import Metric_SSIM, Metric_PSNR
>>>
>>> # Phase-domain SSIM (PDA-SSIM — final proposed)
>>> ssim_val = Metric_SSIM['pda'](img_ref, img_dist)
>>>
>>> # Phase-domain PSNR (PDA-PSNR — final proposed)
>>> _, psnr_val, _ = Metric_PSNR['pda'](img_ref, img_dist)
"""

from metrics.ssim.ssim_original import structural_similarity as ssim_original
from metrics.ssim.ssim_circular import ssim_circular
from metrics.ssim.ssim_pda      import structural_similarity as ssim_pda

from metrics.psnr.psnr_original import cal_psnr
from metrics.psnr.psnr_circular import psnr_circular
from metrics.psnr.psnr_pda      import pda_psnr

# ---------------------------------------------------------------------------
# SSIM metric registry
# ---------------------------------------------------------------------------
Metric_SSIM = {
    'original': ssim_original,  # Standard SSIM (Wang 2004)
    'circular': ssim_circular,   # C-SSIM — 256-period circular SSIM
    'pda':      ssim_pda,       # ★ PDA-SSIM — proposed (ACM MM 2025)
}

# ---------------------------------------------------------------------------
# PSNR metric registry
# ---------------------------------------------------------------------------
Metric_PSNR = {
    'original': cal_psnr,      # Standard PSNR
    'circular': psnr_circular, # C-PSNR — 256-period circular PSNR
    'pda':      pda_psnr,      # ★ PDA-PSNR — proposed (ACM MM 2025)
}

__all__ = ['Metric_SSIM', 'Metric_PSNR']
