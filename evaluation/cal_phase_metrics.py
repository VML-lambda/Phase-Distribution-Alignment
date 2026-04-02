"""
Phase-Domain Metric Calculator

Computes three types of phase-domain metrics for each (method, noise, quality):

  original  : standard SSIM / PSNR directly on phase pixel values
  circular  : mean-centered 256-period circular SSIM / PSNR (C-SSIM / C-PSNR)
  pda       : PDA-SSIM / PDA-PSNR (proposed, ACM MM 2025)

Results are first written as per-image text files:
  {method_path}/{noise}/{quality}/phase_ssim.txt        — standard SSIM
  {method_path}/{noise}/{quality}/phase_psnr.txt        — standard PSNR
  {method_path}/{noise}/{quality}/phase_cssim_{ver}.txt — circular / PDA SSIM
  {method_path}/{noise}/{quality}/phase_cpsnr_{ver}.txt — circular / PDA PSNR

Then all results are aggregated into:
  {method_path}/graph_{ver}/{method}.txt  (tab-separated, for evaluate.py)

Data format: each .txt row = "r g b avg" (one row per image).

VERSION_MAP — metric variant per version name
  'pda'      : PDA-SSIM / PDA-PSNR  ★ proposed (ACM MM 2025)
  'circular' : C-SSIM / C-PSNR — 256-period circular baseline

Usage
-----
python cal_phase_metrics.py --data_path /data/dpi_db --methods SGD NHVC
python cal_phase_metrics.py --data_path /data/dpi_db --methods SGD --version circular
python cal_phase_metrics.py --data_path /data/dpi_db --methods SGD --skip_compute
"""

import argparse
import copy
import math
import os

import cv2
import numpy as np
import pandas as pd

from metrics.ssim.ssim_original import structural_similarity as ssim_orig
from metrics.psnr.psnr_original import cal_psnr as psnr_orig
from metrics import Metric_SSIM, Metric_PSNR

CHANNELS = ['red', 'green', 'blue']
NOISES   = ['gb', 'wn', 'sp', 'ms', 'cc', 'no', 'block_cc', 'block_gb', 'block_ms', 'un']
CODEC    = ['hevc2', 'hevc', 'vvc']
QP       = [27, 32, 37, 42, 47]

# ---------------------------------------------------------------------------
# VERSION_MAP: version name → (Metric_SSIM key, Metric_PSNR key)
# ---------------------------------------------------------------------------
VERSION_MAP = {
    'pda':      ('pda',      'pda'     ),  # ★ PDA-SSIM / PDA-PSNR — proposed (ACM MM 2025)
    'circular': ('circular', 'circular'),  # C-SSIM / C-PSNR — 256-period circular baseline
}


# ---------------------------------------------------------------------------
# Per-image metric computation
# ---------------------------------------------------------------------------

def _read_img(path, img_name, channel):
    return cv2.imread(os.path.join(path, channel, img_name + '.png'),
                      cv2.IMREAD_GRAYSCALE)


def _write_metric(folder, filename, rows):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), 'w') as f:
        for row in rows:
            f.write(' '.join(str(v) for v in row) + '\n')


def compute_phase_metrics(method_path, img_names, noise, quality,
                          version, ssim_fn, psnr_fn):
    """Compute and save per-image phase metrics for one (noise, quality) pair.

    Parameters
    ----------
    method_path : str
        Root directory for the CGH method (e.g. /data/dpi_db/DPAC).
    img_names : list of str
        Image base names (without extension).
    noise : str
        Distortion type (e.g. 'gb', 'wn', 'hevc').
    quality : int or str
        Quality level (0–4 for synthetic noise; QP value for codecs).
    version : str
        Version name ('pda' or 'circular'), used to name output files.
    ssim_fn : callable
        Phase-domain SSIM function from Metric_SSIM (e.g. Metric_SSIM['pda']).
    psnr_fn : callable
        Phase-domain PSNR function from Metric_PSNR (e.g. Metric_PSNR['pda']).
    """
    ori_path   = os.path.join(method_path, 'ori')
    noise_path = os.path.join(method_path, noise, str(quality))

    ssim_rows, psnr_rows, cssim_rows, cpsnr_rows = [], [], [], []

    for img_name in img_names:
        ssim_vals, psnr_vals, cssim_vals, cpsnr_vals = [], [], [], []

        for c in CHANNELS:
            img_ori   = _read_img(ori_path,   img_name, c)
            img_noise = _read_img(noise_path, img_name, c)
            if img_ori is None or img_noise is None:
                continue

            img_ori_f   = img_ori.astype(np.float64)
            img_noise_f = img_noise.astype(np.float64)

            # Original SSIM (gaussian_weights=True for standard Wang 2004 mode)
            ssim_val = ssim_orig(copy.deepcopy(img_ori_f),
                                 copy.deepcopy(img_noise_f),
                                 gaussian_weights=True, data_range=255)
            ssim_vals.append(float(ssim_val))

            # Original PSNR (normalized to [0,1])
            _, psnr_val, _ = psnr_orig(copy.deepcopy(img_ori_f) / 255.,
                                       copy.deepcopy(img_noise_f) / 255.)
            psnr_vals.append(float(psnr_val))

            # Phase-domain SSIM (version-specific)
            cssim_val = ssim_fn(copy.deepcopy(img_ori_f),
                                copy.deepcopy(img_noise_f))
            cssim_vals.append(float(cssim_val))

            # Phase-domain PSNR (version-specific)
            _, cpsnr_val, _ = psnr_fn(copy.deepcopy(img_ori_f),
                                      copy.deepcopy(img_noise_f))
            cpsnr_vals.append(float(cpsnr_val))

        if ssim_vals:
            ssim_rows.append(ssim_vals  + [sum(ssim_vals)  / len(ssim_vals)])
            psnr_rows.append(psnr_vals  + [sum(psnr_vals)  / len(psnr_vals)])
            cssim_rows.append(cssim_vals + [sum(cssim_vals) / len(cssim_vals)])
            cpsnr_rows.append(cpsnr_vals + [sum(cpsnr_vals) / len(cpsnr_vals)])

    _write_metric(noise_path, 'phase_ssim.txt',                ssim_rows)
    _write_metric(noise_path, 'phase_psnr.txt',                psnr_rows)
    _write_metric(noise_path, f'phase_cssim_{version}.txt',    cssim_rows)
    _write_metric(noise_path, f'phase_cpsnr_{version}.txt',    cpsnr_rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _read_avg_col(folder, filename, n_images):
    """Read the last column (avg) from a space-separated metric file."""
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return [float('nan')] * n_images
    with open(path) as f:
        lines = f.readlines()
    return [float(l.strip().split()[-1]) for l in lines[:n_images]]


def aggregate_method(method_path, method, img_names, save_dir, version):
    """Read per-(noise, quality) metric files and write graph_{version}/{method}.txt.

    Parameters
    ----------
    method_path : str
        Root directory for the CGH method.
    method : str
        Method name (used as the output filename stem).
    img_names : list of str
        Image base names.
    save_dir : str
        Directory where {method}.txt is written (typically graph_{version}/).
    version : str
        Version name used in input filenames (e.g. 'pda' → phase_cssim_pda.txt).
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(img_names)

    records = []

    for noise in NOISES:
        for quality in range(5):
            folder = os.path.join(method_path, noise, str(quality))
            ssim   = _read_avg_col(folder, 'phase_ssim.txt',                n)
            psnr   = _read_avg_col(folder, 'phase_psnr.txt',                n)
            cssim  = _read_avg_col(folder, f'phase_cssim_{version}.txt',    n)
            cpsnr  = _read_avg_col(folder, f'phase_cpsnr_{version}.txt',    n)
            for idx in range(n):
                records.append({
                    'noise':      noise,
                    'quality':    str(quality),
                    'index':      idx + 1,
                    'Phase_SSIM':  ssim[idx],
                    'Phase_PSNR':  psnr[idx],
                    'Phase_CSSIM': cssim[idx],
                    'Phase_CPSNR': cpsnr[idx],
                })

    for codec in CODEC:
        for qp in QP:
            folder = os.path.join(method_path, codec, str(qp))
            ssim   = _read_avg_col(folder, 'phase_ssim.txt',                n)
            psnr   = _read_avg_col(folder, 'phase_psnr.txt',                n)
            cssim  = _read_avg_col(folder, f'phase_cssim_{version}.txt',    n)
            cpsnr  = _read_avg_col(folder, f'phase_cpsnr_{version}.txt',    n)
            for idx in range(n):
                records.append({
                    'noise':      codec,
                    'quality':    str(qp),
                    'index':      idx + 1,
                    'Phase_SSIM':  ssim[idx],
                    'Phase_PSNR':  psnr[idx],
                    'Phase_CSSIM': cssim[idx],
                    'Phase_CPSNR': cpsnr[idx],
                })

    df = pd.DataFrame(records)
    out_path = os.path.join(save_dir, method + '.txt')
    df.to_csv(out_path, sep='\t', index=False, encoding='utf-8')
    print(f'Saved: {out_path}  ({len(df)} rows)')
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute phase-domain metrics (PDA pipeline)')
    parser.add_argument('--data_path', required=True,
                        help='Root of DPI-DB; each method is a subdirectory')
    parser.add_argument('--methods', nargs='+', default=['DPAC', 'GS', 'SGD', 'NHVC'],
                        help='PoH method subdirectory names')
    parser.add_argument('--version', default='pda',
                        choices=list(VERSION_MAP.keys()),
                        help=(
                            'Metric variant to compute (default: pda). '
                            f'Available: {sorted(VERSION_MAP.keys())}'
                        ))
    parser.add_argument('--skip_compute', action='store_true',
                        help='Skip per-image computation; only aggregate existing txt files')
    args = parser.parse_args()

    ssim_key, psnr_key = VERSION_MAP[args.version]
    ssim_fn = Metric_SSIM[ssim_key]
    psnr_fn = Metric_PSNR[psnr_key]
    print(f'Version {args.version!r}: SSIM={ssim_key!r}, PSNR={psnr_key!r}')

    for method in args.methods:
        method_path = os.path.join(args.data_path, method)
        ori_dir = os.path.join(method_path, 'ori', CHANNELS[0])
        if not os.path.isdir(ori_dir):
            print(f'[WARN] {method}: ori dir not found, skipping.')
            continue
        img_names = sorted([f.split('.')[0] for f in os.listdir(ori_dir)
                            if f.endswith('.png')])

        if not args.skip_compute:
            for noise in NOISES:
                for quality in range(5):
                    print(f'[{method}] {noise} q={quality}')
                    compute_phase_metrics(method_path, img_names, noise, quality,
                                         args.version, ssim_fn, psnr_fn)
            for codec in CODEC:
                for qp in QP:
                    print(f'[{method}] {codec} qp={qp}')
                    compute_phase_metrics(method_path, img_names, codec, qp,
                                         args.version, ssim_fn, psnr_fn)

        save_dir = os.path.join(method_path, f'graph_{args.version}')
        aggregate_method(method_path, method, img_names, save_dir, args.version)


if __name__ == '__main__':
    main()
