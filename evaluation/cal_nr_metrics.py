"""
NR (Near-Reality) Domain Metric Calculator

Computes standard SSIM and PSNR on reconstructed (near-reality) images —
i.e., the images obtained by optical reconstruction from distorted phase holograms.

Input structure
---------------
{method_path}/ori/recon/{img_name}.png      — reference reconstruction
{method_path}/{noise}/{quality}/recon/{img_name}.png  — distorted reconstruction

Output (per distortion-quality pair)
-------------------------------------
{method_path}/{noise}/{quality}/ssim.txt    — per-image NR-SSIM (r g b avg)
{method_path}/{noise}/{quality}/psnr.txt    — per-image NR-PSNR (r g b avg)
{method_path}/{noise}/{quality}/rmse.txt    — per-image NR-RMSE (r g b avg)
{method_path}/{noise}/{quality}/kl.txt      — per-image histogram KL (r g b avg)

Then all NR results are aggregated into:
{method_path}/{method}.txt  (tab-separated, for evaluate.py)

Usage
-----
python cal_nr_metrics.py --data_path /data/dpi_db --methods SGD NHVC
"""

import argparse
import copy
import math
import os

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from metrics.ssim.ssim_original import structural_similarity as ssim_lcs

CHANNELS = ['red', 'green', 'blue']
NOISES   = ['gb', 'wn', 'sp', 'ms', 'cc', 'no', 'block_cc', 'block_gb', 'block_ms', 'un']
CODEC    = ['hevc2', 'hevc', 'vvc']
QP       = [27, 32, 37, 42, 47]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def cal_psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0.0, 100.0, np.zeros_like(img1)
    rmse = math.sqrt(mse)
    psnr = 20 * math.log10(1.0 / rmse)
    return rmse, psnr, np.abs(img1 - img2)


def _kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.where(p != 0, p * np.log(np.maximum(p / np.maximum(q, 1e-12), 1e-12)), 0)))


def smoothed_hist_kl(a, b, nbins=10, sigma=1):
    ah = np.histogram(a / 255., bins=nbins)[0].astype(float)
    bh = np.histogram(b / 255., bins=nbins)[0].astype(float)
    asmooth = gaussian_filter(ah, sigma)
    bsmooth = gaussian_filter(bh, sigma)
    kl_val = _kl(asmooth, bsmooth)
    return float(np.log(kl_val)) if kl_val > 0 else 0.0


def _write_metric(folder, filename, rows):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), 'w') as f:
        for row in rows:
            f.write(' '.join(str(v) for v in row) + '\n')


# ---------------------------------------------------------------------------
# Per (noise, quality) computation
# ---------------------------------------------------------------------------

def compute_nr_metrics(method_path, img_names, noise, quality):
    """Compute NR-domain SSIM, PSNR, RMSE, KL for one (noise, quality) pair."""
    ori_path   = os.path.join(method_path, 'ori', 'recon')
    noise_path = os.path.join(method_path, noise, str(quality), 'recon')

    ssim_rows, psnr_rows, rmse_rows, kl_rows = [], [], [], []

    for img_name in img_names:
        img_ori   = cv2.imread(os.path.join(ori_path,   img_name + '.png'))
        img_noise = cv2.imread(os.path.join(noise_path, img_name + '.png'))
        if img_ori is None or img_noise is None:
            continue

        ssim_vals, psnr_vals, rmse_vals, kl_vals = [], [], [], []

        for c in range(3):
            ori_ch   = img_ori[:, :, 2 - c].astype(np.float64)
            noise_ch = img_noise[:, :, 2 - c].astype(np.float64)

            kl_val = smoothed_hist_kl(ori_ch, noise_ch)

            ssim_val, *_ = ssim_lcs(copy.deepcopy(ori_ch), copy.deepcopy(noise_ch),
                                     gaussian_weights=True, full=True)
            rmse_val, psnr_val, _ = cal_psnr(copy.deepcopy(ori_ch), copy.deepcopy(noise_ch))

            ssim_vals.append(float(ssim_val))
            psnr_vals.append(float(psnr_val))
            rmse_vals.append(float(rmse_val))
            kl_vals.append(float(kl_val))

        def _row(vals):
            return vals + [sum(vals) / len(vals)]

        ssim_rows.append(_row(ssim_vals))
        psnr_rows.append(_row(psnr_vals))
        rmse_rows.append(_row(rmse_vals))
        kl_rows.append(_row(kl_vals))

    out_dir = os.path.join(method_path, noise, str(quality))
    _write_metric(out_dir, 'ssim.txt', ssim_rows)
    _write_metric(out_dir, 'psnr.txt', psnr_rows)
    _write_metric(out_dir, 'rmse.txt', rmse_rows)
    _write_metric(out_dir, 'kl.txt',   kl_rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _read_avg_col(folder, filename, n_images):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return [float('nan')] * n_images
    with open(path) as f:
        lines = f.readlines()
    return [float(l.strip().split()[-1]) for l in lines[:n_images]]


def aggregate_method(method_path, method, img_names):
    """Aggregate per-(noise, quality) NR metrics into {method}.txt."""
    n = len(img_names)
    records = []

    for noise in NOISES:
        for quality in range(5):
            folder = os.path.join(method_path, noise, str(quality))
            ssim = _read_avg_col(folder, 'ssim.txt', n)
            psnr = _read_avg_col(folder, 'psnr.txt', n)
            for idx in range(n):
                records.append({
                    'noise':   noise,
                    'quality': str(quality),
                    'index':   idx + 1,
                    'NR_SSIM': ssim[idx],
                    'NR_PSNR': psnr[idx],
                })

    for codec in CODEC:
        for qp in QP:
            folder = os.path.join(method_path, codec, str(qp))
            ssim = _read_avg_col(folder, 'ssim.txt', n)
            psnr = _read_avg_col(folder, 'psnr.txt', n)
            for idx in range(n):
                records.append({
                    'noise':   codec,
                    'quality': str(qp),
                    'index':   idx + 1,
                    'NR_SSIM': ssim[idx],
                    'NR_PSNR': psnr[idx],
                })

    df = pd.DataFrame(records)
    out_path = os.path.join(method_path, method + '.txt')
    df.to_csv(out_path, sep='\t', index=False, encoding='utf-8')
    print(f'Saved: {out_path}  ({len(df)} rows)')
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute NR-domain SSIM/PSNR metrics')
    parser.add_argument('--data_path', required=True,
                        help='Root of DPI-DB; each method is a subdirectory')
    parser.add_argument('--methods', nargs='+', default=['DPAC', 'GS', 'SGD', 'NHVC'],
                        help='PoH method subdirectory names')
    parser.add_argument('--skip_compute', action='store_true',
                        help='Skip per-image computation; only aggregate existing txt files')
    args = parser.parse_args()

    for method in args.methods:
        method_path = os.path.join(args.data_path, method)
        ori_recon = os.path.join(method_path, 'ori', 'recon')
        if not os.path.isdir(ori_recon):
            print(f'[WARN] {method}: ori/recon not found, skipping.')
            continue
        img_names = sorted([f.split('.')[0] for f in os.listdir(ori_recon)
                            if f.endswith('.png')])

        if not args.skip_compute:
            for noise in NOISES:
                for quality in range(5):
                    print(f'[{method}] NR {noise} q={quality}')
                    compute_nr_metrics(method_path, img_names, noise, quality)
            for codec in CODEC:
                for qp in QP:
                    print(f'[{method}] NR {codec} qp={qp}')
                    compute_nr_metrics(method_path, img_names, codec, qp)

        aggregate_method(method_path, method, img_names)


if __name__ == '__main__':
    main()
