"""
PLCC / SROCC Evaluation

Reads pre-computed NR and phase-domain metrics, applies a linear fit to
PDA-SSIM scores, computes Pearson (PLCC) and Spearman (SROCC) correlations,
and saves:
  1. pcc.txt   -- per-method PLCC/SROCC table (input for draw_graph.py)
  2. data.tsv  -- merged raw data with fitted PDA-SSIM (input for draw_graph.py)

For visualization, run draw_graph.py after this script.

Data input -- 2D (--data_2d, default subfolder layout from cal_nr/cal_phase pipeline)
----------
{data_2d}/{method}/{method}.txt                  -- NR_PSNR, NR_SSIM, noise, quality
{data_2d}/{method}/graph_{version}/{method}.txt  -- Phase_PSNR, Phase_CPSNR,
                                                    Phase_SSIM, Phase_CSSIM

Data input -- 2D (--flat, compatible with original cal_corr3.py data)
----------
{data_2d}/{method}.txt                  -- NR metrics
{data_2d}/graph_{version}/{method}.txt  -- Phase metrics

Data input -- 3D B-COM (--data_3d)
----------
{data_3d}/bcom.txt                          -- NR metrics
{data_3d}/graph_{version}/average_metric.txt -- Phase metrics

Usage
-----
# 2D only (new pipeline)
python evaluate.py --data_2d /data/dpi_db --methods DPAC GS SGD NHVC --out_dir .

# 2D only (existing cal_corr3.py data, e.g. 2K_poh)
python evaluate.py --data_2d /SSIM/2K_poh --methods DPAC GS SGD HOLONET NHVC --flat --out_dir .

# 3D only
python evaluate.py --data_3d /SSIM/2K_full --out_dir .

# 2D + 3D
python evaluate.py --data_2d /SSIM/2K_poh --data_3d /SSIM/2K_full --methods DPAC GS SGD HOLONET NHVC --flat --out_dir .

# Use circular baseline instead of PDA
python evaluate.py --data_2d /data/dpi_db --methods DPAC GS SGD NHVC --version circular --out_dir .
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NOISES = ['wn', 'gb', 'sp', 'cc', 'no', 'un', 'block_gb', 'block_ms', 'block_cc',
          'hevc', 'vvc', 'hevc2']
CODEC  = ['hevc2', 'hevc', 'vvc']

# Linear scaling for PDA-SSIM (fitted on DPI-DB)
FIT_A, FIT_B = 2.7, -1.87


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_data(data_2d, method, version, flat=False):
    """Load and merge NR + phase metrics for one 2D method.

    Parameters
    ----------
    data_2d : str
        Root 2D data directory (--data_2d).
    method : str
        Method name (e.g. 'DPAC', 'SGD').
    version : str
        Metric variant ('pda' or 'circular'). Selects graph_{version}/ folder.
    flat : bool
        If True, use flat layout compatible with cal_corr3.py:
            {data_2d}/{method}.txt
            {data_2d}/graph_{version}/{method}.txt
        If False (default), use subfolder layout from cal_nr/cal_phase pipeline:
            {data_2d}/{method}/{method}.txt
            {data_2d}/{method}/graph_{version}/{method}.txt
    """
    if flat:
        nr_path    = os.path.join(data_2d, method + '.txt')
        phase_path = os.path.join(data_2d, 'graph_' + version, method + '.txt')
    else:
        nr_path    = os.path.join(data_2d, method, method + '.txt')
        phase_path = os.path.join(data_2d, method, 'graph_' + version, method + '.txt')

    if not os.path.exists(nr_path):
        raise FileNotFoundError('NR metric file not found: ' + nr_path)
    if not os.path.exists(phase_path):
        raise FileNotFoundError('Phase metric file not found: ' + phase_path)

    nr    = pd.read_csv(nr_path,    sep='\t', engine='python', encoding='cp949')
    phase = pd.read_csv(phase_path, sep='\t', engine='python', encoding='cp949')

    # Both files share the same columns.  Extract only the proposed metrics
    # from graph{version} file and rename, then merge on key columns.
    # This matches the original cal_corr3.py logic:
    #   data['Phase_ProPSNR'] = data_new['Phase_CPSNR'].copy()
    #   data['Phase_ProSSIM'] = data_new['Phase_CSSIM'].copy()
    key_cols = [c for c in ['noise', 'quality', 'index'] if c in phase.columns]
    phase_sub = phase[key_cols + ['Phase_CPSNR', 'Phase_CSSIM']].rename(
        columns={'Phase_CPSNR': 'Phase_ProPSNR', 'Phase_CSSIM': 'Phase_ProSSIM'})

    data = pd.merge(nr, phase_sub, on=key_cols, how='inner')
    data['POH generation method'] = method

    return data


def get_bcom_data(data_3d, version):
    """Load B-COM / 3D hologram data.

    Uses average_metric.txt (averaged over all 3D views) instead of
    per-method files.  Corresponds to cal_corr3.py is_3D=1 behavior.

    Parameters
    ----------
    data_3d : str
        Root 3D data directory (--data_3d).
    version : str
        Metric variant ('pda' or 'circular'). Selects graph_{version}/ folder.
        Expected files:
          {data_3d}/bcom.txt                          -- NR metrics
          {data_3d}/graph_{version}/average_metric.txt -- Phase metrics
    """
    nr_path    = os.path.join(data_3d, 'bcom.txt')
    phase_path = os.path.join(data_3d, 'graph_' + version, 'average_metric.txt')

    if not os.path.exists(nr_path):
        raise FileNotFoundError('B-COM NR file not found: ' + nr_path)
    if not os.path.exists(phase_path):
        raise FileNotFoundError('B-COM phase file not found: ' + phase_path)

    nr    = pd.read_csv(nr_path,    sep='\t', engine='python', encoding='cp949')
    phase = pd.read_csv(phase_path, sep='\t', engine='python', encoding='cp949')

    key_cols = [c for c in ['noise', 'quality', 'index'] if c in phase.columns]
    phase_sub = phase[key_cols + ['Phase_CPSNR', 'Phase_CSSIM']].rename(
        columns={'Phase_CPSNR': 'Phase_ProPSNR', 'Phase_CSSIM': 'Phase_ProSSIM'})

    data = pd.merge(nr, phase_sub, on=key_cols, how='inner')
    data['POH generation method'] = 'bcom'

    return data


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def get_rmse(x, y):
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))


def get_ssim_correlation(data, method):
    pcc   = data[['NR_SSIM', 'Phase_SSIM', 'Phase_CSSIM', 'Phase_ProSSIM']].corr(method='pearson')
    srocc = data[['NR_SSIM', 'Phase_SSIM', 'Phase_CSSIM', 'Phase_ProSSIM']].corr(method='spearman')
    labels = ['original', 'circular', 'PDA']
    cols   = ['Phase_SSIM', 'Phase_CSSIM', 'Phase_ProSSIM']

    PCC   = pd.DataFrame({'POH generation method': [method]*3, 'Metric': labels,
                          'correlation coefficient': [pcc['NR_SSIM'][c] for c in cols]})
    SROCC = pd.DataFrame({'POH generation method': [method]*3, 'Metric': labels,
                          'correlation coefficient': [srocc['NR_SSIM'][c] for c in cols]})
    return PCC, SROCC


def get_psnr_correlation(data, method):
    pcc   = data[['NR_PSNR', 'Phase_PSNR', 'Phase_CPSNR', 'Phase_ProPSNR']].corr(method='pearson')
    srocc = data[['NR_PSNR', 'Phase_PSNR', 'Phase_CPSNR', 'Phase_ProPSNR']].corr(method='spearman')
    labels = ['original', 'circular', 'PDA']
    cols   = ['Phase_PSNR', 'Phase_CPSNR', 'Phase_ProPSNR']

    PCC   = pd.DataFrame({'POH generation method': [method]*3, 'Metric': labels,
                          'correlation coefficient': [pcc['NR_PSNR'][c] for c in cols]})
    SROCC = pd.DataFrame({'POH generation method': [method]*3, 'Metric': labels,
                          'correlation coefficient': [srocc['NR_PSNR'][c] for c in cols]})
    return PCC, SROCC


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute PLCC/SROCC for PDA evaluation (saves pcc.txt + data.tsv)')
    parser.add_argument('--data_2d', default=None,
                        help='Root of 2D DPI-DB data directory '
                             '(omit to run 3D-only with --data_3d)')
    parser.add_argument('--methods', nargs='+', default=['DPAC', 'GS', 'SGD', 'NHVC'],
                        help='PoH method names to evaluate (2D)')
    parser.add_argument('--data_3d', default=None,
                        help='Root of 3D B-COM data directory '
                             '(loads bcom.txt + graph{version}/average_metric.txt; '
                             'omit to run 2D-only)')
    parser.add_argument('--out_dir', default='.',
                        help='Output directory for pcc.txt and data.tsv (default: current dir)')
    parser.add_argument('--flat', action='store_true',
                        help='Use flat layout compatible with cal_corr3.py output '
                             '({data_2d}/{method}.txt, {data_2d}/graph_{version}/{method}.txt)')
    parser.add_argument('--version', default='pda', choices=['pda', 'circular'],
                        help="Metric variant; reads graph_{version}/ folder (default: 'pda')")
    parser.add_argument('--no_fit', action='store_true',
                        help='Disable linear scaling of PDA-SSIM (FIT_A=2.7, FIT_B=-1.87)')
    args = parser.parse_args()

    if args.data_2d is None and args.data_3d is None:
        parser.error('At least one of --data_2d or --data_3d must be provided.')

    os.makedirs(args.out_dir, exist_ok=True)

    # Load 2D method data
    Data_t = pd.DataFrame()
    if args.data_2d:
        for method in args.methods:
            try:
                df = get_data(args.data_2d, method, args.version, flat=args.flat)
                Data_t = pd.concat([Data_t, df], ignore_index=True)
                print('[OK] Loaded 2D: ' + method)
            except FileNotFoundError as e:
                print('[WARN] ' + str(e))

    # Load 3D B-COM data
    if args.data_3d:
        try:
            df = get_bcom_data(args.data_3d, args.version)
            Data_t = pd.concat([Data_t, df], ignore_index=True)
            print('[OK] Loaded 3D: bcom')
        except FileNotFoundError as e:
            print('[WARN] ' + str(e))

    if Data_t.empty:
        print('[ERROR] No data loaded. Exiting.')
        return

    # Apply linear fit to PDA-SSIM
    if not args.no_fit:
        Data_t['Phase_ProSSIM'] = np.clip(Data_t['Phase_ProSSIM'] * FIT_A + FIT_B, 0, 1)

    # Compute PLCC / SROCC for each method
    PCC_t   = pd.DataFrame()
    SROCC_t = pd.DataFrame()

    for method in Data_t['POH generation method'].unique():
        sub       = Data_t[Data_t['POH generation method'] == method]
        sub_valid = sub[sub['noise'] != 'ms']

        pcc_s,  srocc_s = get_ssim_correlation(sub_valid, method)
        pcc_p,  srocc_p = get_psnr_correlation(sub_valid, method)
        pcc_s['metric']   = 'ssim';  srocc_s['metric']   = 'ssim'
        pcc_p['metric']   = 'psnr';  srocc_p['metric']   = 'psnr'

        PCC_t   = pd.concat([PCC_t,   pcc_s,  pcc_p],  ignore_index=True)
        SROCC_t = pd.concat([SROCC_t, srocc_s, srocc_p], ignore_index=True)

    PCC_t['Evaluation protocol']   = 'PLCC'
    SROCC_t['Evaluation protocol'] = 'SROCC'

    all_t    = pd.concat([PCC_t, SROCC_t], ignore_index=True)

    # Save pcc.txt (per-method PLCC/SROCC -- input for draw_graph.py)
    pcc_path = os.path.join(args.out_dir, 'pcc.txt')
    all_t.to_csv(pcc_path, sep='\t', index=False, encoding='utf-8')
    print('Saved: ' + pcc_path)

    # Save data.tsv (merged raw data with fitted PDA-SSIM -- input for draw_graph.py)
    data_path = os.path.join(args.out_dir, 'data.tsv')
    Data_t.to_csv(data_path, sep='\t', index=False, encoding='utf-8')
    print('Saved: ' + data_path)

    # Print summary
    print(all_t.groupby(['Evaluation protocol', 'metric', 'Metric'])
               ['correlation coefficient'].mean().unstack('Metric'))


if __name__ == '__main__':
    main()
