"""
PDA Holographic Metric Visualization

Reads pre-computed outputs from evaluate.py and produces:

  1. scatter.png          -- 2x3 scatter plot (Figure 7 style)
                             Phase_PSNR / C-PSNR / PDA-PSNR  (top row)
                             Phase_SSIM / C-SSIM / PDA-SSIM  (bottom row)
                             per-noise-type trajectories with 95% CI + PLCC/SROCC annotations

  2. catplot_psnr.png     -- PLCC & SROCC per metric per CGH method (PSNR)
  3. catplot_ssim.png     -- PLCC & SROCC per metric per CGH method (SSIM)

Inputs (produced by evaluate.py)
---------------------------------
  pcc.txt   -- tab-separated PLCC/SROCC table
  data.tsv  -- tab-separated raw merged data with fitted PDA-SSIM

Usage
-----
python draw_graph.py --pcc ./pcc.txt --data ./data.tsv --out_dir .
python draw_graph.py --no_scatter        # catplots only
python draw_graph.py --no_catplot        # scatter only
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Visualization constants
# ---------------------------------------------------------------------------
NOISES     = ['wn', 'gb', 'sp', 'cc', 'no', 'un',
               'block_gb', 'block_ms', 'block_cc', 'hevc', 'vvc', 'hevc2']
CODEC      = ['hevc2', 'hevc', 'vvc']
LEGENDS    = ['wn', 'gb', 'sp', 'cc', 'no', 'un',
               'block_gb', 'block_ms', 'block_cc', 'hevc', 'vvc', 'hevc for POH']

FONTSIZE       = 13
LABEL_FONTSIZE = 13
TITLE_FONTSIZE = 50
LEGENDS_SIZE   = 8
FONT           = {'family': 'Cambria', 'size': LABEL_FONTSIZE}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ci_band(df_grouped, col):
    """Return (mean, lower_95ci, upper_95ci) for one column of a grouped frame."""
    mean  = df_grouped[col]['mean']
    std   = df_grouped[col]['std']
    count = df_grouped[col]['count']
    ci    = 1.96 * std / np.sqrt(count)
    return mean, mean - ci, mean + ci


def _sorted_lists(x_series, y_mean, y_lo, y_hi):
    """Zip four series, sort by x, and return four plain lists."""
    zipped = sorted(zip(x_series, y_mean, y_lo, y_hi))
    return ([d[0] for d in zipped], [d[1] for d in zipped],
            [d[2] for d in zipped], [d[3] for d in zipped])


# ---------------------------------------------------------------------------
# Scatter plot  (2x3 phase-domain metrics vs NR metrics)
# ---------------------------------------------------------------------------

def draw_scatter(data_t, pcc, out_dir):
    """Draw 2x3 scatter plot of phase-domain metrics vs NR reference metrics.

    Parameters
    ----------
    data_t : pd.DataFrame
        Merged data from data.tsv (evaluate.py output).
        Required columns: noise, quality, NR_PSNR, NR_SSIM,
        Phase_PSNR, Phase_CPSNR, Phase_ProPSNR,
        Phase_SSIM, Phase_CSSIM, Phase_ProSSIM.
    pcc : pd.DataFrame
        Correlation table from pcc.txt (evaluate.py output).
        Required columns: Metric, metric, Evaluation protocol,
        correlation coefficient, POH generation method.
    out_dir : str
        Directory where scatter.png is saved.
    """
    data = data_t[data_t['noise'] != 'ms'].copy()

    # --- Build averaged PLCC / SROCC from pcc for corner annotations ---
    def _avg_corr(metric_type, rename_map):
        df = pcc[pcc['metric'] == metric_type].copy()
        df = df[df['Evaluation protocol'].isin(['PLCC', 'SROCC'])]
        df['Metric'] = df['Metric'].map(rename_map).fillna(df['Metric'])
        return (df.groupby(['Metric', 'Evaluation protocol'])
                  ['correlation coefficient'].mean())

    psnr_avg = _avg_corr('psnr', {'original': 'PSNR',
                                   'circular': 'C-PSNR',
                                   'PDA':      'PDA-PSNR'})
    ssim_avg = _avg_corr('ssim', {'original': 'SSIM',
                                   'circular': 'C-SSIM',
                                   'PDA':      'PDA-SSIM'})

    # --- Figure layout: 2 rows x 3 cols ---
    fig = plt.figure(figsize=(10, 7), dpi=200)
    gs_layout = gridspec.GridSpec(nrows=2, ncols=3,
                                  height_ratios=[1, 1], width_ratios=[1, 1, 1])
    axs = [[plt.subplot(gs_layout[i * 3 + j]) for j in range(3)] for i in range(2)]

    n_color = len(LEGENDS)
    colors  = plt.cm.nipy_spectral(np.linspace(0, 1, n_color))
    alpha   = 0.2
    noise_label_map = dict(zip(NOISES, LEGENDS))

    plt.rc('font', **FONT)
    sns.set_style('whitegrid')

    # --- Draw per-noise trajectories ---
    for i, noise in enumerate(NOISES):
        df_n = data[data['noise'] == noise]
        if df_n.empty:
            continue

        df_g = df_n.groupby(['quality']).agg(['mean', 'std', 'count'])

        nr_psnr_mean, nr_psnr_lo, nr_psnr_hi = _ci_band(df_g, 'NR_PSNR')
        nr_ssim_mean, nr_ssim_lo, nr_ssim_hi = _ci_band(df_g, 'NR_SSIM')

        # Row 0: PSNR metrics
        psnr_cols = ['Phase_PSNR', 'Phase_CPSNR', 'Phase_ProPSNR']
        for col, ax in zip(psnr_cols, axs[0]):
            x, y, lo, hi = _sorted_lists(df_g[col]['mean'],
                                          nr_psnr_mean, nr_psnr_lo, nr_psnr_hi)
            ax.plot(x, y, color=colors[i])
            ax.fill_between(x, lo, hi, alpha=alpha, color=colors[i], linewidth=0)

        # Row 1: SSIM metrics
        ssim_cols = ['Phase_SSIM', 'Phase_CSSIM', 'Phase_ProSSIM']
        for col, ax in zip(ssim_cols, axs[1]):
            x, y, lo, hi = _sorted_lists(df_g[col]['mean'],
                                          nr_ssim_mean, nr_ssim_lo, nr_ssim_hi)
            ax.plot(x, y, color=colors[i],
                    label=noise_label_map.get(noise, noise))
            ax.fill_between(x, lo, hi, alpha=alpha, color=colors[i], linewidth=0)

    # --- Axis labels and limits ---
    row0_xlabels = ['PSNR (Phase)', 'C-PSNR (Phase)', 'PDA-PSNR (Phase)']
    row1_xlabels = ['SSIM (Phase)', 'C-SSIM (Phase)', 'PDA-SSIM (Phase)']

    for xlabel, ax in zip(row0_xlabels, axs[0]):
        ax.set_xlabel(xlabel, fontdict={'family': 'serif'}, fontsize=LABEL_FONTSIZE)
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 50])
    axs[0][0].set_ylabel('PSNR (NR)', fontdict={'family': 'serif'}, fontsize=LABEL_FONTSIZE)

    for xlabel, ax in zip(row1_xlabels, axs[1]):
        ax.set_xlabel(xlabel, fontdict={'family': 'serif'}, fontsize=LABEL_FONTSIZE)
        ax.set_xlim([-0.4, 1.1])
        ax.set_ylim([-0.4, 1.1])
    axs[1][0].set_ylabel('SSIM (NR)', fontdict={'family': 'serif'}, fontsize=LABEL_FONTSIZE)

    # --- Reference lines ---
    psnr_ref  = list(range(0, 51))
    ssim_ref  = [-0.4, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.2]
    zeros     = [0] * len(ssim_ref)
    kw_ref    = dict(color='gray', linestyle='--', linewidth=1, alpha=0.7)
    for ax in axs[0]:
        ax.plot(psnr_ref, psnr_ref, **kw_ref)
    for ax in axs[1]:
        ax.plot(ssim_ref, ssim_ref,  **kw_ref)
        ax.plot(ssim_ref, zeros,     **kw_ref)
        ax.plot(zeros,   ssim_ref,   **kw_ref)

    # --- PLCC / SROCC corner annotations ---
    def _ann_psnr(ax, key):
        try:
            plcc  = psnr_avg[key]['PLCC']
            srocc = psnr_avg[key]['SROCC']
            ax.text(48.5, 2.3, f'PLCC: {plcc:.4f}\nSROCC: {srocc:.4f}',
                    horizontalalignment='right',
                    fontdict={'family': 'serif', 'fontsize': FONTSIZE})
        except KeyError:
            pass

    def _ann_ssim(ax, key):
        try:
            plcc  = ssim_avg[key]['PLCC']
            srocc = ssim_avg[key]['SROCC']
            ax.text(1.04, -0.33, f'PLCC: {plcc:.4f}\nSROCC: {srocc:.4f}',
                    horizontalalignment='right',
                    fontdict={'family': 'serif', 'fontsize': FONTSIZE})
        except KeyError:
            pass

    _ann_psnr(axs[0][0], 'PSNR')
    _ann_psnr(axs[0][1], 'C-PSNR')
    _ann_psnr(axs[0][2], 'PDA-PSNR')
    _ann_ssim(axs[1][0], 'SSIM')
    _ann_ssim(axs[1][1], 'C-SSIM')
    _ann_ssim(axs[1][2], 'PDA-SSIM')

    # --- Shared legend at bottom ---
    custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(n_color)]
    fig.subplots_adjust(bottom=0.22)
    fig.legend(custom_lines, LEGENDS,
               loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=6, framealpha=0.5, fontsize=11, title='Noise Type',
               columnspacing=1.8, handletextpad=0.9,
               labelspacing=0.5, borderaxespad=1.5)

    out_path = os.path.join(out_dir, 'scatter.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close('all')
    print('Saved: ' + out_path)


# ---------------------------------------------------------------------------
# Catplot  (PLCC / SROCC comparison across CGH methods)
# ---------------------------------------------------------------------------

def draw_catplot(pcc, out_dir):
    """Draw seaborn catplots of PLCC/SROCC per metric per CGH method.

    Produces:
      catplot_psnr.png -- PSNR metrics
      catplot_ssim.png -- SSIM metrics

    Parameters
    ----------
    pcc : pd.DataFrame
        Correlation table from pcc.txt (evaluate.py output).
    out_dir : str
        Directory where PNG files are saved.
    """
    palette = sns.color_palette('tab10')

    def _catplot(metric_type, rename_map, ylabel, fname):
        df = pcc[pcc['metric'] == metric_type].copy()
        df = df[df['Evaluation protocol'].isin(['PLCC', 'SROCC'])]
        df['Metric'] = df['Metric'].map(rename_map).fillna(df['Metric'])

        plt.rc('font', **FONT)
        g = sns.catplot(
            x='Metric', y='correlation coefficient',
            hue='POH generation method', col='Evaluation protocol',
            palette=palette, height=3.5, aspect=0.9,
            kind='point', legend=False, data=df,
            markers='o', sharey=False)

        ax0 = g.axes[0, 0]
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles=handles, labels=labels,
                   loc='lower right', title='CGH method',
                   fontsize=LEGENDS_SIZE, title_fontsize=LEGENDS_SIZE + 1.5)

        g.set_titles('{col_name}', fontsize=TITLE_FONTSIZE)
        g.despine(left=True)
        g.set_ylabels(ylabel, fontsize=15)
        g.set_xlabels('Phase domain metrics', fontsize=15)

        out_path = os.path.join(out_dir, fname)
        g.savefig(out_path, bbox_inches='tight')
        plt.close('all')
        print('Saved: ' + out_path)

    _catplot('psnr',
             {'original': 'PSNR', 'circular': 'C-PSNR', 'PDA': 'PDA-PSNR'},
             'Correlation to NR-PSNR',
             'catplot_psnr.png')

    _catplot('ssim',
             {'original': 'SSIM', 'circular': 'C-SSIM', 'PDA': 'PDA-SSIM'},
             'Correlation to NR-SSIM',
             'catplot_ssim.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Draw scatter and catplot figures from evaluate.py outputs '
                    '(reads pcc.txt + data.tsv, saves scatter.png + catplot_*.png)')
    parser.add_argument('--pcc', default='./pcc.txt',
                        help='Path to pcc.txt produced by evaluate.py (default: ./pcc.txt)')
    parser.add_argument('--data', default='./data.tsv',
                        help='Path to data.tsv produced by evaluate.py (default: ./data.tsv)')
    parser.add_argument('--out_dir', default='.',
                        help='Output directory for PNG figures (default: current dir)')
    parser.add_argument('--no_scatter', action='store_true',
                        help='Skip scatter.png (draw catplots only)')
    parser.add_argument('--no_catplot', action='store_true',
                        help='Skip catplot_*.png (draw scatter only)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pcc.txt (required for both figures)
    if not os.path.exists(args.pcc):
        raise FileNotFoundError('pcc.txt not found: ' + args.pcc)
    pcc = pd.read_csv(args.pcc, sep='\t', engine='python', encoding='utf-8')

    # Scatter plot
    if not args.no_scatter:
        if not os.path.exists(args.data):
            raise FileNotFoundError('data.tsv not found: ' + args.data)
        data_t = pd.read_csv(args.data, sep='\t', engine='python', encoding='utf-8')
        draw_scatter(data_t, pcc, args.out_dir)

    # Catplots
    if not args.no_catplot:
        draw_catplot(pcc, args.out_dir)


if __name__ == '__main__':
    main()
