"""
Example: Create DPI-DB Dataset

Demonstrates the full DPI-DB (Distortion in Phase Image Database) creation
pipeline:

1. Apply 10 distortion types × 5 quality levels to PoH phase images.
2. (Optional) Convert to YUV and encode with HEVC / VVC.
3. Compute phase-domain metrics (PDA + baseline variants).
4. Compute NR-domain metrics on reconstructed images.
5. Run correlation analysis (PLCC / SROCC) and generate plots.

Usage
-----
python example_create_dataset.py \\
    --poh_path  ./data/poh \\
    --data_path ./data/dpi_db \\
    --methods   SGD NHVC \\
    --steps     distort phase_metrics evaluate
"""

import argparse
import subprocess
import sys
import os


STEPS = ['distort', 'phase_metrics', 'nr_metrics', 'evaluate']


def run(cmd):
    print(f'\n>>> {" ".join(cmd)}')
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f'[ERROR] Command failed with exit code {result.returncode}')
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end DPI-DB dataset creation and evaluation')
    parser.add_argument('--poh_path',  required=True,
                        help='Directory with PoH phase images (ori/{red,green,blue}/*.png)')
    parser.add_argument('--data_path', required=True,
                        help='Root output directory for DPI-DB')
    parser.add_argument('--methods', nargs='+', default=['SGD'],
                        help='PoH generation method names')
    parser.add_argument('--steps', nargs='+', default=STEPS, choices=STEPS,
                        help='Pipeline steps to run')
    parser.add_argument('--distortions', nargs='+', default=None,
                        help='Specific distortions (default: all 10)')
    parser.add_argument('--qualities', nargs='+', type=int, default=None,
                        help='Quality levels 0-4 (default: all)')
    args = parser.parse_args()

    methods_arg = args.methods

    # -----------------------------------------------------------------------
    # Step 1: Apply distortions
    # -----------------------------------------------------------------------
    if 'distort' in args.steps:
        cmd = [sys.executable, 'make_dataset/add_distortion.py',
               '--data_path', args.poh_path,
               '--methods'] + methods_arg
        if args.distortions:
            cmd += ['--distortions'] + args.distortions
        if args.qualities:
            cmd += ['--qualities'] + [str(q) for q in args.qualities]
        run(cmd)

    # -----------------------------------------------------------------------
    # Step 2: Phase-domain metrics (PDA + baselines)
    # -----------------------------------------------------------------------
    if 'phase_metrics' in args.steps:
        run([sys.executable, 'evaluation/cal_phase_metrics.py',
             '--data_path', args.data_path,
             '--methods'] + methods_arg)

    # -----------------------------------------------------------------------
    # Step 3: NR-domain metrics (standard SSIM/PSNR on reconstructed images)
    # -----------------------------------------------------------------------
    if 'nr_metrics' in args.steps:
        print('\n[INFO] NR metrics require reconstructed images in {method}/ori/recon/')
        print('       Reconstruct PoH images via ASM propagation before running this step.')
        print('       Reference: Y. Peng et al., Neural Holography (SIGGRAPH Asia 2020)')
        print('       https://github.com/computational-imaging/neural-holography')
        run([sys.executable, 'evaluation/cal_nr_metrics.py',
             '--data_path', args.data_path,
             '--methods'] + methods_arg)

    # -----------------------------------------------------------------------
    # Step 4: Correlation analysis and plots
    # -----------------------------------------------------------------------
    if 'evaluate' in args.steps:
        run([sys.executable, 'evaluation/evaluate.py',
             '--data_path', args.data_path,
             '--methods'] + methods_arg + ['--out_dir', args.data_path])

    print('\nDone.')


if __name__ == '__main__':
    main()
