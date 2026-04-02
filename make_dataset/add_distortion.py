"""
DPI-DB Distortion Generator (per-channel PNG format)

Applies 11 distortion types × 5 quality levels to phase-only hologram (PoH) images.
Input structure: {data_path}/{method}/ori/{red|green|blue}/{img_name}.png
Output:          {data_path}/{method}/{distortion}/{quality}/{red|green|blue}/{img_name}.png

Distortions
-----------
wn         : White (Gaussian) noise
un         : Uniform noise
gb         : Gaussian blur
block_gb   : Block-wise Gaussian blur
sp         : Salt-and-pepper (impulse) noise
ms         : Global mean-shifting (global phase offset)
block_ms   : Block-wise mean-shifting
cc         : Contrast change (linear)
block_cc   : Block-wise contrast change
no         : Normalization (range rescaling)

Usage
-----
python add_distortion.py --data_path /data/dpi_db --method SGD --distortions wn gb ms
"""

import argparse
import os
import random

import cv2
import numpy as np
from skimage.util import random_noise

channels = ['red', 'green', 'blue']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(img, folder_path, distortion, quality, channel, img_name):
    save_path = os.path.join(folder_path, distortion, str(quality), channel)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, img_name + '.png'), img)


def _load(folder_path, channel, img_name):
    path = os.path.join(folder_path, 'ori', channel, img_name + '.png')
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Distortion functions
# ---------------------------------------------------------------------------

def white_noise(folder_path, img_names, quality):
    """Gaussian additive noise (sigma randomly sampled per image)."""
    start = [0.003, 0.02, 0.05, 0.09, 0.14]
    end   = [0.02,  0.05, 0.09, 0.14, 0.20]
    values = []
    for img_name in img_names:
        sigma = random.uniform(start[quality], end[quality])
        values.append(sigma)
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64) / 255.
            noise_img = (img + np.random.normal(0, sigma, img.shape)) * 255.
            _save((noise_img % 256).astype(np.uint8), folder_path, 'wn', quality, c, img_name)
    with open(os.path.join(folder_path, 'wn', str(quality), 'info.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in values)


def uniform_noise(folder_path, img_names, quality):
    """Uniform additive noise."""
    start = [0.003, 0.04, 0.10, 0.18, 0.28]
    end   = [0.04,  0.10, 0.18, 0.28, 0.40]
    for img_name in img_names:
        sigma = random.uniform(start[quality], end[quality])
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64) / 255.
            noise_img = (img + np.random.uniform(start[quality], end[quality], img.shape)) * 255.
            _save((noise_img % 256).astype(np.uint8), folder_path, 'un', quality, c, img_name)


def gaussian_blur(folder_path, img_names, quality):
    """Global Gaussian blur."""
    start = [0.3, 0.4, 0.5, 0.6, 0.7]
    end   = [0.4, 0.5, 0.6, 0.7, 0.8]
    values = []
    for img_name in img_names:
        sigma = random.uniform(start[quality], end[quality])
        values.append(sigma)
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            blurred = cv2.GaussianBlur(img, (0, 0), sigma) % 256
            _save(blurred.astype(np.uint8), folder_path, 'gb', quality, c, img_name)
    with open(os.path.join(folder_path, 'gb', str(quality), 'info.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in values)


def block_gaussian_blur(folder_path, img_names, quality):
    """Block-wise Gaussian blur (block_size=120)."""
    levels = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    block_size = 120
    for img_name in img_names:
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            for i in range(img.shape[0] // block_size):
                for j in range(img.shape[1] // block_size):
                    sigma = random.uniform(levels[quality], levels[quality + 1])
                    block = img[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)]
                    img[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)] = \
                        cv2.GaussianBlur(block, (0, 0), sigma)
            _save((img % 256).astype(np.uint8), folder_path, 'block_gb', quality, c, img_name)


def impulse_noise(folder_path, img_names, quality):
    """Salt-and-pepper noise."""
    start = [0.0005, 0.020, 0.035, 0.050, 0.065]
    end   = [0.020,  0.035, 0.050, 0.065, 0.080]
    values = []
    for img_name in img_names:
        sigma = random.uniform(start[quality], end[quality])
        values.append(sigma)
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64) / 255.
            noisy = (random_noise(img, mode='s&p', amount=sigma) * 255) % 256
            _save(noisy.astype(np.uint8), folder_path, 'sp', quality, c, img_name)
    with open(os.path.join(folder_path, 'sp', str(quality), 'info.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in values)


def mean_shifting(folder_path, img_names, quality):
    """Global phase offset (mean-shifting in 256-period domain)."""
    levels = [0, 25, 50, 75, 100, 125]
    values = []
    for img_name in img_names:
        shifted = random.uniform(levels[quality], levels[quality + 1])
        if random.randint(0, 1):
            shifted = -shifted
        values.append(shifted)
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            _save(((img + shifted) % 256).astype(np.uint8), folder_path, 'ms', quality, c, img_name)
    with open(os.path.join(folder_path, 'ms', str(quality), 'info.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in values)


def block_mean_shifting(folder_path, img_names, quality):
    """Block-wise random phase offset (block_size=120)."""
    levels = [0, 25, 50, 75, 100, 125]
    block_size = 120
    for img_name in img_names:
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            for i in range(img.shape[0] // block_size):
                for j in range(img.shape[1] // block_size):
                    shift = random.uniform(levels[quality], levels[quality + 1])
                    if random.randint(0, 1):
                        shift = -shift
                    img[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)] += shift
            _save((img % 256).astype(np.uint8), folder_path, 'block_ms', quality, c, img_name)


def contrast_change(folder_path, img_names, quality):
    """Linear contrast change (alpha * x + gamma)."""
    levels = [0, 12, 24, 37, 50, 64]
    values = []
    for img_name in img_names:
        contrast = random.uniform(levels[quality], levels[quality + 1])
        if random.randint(0, 1):
            contrast = -contrast
        values.append(contrast)
        f_val = 131 * (contrast + 127) / (127 * (131 - contrast))
        gamma = 127 * (1 - f_val)
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            _save((np.clip(f_val * img + gamma, 0, 255) % 256).astype(np.uint8),
                  folder_path, 'cc', quality, c, img_name)
    with open(os.path.join(folder_path, 'cc', str(quality), 'info.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in values)


def block_contrast_change(folder_path, img_names, quality):
    """Block-wise random contrast change (block_size=120)."""
    levels = [0, 12, 24, 37, 50, 64]
    block_size = 120
    for img_name in img_names:
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            for i in range(img.shape[0] // block_size):
                for j in range(img.shape[1] // block_size):
                    sigma = random.uniform(levels[quality], levels[quality + 1])
                    if random.randint(0, 1):
                        sigma = -sigma
                    f_val = 131 * (sigma + 127) / (127 * (131 - sigma))
                    gamma = 127 * (1 - f_val)
                    block = img[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)]
                    img[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)] = \
                        (f_val * block + gamma) % 256
            _save(img.astype(np.uint8), folder_path, 'block_cc', quality, c, img_name)


def normalization(folder_path, img_names, quality):
    """Range normalization (min-max scaling with random target range)."""
    levels = [15, 30, 45, 60, 75, 90]
    values = []
    for img_name in img_names:
        gamma = random.uniform(levels[quality], levels[quality + 1]) + 255
        if random.randint(0, 1):
            gamma = -gamma + 510
        values.append(gamma)
        for c in channels:
            img = _load(folder_path, c, img_name).astype(np.float64)
            minval, maxval = img.min(), img.max()
            if minval != maxval:
                img -= minval
                img *= (gamma / (maxval - minval))
            _save((img % 256).astype(np.uint8), folder_path, 'no', quality, c, img_name)
    with open(os.path.join(folder_path, 'no', str(quality), 'info.txt'), 'w') as f:
        f.writelines(f'{v}\n' for v in values)


# ---------------------------------------------------------------------------
# Distortion registry
# ---------------------------------------------------------------------------

DISTORTIONS = {
    'wn':       white_noise,
    'un':       uniform_noise,
    'gb':       gaussian_blur,
    'block_gb': block_gaussian_blur,
    'sp':       impulse_noise,
    'ms':       mean_shifting,
    'block_ms': block_mean_shifting,
    'cc':       contrast_change,
    'block_cc': block_contrast_change,
    'no':       normalization,
}

DEFAULT_DISTORTIONS = list(DISTORTIONS.keys())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Apply distortions to PoH phase images (DPI-DB format)')
    parser.add_argument('--data_path', required=True,
                        help='Root of DPI-DB: {data_path}/{method}/ori/{red|green|blue}/*.png')
    parser.add_argument('--methods', nargs='+', default=['DPAC', 'GS', 'SGD', 'NHVC'],
                        help='PoH generation methods (subdirectory names)')
    parser.add_argument('--distortions', nargs='+', default=DEFAULT_DISTORTIONS,
                        choices=DEFAULT_DISTORTIONS, help='Distortion types to apply')
    parser.add_argument('--qualities', nargs='+', type=int, default=list(range(5)),
                        choices=range(5), metavar='Q', help='Quality levels 0–4')
    args = parser.parse_args()

    for method in args.methods:
        folder_path = os.path.join(args.data_path, method)
        ori_dir = os.path.join(folder_path, 'ori', channels[0])
        if not os.path.isdir(ori_dir):
            print(f'[WARN] Skipping {method}: {ori_dir} not found.')
            continue
        img_names = [f.split('.')[0] for f in os.listdir(ori_dir) if f.endswith('.png')]

        for distortion in args.distortions:
            fn = DISTORTIONS[distortion]
            for q in args.qualities:
                print(f'[{method}] {distortion} quality={q}')
                fn(folder_path, img_names, q)


if __name__ == '__main__':
    main()
