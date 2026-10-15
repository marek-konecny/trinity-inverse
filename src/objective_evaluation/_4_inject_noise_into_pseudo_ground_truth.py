if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import glob
import json
import argparse
import tifffile
import numpy as np
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _1_constants import TIFF_MAX_INTENSITY_VALUE

# 0. CONFIG
# ==========================================
DETECTOR_NAMES = ['ETD', 'T1', 'T2']
parser = argparse.ArgumentParser()
parser.add_argument('--pgt_dir', type=str, default=os.path.join(os.path.dirname(__file__), "pseudo_ground_truth"))
parser.add_argument('--noise_steps', type=int, default=30, help="Steps for noise injection")
parser.add_argument('--noise_realizations', type=int, default=15, help="Independent noise realizations per step")
parser.add_argument('--noise_sigma_max', type=float, default=2.0)
parser.add_argument('--noise_model_coefficients_json', type=str, default=os.path.join(os.path.dirname(__file__), "pseudo_ground_truth", "noise_model_coefficients.json"))
args = parser.parse_args()

args.sigmas = (np.linspace(0, np.sqrt(args.noise_sigma_max), args.noise_steps) ** 2)[1:]

noise_coeffs = None
global_max_base_sigma = 1.0
if args.noise_model_coefficients_json and os.path.exists(args.noise_model_coefficients_json):
    with open(args.noise_model_coefficients_json, "r") as f:
        noise_coeffs = json.load(f)

    x_eval = np.linspace(0, 1.0, 1000)
    max_sigmas = [np.max(np.sqrt(np.maximum(np.polyval(p, x_eval), 0.0))) for p in noise_coeffs.values()]
    if max_sigmas:
        global_max_base_sigma = max(max_sigmas)

    print(f"Loaded noise scaling models for: {list(noise_coeffs.keys())}")


# 1. NOISE INJECTION
# ==========================================
def _inject_noise_for_sample(sample_dir):
    clean_dir = os.path.join(sample_dir, "clean")
    if not os.path.isdir(clean_dir):
        return

    clean_imgs = {}
    for det in DETECTOR_NAMES:
        img_path = os.path.join(clean_dir, f"{det}.tiff")
        if os.path.exists(img_path):
            clean_imgs[det] = tifffile.imread(img_path).astype(np.float64)
        else:
            return

    for sigma in args.sigmas:
        for realization in range(args.noise_realizations):
            noisy_dir = os.path.join(sample_dir, f"noisy_sigma{sigma:.3f}_rel{realization:02d}")
            os.makedirs(noisy_dir, exist_ok=True)

            for det in DETECTOR_NAMES:
                pgt_cropped = clean_imgs[det]
                normalized_I = pgt_cropped / TIFF_MAX_INTENSITY_VALUE

                if noise_coeffs and det in noise_coeffs:
                    p = noise_coeffs[det]
                    base_variance = np.polyval(p, normalized_I)
                    base_sigma = np.sqrt(np.maximum(base_variance, 0.0))
                    local_sigma = sigma * (base_sigma / global_max_base_sigma)
                else:
                    local_sigma = sigma * np.sqrt(normalized_I)

                noise = np.random.normal(0, local_sigma * TIFF_MAX_INTENSITY_VALUE, size=pgt_cropped.shape)
                noisy_img = np.clip(np.round(pgt_cropped + noise), 0, TIFF_MAX_INTENSITY_VALUE).astype(np.uint16)
                tifffile.imwrite(os.path.join(noisy_dir, f"{det}.tiff"), noisy_img)

sample_dirs = [d for d in glob.glob(os.path.join(args.pgt_dir, "*")) if os.path.isdir(d)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(_inject_noise_for_sample, sample_dir) for sample_dir in sample_dirs]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Injecting noise per sample"):
        future.result()
