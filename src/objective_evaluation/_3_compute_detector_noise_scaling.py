if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import json
import argparse
import tifffile
import numpy as np
import scipy.ndimage
import scipy.optimize
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _1_constants import TIFF_MAX_INTENSITY_VALUE
from _2_data_loading_helpers import load_intra_sample_registration_data, parse_data_structure

# 0. CONFIG
# ==========================================
DETECTOR_NAMES = ['ETD', 'T1', 'T2']
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), "evaluation_results"))
parser.add_argument('--intra_sample_registration_data_csv_path', type=str, required=True)
parser.add_argument('--pgt_dir', type=str, default=os.path.join(os.path.dirname(__file__), "pseudo_ground_truth"))
parser.add_argument('--num_intensity_bins', type=int, default=500)
args = parser.parse_args()

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.4
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.size'] = 0.0
mpl.rcParams['ytick.major.size'] = 0.0

os.makedirs(args.output_dir, exist_ok=True)


# 1. DATA LOADING
# ==========================================
intra_sample_registration_data = load_intra_sample_registration_data(args.intra_sample_registration_data_csv_path)
all_triplets = parse_data_structure(args.input_data_root, None, intra_sample_registration_data)

samples = {}
for triplet in all_triplets:
    sample_id = triplet.get('sample_group_ID')
    if sample_id is not None:
        if sample_id not in samples:
            samples[sample_id] = []
        samples[sample_id].append(triplet)

def update_crop_bounds(shift, bounds, shape):
    ct, cb, cl, cr = bounds
    H, W = shape
    dy, dx = shift
    ct = max(ct, int(np.ceil(max(0, dy))))
    cb = min(cb, H - int(np.ceil(max(0, -dy))))
    cl = max(cl, int(np.ceil(max(0, dx))))
    cr = min(cr, W - int(np.ceil(max(0, -dx))))
    return ct, cb, cl, cr


# 2. NOISE AGGREGATION
# ==========================================
bins = np.linspace(0, TIFF_MAX_INTENSITY_VALUE, args.num_intensity_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

sum_sq_res = {det: np.zeros(args.num_intensity_bins, dtype=np.float64) for det in DETECTOR_NAMES}
counts = {det: np.zeros(args.num_intensity_bins, dtype=np.float64) for det in DETECTOR_NAMES}

for sample_id, group in tqdm(samples.items(), desc="Analyzing noise scaling per sample"):
    specimen_name = os.path.basename(group[0]['specimen_dir'])
    sample_folder_name = f"{specimen_name}_sample{sample_id:02d}"
    clean_dir = os.path.join(args.pgt_dir, sample_folder_name, "clean")

    if not os.path.isdir(clean_dir):
        continue

    shifts = []
    for triplet in group:
        reg_info = intra_sample_registration_data[triplet['ID']]
        shifts.append((reg_info['shift_Y'], reg_info['shift_X']))

    sample_img_shape = tifffile.imread(group[0][DETECTOR_NAMES[0]]).shape
    H, W = sample_img_shape
    crop_bounds = (0, H, 0, W)
    for shift in shifts:
        crop_bounds = update_crop_bounds(shift, crop_bounds, (H, W))
    ct, cb, cl, cr = crop_bounds

    for det in DETECTOR_NAMES:
        pgt_clean_path = os.path.join(clean_dir, f"{det}.tiff")
        if not os.path.exists(pgt_clean_path):
            continue

        pgt_clean = tifffile.imread(pgt_clean_path).astype(np.float64)

        bin_indices = np.digitize(pgt_clean, bins) - 1
        bin_indices = np.clip(bin_indices, 0, args.num_intensity_bins - 1)

        for idx, triplet in enumerate(group):
            raw_img = tifffile.imread(triplet[det]).astype(np.float64)
            shift = shifts[idx]
            shifted_raw = scipy.ndimage.shift(raw_img, shift=shift, mode='constant', cval=0.0)
            cropped_raw = shifted_raw[ct:cb, cl:cr]

            residual = cropped_raw - pgt_clean
            squared_residual = residual ** 2

            np.add.at(sum_sq_res[det], bin_indices, squared_residual)
            np.add.at(counts[det], bin_indices, 1)


# 3. REGRESSION & PLOTTING
# ==========================================
def poly_model(p, x): return np.polyval(p, x)
def poly_residuals(p, x, y): return poly_model(p, x) - y

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = {'ETD': 'blue', 'T1': '#3ABDD4', 'T2': 'orangered'}

all_y_data = []
for det in DETECTOR_NAMES:
    valid = counts[det] > 0
    if np.any(valid):
        y_det = (sum_sq_res[det][valid] / counts[det][valid]) / (TIFF_MAX_INTENSITY_VALUE ** 2)
        all_y_data.append(y_det)
concat_y_data = np.concatenate(all_y_data)
global_y_min = np.min(concat_y_data)
global_y_max = np.percentile(concat_y_data, 99)

model_coeffs = {}

for det in DETECTOR_NAMES:
    valid = counts[det] > 0
    x_data = bin_centers[valid] / TIFF_MAX_INTENSITY_VALUE

    # normalize variance
    y_data = (sum_sq_res[det][valid] / counts[det][valid]) / (TIFF_MAX_INTENSITY_VALUE ** 2)

    p0 = np.polyfit(x_data, y_data, 3)
    res = scipy.optimize.least_squares(
        poly_residuals,
        p0,
        args=(x_data, y_data),
        loss='soft_l1'
    )
    model_coeffs[det] = res.x.tolist()
    y_fit = poly_model(res.x, x_data)

    ax.plot(x_data, y_data, color=colors[det], alpha=0.4, linewidth=0, marker='o', markersize=1, markeredgewidth=0)
    ax.plot(x_data, y_fit, label=det, color=colors[det], linewidth=0.9)

    # curve annotations
    visible_mask = y_fit <= global_y_max
    max_idx = np.where(visible_mask)[0][np.argmax(y_fit[visible_mask])] if np.any(visible_mask) else np.argmax(y_fit)
    ax.annotate(det, (x_data[max_idx], y_fit[max_idx]), xytext=(0, 6), textcoords="offset points", color=colors[det], ha='center', va='bottom')

ax.set_ylabel(r"Average per-detector noise variance $\sigma^2_{s,d}$")
ax.set_xlabel(r"Clean pseudo-ground-truth $I_{s,d}^{*}$ intensity")
ax.set_xlim(0, 1)
ax.set_ylim(global_y_min, global_y_max)

# force edge ticks
fig.canvas.draw()
span_x = 1.0
internal_ticks_x = [t for t in ax.get_xticks() if t > 0.08 and t < 0.92]
ax.set_xticks([0.0] + internal_ticks_x + [1.0])

span_y = global_y_max - global_y_min
internal_ticks_y = [t for t in ax.get_yticks() if (t - global_y_min) > 0.08 * span_y and (global_y_max - t) > 0.08 * span_y]
ax.set_yticks([global_y_min] + internal_ticks_y + [global_y_max])

# align x-axis edge ticks
fig.canvas.draw()
for label in ax.get_xticklabels():
    pos = label.get_position()[0]
    if np.isclose(pos, 0.0, rtol=1e-4, atol=1e-4):
        label.set_ha('left')
    elif np.isclose(pos, 1.0, rtol=1e-4, atol=1e-4):
        label.set_ha('right')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "noise_scaling_analysis.pdf"), format="pdf", bbox_inches="tight", pad_inches=0)
plt.close()
with open(os.path.join(args.pgt_dir, "noise_model_coefficients.json"), "w") as f:
    json.dump(model_coeffs, f, indent=4)

print(f"Saved noise scaling plot to {os.path.join(args.output_dir, 'noise_scaling_analysis.pdf')}")
print(f"Saved noise model coefficients to {os.path.join(args.pgt_dir, 'noise_model_coefficients.json')}")
print(json.dumps(model_coeffs, indent=4))
