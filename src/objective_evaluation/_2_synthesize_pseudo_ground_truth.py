if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import argparse
import tifffile
import numpy as np
import scipy.ndimage
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _1_constants import TIFF_MAX_INTENSITY_VALUE
from _2_data_loading_helpers import load_intra_sample_registration_data, parse_data_structure

# 0. CONFIG
# ==========================================
DETECTOR_NAMES = ['ETD', 'T1', 'T2']
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--skipped_specimen_names', type=str, default="SiAl1")
parser.add_argument('--intra_sample_registration_data_csv_path', type=str, required=True)
parser.add_argument('--pgt_dir', type=str, default=os.path.join(os.path.dirname(__file__), "pseudo_ground_truth"))
parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), "evaluation_results"))
args = parser.parse_args()
args.skipped_specimen_names = [n.strip() for n in args.skipped_specimen_names.split(',')]

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid.axis'] = 'x'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.4
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.size'] = 0.0
mpl.rcParams['ytick.major.size'] = 0.0
#mpl.rcParams["text.usetex"] = True


# 1. SETUP & DATA LOADING
# ==========================================
intra_sample_registration_data = load_intra_sample_registration_data(args.intra_sample_registration_data_csv_path)
all_triplets = parse_data_structure(args.input_data_root, None, intra_sample_registration_data)

samples = {}
for triplet in all_triplets:
    if any(skipped_name in triplet['specimen_dir'] for skipped_name in args.skipped_specimen_names):
        continue
    sample_id = triplet.get('sample_group_ID')
    if sample_id is not None:
        if sample_id not in samples:
            samples[sample_id] = []
        samples[sample_id].append(triplet)

os.makedirs(args.pgt_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)


# 2. PGT SYNTHESIS & OTSU THRESHOLDING
# ==========================================
def update_crop_bounds(shift, bounds, shape):
    ct, cb, cl, cr = bounds
    H, W = shape
    dy, dx = shift
    ct = max(ct, int(np.ceil(max(0, dy))))
    cb = min(cb, H - int(np.ceil(max(0, -dy))))
    cl = max(cl, int(np.ceil(max(0, dx))))
    cr = min(cr, W - int(np.ceil(max(0, -dx))))
    return ct, cb, cl, cr

def _preprocess_single_sample(sample_id, group):
    group_snrs = []
    shifts = []
    for triplet in group:
        reg_info = intra_sample_registration_data[triplet['ID']]
        group_snrs.append(reg_info['snr_of_avg_img'])
        shifts.append((reg_info['shift_Y'], reg_info['shift_X']))

    sample_img_shape = tifffile.imread(group[0][DETECTOR_NAMES[0]]).shape
    H, W = sample_img_shape
    crop_bounds = (0, H, 0, W)
    for shift in shifts:
        crop_bounds = update_crop_bounds(shift, crop_bounds, (H, W))

    sample_results = {}
    for det in DETECTOR_NAMES:
        pgt_numerator = np.zeros(sample_img_shape, dtype=np.float64)
        pgt_denominator = 0.0

        for idx, triplet in enumerate(group):
            img = tifffile.imread(triplet[det])
            shift = shifts[idx]
            shifted_img = scipy.ndimage.shift(img, shift=shift, mode='constant', cval=0.0)
            weight = group_snrs[idx]

            pgt_numerator += shifted_img * weight
            pgt_denominator += weight

        pgt_img = pgt_numerator / pgt_denominator if pgt_denominator > 0 else np.zeros_like(pgt_numerator)
        sample_results[det] = pgt_img

    return sample_id, {
        'specimen_name': os.path.basename(group[0]['specimen_dir']),
        'max_sample_snr': np.max(group_snrs),
        'crop_bounds': crop_bounds,
        'pgt_imgs': sample_results
    }

pipeline_data = {}

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(_preprocess_single_sample, sample_id, group): sample_id for sample_id, group in samples.items()}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preprocessing samples"):
        sample_id, data = future.result()
        pipeline_data[sample_id] = data

print("Otsu-thresholding samples by maximum averaged-image SNR estimate (without outliers)...")
raw_snrs = np.array([pipeline_data[sample_index]['max_sample_snr'] for sample_index in pipeline_data])

(q1, q3) = np.percentile(raw_snrs, [25, 75])
iqr = q3 - q1
inlier_lower_bound = q1 - 1.5*iqr
inlier_upper_bound = q3 + 1.5*iqr
pipeline_data_filtered = {sample_index: data for sample_index, data in pipeline_data.items() if inlier_lower_bound <= data['max_sample_snr'] <= inlier_upper_bound}
filtered_snrs = np.array([data['max_sample_snr'] for data in pipeline_data_filtered.values()])
otsu_thresh = threshold_otsu(filtered_snrs[filtered_snrs > 0])

plt.figure(figsize=(9, 2))
outlier_labels = ["outlier" if (snr < inlier_lower_bound or snr > inlier_upper_bound) else "inlier" for snr in raw_snrs]
sns.stripplot(x=raw_snrs, hue=outlier_labels, palette={"inlier": "blue", "outlier": "orangered"}, jitter=0.25, legend=False)
plt.axvline(otsu_thresh, linestyle='--', linewidth=1, color='blue')
plt.annotate(
    rf"\textit{{high-quality}} sample threshold: ${otsu_thresh:.2f}$" if mpl.rcParams["text.usetex"] else f"high-quality sample threshold: {otsu_thresh:.2f}",
    (otsu_thresh, 0.85),
    xycoords=('data', 'axes fraction'),
    xytext=(5, 0),
    textcoords="offset points",
    color='blue',
    va='center',
    ha='left',
)
plt.xlabel(r"Maximum averaged-image SNR estimate ($\hat{\mathrm{SNR}^{*}}_{\!\!s,t}$) per sample")
plt.xlim(0, None)
plt.gcf().canvas.draw()
plt.gca().get_xticklabels()[0].set_ha('left')
plt.savefig(os.path.join(args.output_dir, "objective_evaluation_otsu_threshold_distribution.pdf"), format="pdf", bbox_inches="tight", pad_inches=0)
plt.close()

valid_samples = []
for sample_index, data in pipeline_data.items():
    if data['max_sample_snr'] >= otsu_thresh:
        valid_samples.append(sample_index)

def _save_clean_sample(sample_id, data):
    base_sample_dir = os.path.join(args.pgt_dir, f"{data['specimen_name']}_sample{sample_id:02d}")
    clean_dir = os.path.join(base_sample_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    ct, cb, cl, cr = data['crop_bounds']

    for det in DETECTOR_NAMES:
        pgt_cropped = data['pgt_imgs'][det][ct:cb, cl:cr]
        tifffile.imwrite(os.path.join(clean_dir, f"{det}.tiff"), pgt_cropped.astype(np.uint16))

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(_save_clean_sample, sample_index, pipeline_data[sample_index]) for sample_index in valid_samples]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Saving clean PGTs"):
        future.result()
