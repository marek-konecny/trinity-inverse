if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import json
import hashlib
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 0. CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--preinferred_latents_root', type=str, required=True)
parser.add_argument('--output_pdf_path', type=str, default="learned_detector_parameters.pdf")
args = parser.parse_args()

DETECTOR_NAMES = ["ETD", "T1", "T2"]
SPECIMEN_NAME_TO_FOLD_MAP = {
    "full-headlights": "fold 6",
    "BiNi12-DSC": "fold 2",
    "14MoSiO2": "fold 1"
}

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.4
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['ytick.major.size'] = 0.0


# 1. SETUP & DATA COLLECTION
# ==========================================
seen_hashes = set()
sample_names = []

data_norm_px = [[], [], []]
data_norm_py = [[], [], []]
data_p_mag = [[], [], []]
data_alpha = [[], [], []]
data_w = [[], [], []]
data_b = [[], [], []]
data_beta = []

for item in sorted(os.listdir(args.preinferred_latents_root)):
    full_path = os.path.join(args.preinferred_latents_root, item)
    params_path = os.path.join(full_path, "params.json")
    latents_path = os.path.join(full_path, "latents.npz")

    with open(params_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash in seen_hashes:
        continue
    seen_hashes.add(file_hash)

    with open(params_path, "r") as parameters_file:
        parameters = json.load(parameters_file)

    det_p_vectors = np.array(parameters["det_p_vectors"], dtype=np.float32)
    alpha_logits = np.array(parameters["det_alpha_logits"], dtype=np.float32).flatten()
    det_alpha = 1.0 / (1.0 + np.exp(-alpha_logits))
    det_se_w = np.array(parameters["det_se_w"], dtype=np.float32).flatten()
    det_se_b = np.array(parameters["det_se_b"], dtype=np.float32).flatten()

    sample_names.append(item)

    for i in range(3):
        px = det_p_vectors[i, 0]
        py = det_p_vectors[i, 1]
        magnitude = np.hypot(px, py)
        safe_magnitude = max(magnitude, 1e-12)

        data_norm_px[i].append(px / safe_magnitude)
        data_norm_py[i].append(py / safe_magnitude)
        data_p_mag[i].append(magnitude)

        data_alpha[i].append(det_alpha[i])
        data_w[i].append(det_se_w[i])
        data_b[i].append(det_se_b[i])

    se_beta = np.array(parameters["se_beta"], dtype=np.float32).flatten()[0]
    data_beta.append(se_beta)

if not data_beta:
    raise ValueError("No parameter data found.")


# 2. PLOTTING
# ==========================================
fig, axes = plt.subplots(1, 6, figsize=(5.5, 3), sharey=True)

def plot_grouped_box(ax, data, xlabel_text, labels):
    bplot = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=labels,
        medianprops=dict(color='orangered', linewidth=1),
        flierprops=dict(marker='o', markersize=3, markeredgecolor='blue', markerfacecolor='blue', markeredgewidth=0),
        widths=0.8,
        capwidths=0.5,
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5)
    )

    for patch in bplot['boxes']:
        patch.set_facecolor('#FFFFFF')
        patch.set_linewidth(0.5)

    for i, flier in enumerate(bplot['fliers']):
        y_data = flier.get_ydata()
        x_data = flier.get_xdata()

        if len(y_data) == 0:
            continue

        detector_name = labels[i] if len(labels) > 1 else labels[0]

        for x, y in zip(x_data, y_data):
            original_index = np.argmin(np.abs(np.array(data[i]) - y))
            dir_name = sample_names[original_index]
            print(f"Outlier {xlabel_text} | {detector_name} | Value: {y:.4f} | Dir: {dir_name}")

            fold_label = next((fold for key, fold in SPECIMEN_NAME_TO_FOLD_MAP.items() if key in dir_name), "")
            offset_pos = (0.2, 11) if y > np.median(data[i]) or ("$p_y$" in xlabel_text and detector_name == "T2") else (0.2, -11)

            ax.annotate(
                fold_label,
                (x, y),
                xytext=offset_pos,
                textcoords="offset points",
                color='blue',
                va='center',
                ha='center',
                rotation=90,
            )
    if len(labels) == 1:
        ax.set_xlim(-0.5, 2.5)

    ax.spines['left'].set_color('#DDDDDD')
    ax.tick_params(axis='x', bottom=False, pad=0.5)
    ax.set_ylim(-1, 2.1)
    ax.set_xlabel(xlabel_text)

    shift_right = mpl.transforms.ScaledTranslation(4/72, 0, ax.figure.dpi_scale_trans)
    shift_left = mpl.transforms.ScaledTranslation(-4/72, 0, ax.figure.dpi_scale_trans)

    for label in ax.get_xticklabels():
        if label.get_text() == "ETD":
            label.set_transform(label.get_transform() + shift_right)
        elif label.get_text() == "T2":
            label.set_transform(label.get_transform() + shift_left)

axNum = 0
plot_grouped_box(axes[axNum], data_norm_px, r'BSE $p_x$', DETECTOR_NAMES); axNum += 1
plot_grouped_box(axes[axNum], data_norm_py, r'BSE $p_y$', DETECTOR_NAMES); axNum += 1
plot_grouped_box(axes[axNum], data_alpha, r'Coefficient $\alpha$', DETECTOR_NAMES); axNum += 1
plot_grouped_box(axes[axNum], [data_beta], r'Coefficient $\beta$', ["Shared"]); axNum += 1
plot_grouped_box(axes[axNum], data_w, r'Coefficient $w$', DETECTOR_NAMES); axNum += 1
plot_grouped_box(axes[axNum], data_b, r'Coefficient $b$', DETECTOR_NAMES); axNum += 1

for i in range(1, len(axes)):
    axes[i].tick_params(axis='y', left=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0)
plt.savefig(args.output_pdf_path, format="pdf", bbox_inches='tight', pad_inches=0)
