if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 0. CONFIG
# ==========================================
DETECTOR_NAMES = ['ETD', 'T1', 'T2']
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_metrics_csv_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), "evaluation_results"))
args = parser.parse_args()

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.4
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['ytick.major.size'] = 0.0


# 1. SETUP
# ==========================================
os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_csv(args.evaluation_metrics_csv_path)
df_noisy = df[df['condition'] == 'noisy'].copy()

df_noisy['mean_baseline_psnr'] = df_noisy.groupby(['sample', 'detector', 'sigma'])['baseline_psnr'].transform('mean')
df_noisy['mean_baseline_ssim'] = df_noisy.groupby(['sample', 'detector', 'sigma'])['baseline_ssim'].transform('mean')


# 2. PLOTTING
# ==========================================
def create_sample_palette(df_det, value_col):
    sample_means = df_det.groupby('sample')[value_col].max().sort_values(ascending=False)
    ordered_samples = sample_means.index
    return {s: f'#00{v:02x}{(255-v//3):02x}' for s, v in zip(ordered_samples, np.linspace(0, 170, len(ordered_samples)).astype(int))}

for det in DETECTOR_NAMES:
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.9))
    df_det = df_noisy[df_noisy['detector'] == det]

    # psnr plot
    ax_psnr = axes[0]
    palette_psnr = create_sample_palette(df_det, 'recon_psnr')

    sns.lineplot(data=df_det, x='mean_baseline_psnr', y='recon_psnr', hue='sample', palette=palette_psnr, errorbar='ci', ax=ax_psnr, legend=False, linewidth=0.5, alpha=0.7, marker='o', markersize=1.5)
    ax_psnr.axline((0, 0), slope=1, color='orangered', linestyle=':', linewidth=0.7)

    ax_psnr.set_xlabel(fr'Input PSNR [dB]: $\mathbf{{{det}}}$ detector')
    ax_psnr.set_ylabel('Output PSNR [dB]')
    ax_psnr.set_xlim(df_noisy['mean_baseline_psnr'].max(), df_noisy['mean_baseline_psnr'].min())
    ax_psnr.set_ylim(df_noisy['recon_psnr'].min() * 0.98, df_noisy['recon_psnr'].max() * 1.02)
    ax_psnr.set_xlim(60.14522215332777, 6.361647584890786)
    ax_psnr.set_ylim(9.338883309315868, 37.99283835054405)

    ax_psnr.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    fig.canvas.draw()
    left_psnr, right_psnr = ax_psnr.get_xlim()
    span_psnr = abs(left_psnr - right_psnr)
    internal_ticks_psnr = [t for t in ax_psnr.get_xticks() if abs(t - left_psnr) > 0.08 * span_psnr and abs(t - right_psnr) > 0.08 * span_psnr and min(left_psnr, right_psnr) < t < max(left_psnr, right_psnr)]
    ax_psnr.set_xticks([left_psnr] + internal_ticks_psnr + [right_psnr])
    fig.canvas.draw()
    for label in ax_psnr.get_xticklabels():
        pos = label.get_position()[0]
        if np.isclose(pos, left_psnr, rtol=1e-4):
            label.set_ha('left')
        elif np.isclose(pos, right_psnr, rtol=1e-4):
            label.set_ha('right')

    # ssim plot
    ax_ssim = axes[1]
    ordered_samples_ssim = df_det.groupby('sample')['recon_ssim'].max().sort_values(ascending=False).index
    palette_ssim = create_sample_palette(df_det, 'recon_ssim')

    sns.lineplot(data=df_det, x='mean_baseline_ssim', y='recon_ssim', hue='sample', palette=palette_ssim, errorbar='ci', ax=ax_ssim, legend=False, linewidth=0.5, alpha=0.7, marker='o', markersize=1.5)
    ax_ssim.axline((0, 0), slope=1, color='orangered', linestyle=':', linewidth=0.7)

    ax_ssim.set_xlabel(fr'Input SSIM: $\mathbf{{{det}}}$ detector')
    ax_ssim.set_ylabel('Output SSIM')
    ax_ssim.set_xlim(df_noisy['mean_baseline_ssim'].max(), df_noisy['mean_baseline_ssim'].min())
    ax_ssim.set_ylim(df_noisy['recon_ssim'].min() * 0.98, df_noisy['recon_ssim'].max() * 1.02)

    fig.canvas.draw()
    left_ssim, right_ssim = ax_ssim.get_xlim()
    span_ssim = abs(left_ssim - right_ssim)
    internal_ticks_ssim = [t for t in ax_ssim.get_xticks() if abs(t - left_ssim) > 0.08 * span_ssim and abs(t - right_ssim) > 0.08 * span_ssim and min(left_ssim, right_ssim) < t < max(left_ssim, right_ssim)]
    ax_ssim.set_xticks([left_ssim] + internal_ticks_ssim + [right_ssim])
    fig.canvas.draw()
    for label in ax_ssim.get_xticklabels():
        pos = label.get_position()[0]
        if np.isclose(pos, left_ssim, rtol=1e-4):
            label.set_ha('left')
        elif np.isclose(pos, right_ssim, rtol=1e-4):
            label.set_ha('right')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"evaluation_results_{det}.pdf"), format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()
