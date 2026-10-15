if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import sys
import glob
import torch
import argparse
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from _1_constants import TIFF_MAX_INTENSITY_VALUE, PYTORCH_DEVICE
from _4_model_architecture import TrinityInverseModel
from tools.utils import inference_from_latents

# 0. CONFIG
# ==========================================
DETECTOR_NAMES = ['ETD', 'T1', 'T2']
parser = argparse.ArgumentParser()
parser.add_argument('--pseudo_ground_truth_root_dir', type=str, default=os.path.join(os.path.dirname(__file__), "pseudo_ground_truth"))
parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), "evaluation_results"))
parser.add_argument('--inference_checkpoint_path', type=str, required=True)
args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
print(f"pytorch device: {PYTORCH_DEVICE}")

# 1. SETUP
# ==========================================
os.makedirs(args.output_dir, exist_ok=True)
images_output_dir = os.path.join(args.output_dir, "images")
os.makedirs(images_output_dir, exist_ok=True)

model = TrinityInverseModel().to(PYTORCH_DEVICE)

old_state = torch.load(args.inference_checkpoint_path, map_location=PYTORCH_DEVICE)
new_state = {}
for k, v in old_state.items():
    new_k = k
    if k == 'det_vectors': new_k = 'dr_module.p'
    elif k == 'det_alpha_logits': new_k = 'dr_module.alpha_logits'
    elif k == 'se_beta': new_k = 'dr_module.beta'
    elif k == 'det_se_w': new_k = 'dr_module.w'
    elif k == 'det_se_b': new_k = 'dr_module.b'
    elif k.startswith('encoder.'): new_k = k.replace('encoder.', 'iid_module.encoder_block.net.')
    elif k.startswith('aspp.'): new_k = k.replace('aspp.', 'iid_module.aspp_block.')
    elif k.startswith('head_phys.'): new_k = k.replace('head_phys.', 'iid_module.head_surface.')
    elif k.startswith('head_r_low.'): new_k = k.replace('head_r_low.', 'iid_module.head_r_low.')
    elif k.startswith('head_r_high.'): new_k = k.replace('head_r_high.', 'iid_module.head_r_high.')
    new_state[new_k] = v

model.load_state_dict(new_state)
model.eval()

def infer_and_reconstruct(img_etd, img_t1, img_t2):
    stack = np.stack([img_etd, img_t1, img_t2], axis=0)
    inp_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(PYTORCH_DEVICE)

    with torch.inference_mode():
        f_img, c, nx, ny, nz, r_low, r_high, eta, g = model(inp_tensor)

    reconstructions = {}
    for i, det in enumerate(DETECTOR_NAMES):
        recon = inference_from_latents(
            detector_index=i,
            rh_factor=0.0,
            rl_factor=1.0,
            det_p_rot_rad=0.0,
            latent_c=c.squeeze().cpu().numpy(),
            latent_normals=torch.cat([nx, ny, nz], dim=1).squeeze(0).cpu().numpy(),
            latent_r_lows=r_low.squeeze(0).cpu().numpy(),
            latent_r_highs=r_high.squeeze(0).cpu().numpy(),
            det_alpha=1.0 / (1.0 + np.exp(-model.dr_module.alpha_logits.detach().cpu().numpy().flatten())),
            det_se_w=model.dr_module.w.detach().cpu().numpy().flatten(),
            det_se_b=model.dr_module.b.detach().cpu().numpy().flatten(),
            se_beta=model.dr_module.beta.detach().cpu().numpy().flatten()[0],
            det_p_vectors= model.dr_module.p.detach().cpu().numpy(),
            latent_gain=g.flatten().cpu().numpy()
        )
        reconstructions[det] = np.clip(recon, 0, 1)
    return reconstructions


# 2. EVALUATION PIPELINE
# ==========================================
sample_dirs = [d for d in glob.glob(os.path.join(args.pseudo_ground_truth_root_dir, "*")) if os.path.isdir(d)]
results = []

for sample_dir in sample_dirs:
    sample_id = os.path.basename(sample_dir)
    clean_dir = os.path.join(sample_dir, "clean")

    clean_imgs = {}
    for det in DETECTOR_NAMES:
        clean_imgs[det] = tifffile.imread(os.path.join(clean_dir, f"{det}.tiff")).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE

    clean_recons = infer_and_reconstruct(clean_imgs['ETD'], clean_imgs['T1'], clean_imgs['T2'])

    sample_out_dir = os.path.join(images_output_dir, sample_id)
    os.makedirs(sample_out_dir, exist_ok=True)

    for det in DETECTOR_NAMES:
        tifffile.imwrite(os.path.join(sample_out_dir, f"recon_clean_{det}.tiff"), (clean_recons[det] * TIFF_MAX_INTENSITY_VALUE).astype(np.uint16))
        p = psnr(clean_imgs[det], clean_recons[det], data_range=1.0)
        s = ssim(clean_imgs[det], clean_recons[det], data_range=1.0)
        results.append({'sample': sample_id, 'condition': 'clean', 'sigma': 0.0, 'realization': 0, 'detector': det, 'baseline_psnr': float('inf'), 'baseline_ssim': 1.0, 'recon_psnr': p, 'recon_ssim': s})

    noisy_dirs = [d for d in glob.glob(os.path.join(sample_dir, "noisy_sigma*")) if os.path.isdir(d)]
    if not noisy_dirs: continue

    unique_sigmas = sorted(list(set([float(os.path.basename(d).split('_')[1].replace('sigma', '')) for d in noisy_dirs])))
    medium_sigma = unique_sigmas[len(unique_sigmas) // 2]
    worst_sigma = unique_sigmas[-1]

    for noisy_dir in tqdm(noisy_dirs, desc=f"Evaluating noisy reconstructions for sample {sample_id}"):
        dirname = os.path.basename(noisy_dir)
        parts = dirname.split('_')
        sigma_val = float(parts[1].replace('sigma', ''))
        rel_val = int(parts[2].replace('rel', ''))

        noisy_imgs = {}
        for det in DETECTOR_NAMES:
            noisy_imgs[det] = tifffile.imread(os.path.join(noisy_dir, f"{det}.tiff")).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE

        noisy_recons = infer_and_reconstruct(noisy_imgs['ETD'], noisy_imgs['T1'], noisy_imgs['T2'])

        if rel_val == 0 and sigma_val in (medium_sigma, worst_sigma):
            label = "worst" if sigma_val == worst_sigma else "medium"
            for det in DETECTOR_NAMES:
                tifffile.imwrite(os.path.join(sample_out_dir, f"recon_{label}_{det}.tiff"), (noisy_recons[det] * TIFF_MAX_INTENSITY_VALUE).astype(np.uint16))

        for det in DETECTOR_NAMES:
            base_p = psnr(clean_imgs[det], noisy_imgs[det], data_range=1.0)
            base_s = ssim(clean_imgs[det], noisy_imgs[det], data_range=1.0)
            recon_p = psnr(clean_imgs[det], noisy_recons[det], data_range=1.0)
            recon_s = ssim(clean_imgs[det], noisy_recons[det], data_range=1.0)
            results.append({'sample': sample_id, 'condition': 'noisy', 'sigma': sigma_val, 'realization': rel_val, 'detector': det, 'baseline_psnr': base_p, 'baseline_ssim': base_s, 'recon_psnr': recon_p, 'recon_ssim': recon_s})

df = pd.DataFrame(results)
df.to_csv(os.path.join(args.output_dir, "evaluation_metrics.csv"), index=False)
