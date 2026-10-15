if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
"""
Ablation evaluation orchestrator: runs inference on the held-out fold's PGT
samples (clean + noisy) with the trained checkpoints, computes PSNR/SSIM,
and saves raw results to disk (no plotting).
"""
import os
import sys
import json
import glob
import time
import torch
import argparse
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

#
# add src/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from _1_constants import TIFF_MAX_INTENSITY_VALUE, PYTORCH_DEVICE
from _4_model_architecture import TrinityInverseModel, ProposedModelArchitecture
from tools.utils import inference_from_latents
from ablation_config import REGISTRATION_SHIFTS_CSV, eval_dir
from ablation_state import (
    init_state, load_state, print_progress,
    claim_next_eval, mark_eval_completed,
    PH_EVAL_PRETRAIN, resolve_state_path,
)
from ablation_data_prep import ensure_evaluation_data

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

DETECTOR_NAMES = ['ETD', 'T1', 'T2']


def _filter_pgt_samples(pgt_root: str, eval_specimens: list) -> list:
    """Return PGT sample directories belonging to the fold's specimens."""
    sample_dirs = [d for d in glob.glob(os.path.join(pgt_root, "*")) if os.path.isdir(d)]
    filtered = []
    for d in sample_dirs:
        name = os.path.basename(d)
        if any(spec in name for spec in eval_specimens):
            filtered.append(d)
    return filtered


def _infer_and_reconstruct(model, img_etd, img_t1, img_t2):
    """Run model inference and reconstruct per-detector images from latents.
    Returns (recons, latents, inference_time_s, peak_gpu_mem_mb)."""
    stack = np.stack([img_etd, img_t1, img_t2], axis=0)
    inp = torch.from_numpy(stack).unsqueeze(0).float().to(PYTORCH_DEVICE)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    with torch.inference_mode():
        _, c, nx, ny, nz, r_low, r_high, eta, g = model(inp)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_s = time.time() - t_start
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    recons = {}
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
            det_p_vectors=model.dr_module.p.detach().cpu().numpy(),
            latent_gain=g.flatten().cpu().numpy(),
        )
        recons[det] = np.clip(recon, 0, 1)
    return recons, (c, nx, ny, nz, r_low, r_high, eta, g), inference_time_s, peak_gpu_mem_mb


def run_evaluation(config: dict, eval_specimens: list,
                   checkpoint_path: str, pgt_root: str, out_dir: str) -> str:
    """Run evaluation for a single config × fold × phase. Returns path to metrics CSV."""
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    latents_dir = os.path.join(out_dir, "latents")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    latent_ch = config["latent_channels"]
    arch_config = config["arch_config"]
    r_high_skip = config["r_high_skip_connection"]

    model = ProposedModelArchitecture(
        latent_channels=latent_ch,
        r_high_skip_connection=r_high_skip,
        arch_config=arch_config,
    ).to(PYTORCH_DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=PYTORCH_DEVICE))
    model.eval()

    # save detector parameters and model profile
    total_params = sum(p.numel() for p in model.parameters())
    det_params = {
        "det_p_vectors": model.dr_module.p.detach().cpu().numpy().tolist(),
        "det_alpha_logits": model.dr_module.alpha_logits.view(1, 3, 1, 1).detach().cpu().numpy().tolist(),
        "se_beta": model.dr_module.beta.view(1, 1, 1, 1).detach().cpu().numpy().tolist(),
        "det_se_w": model.dr_module.w.view(1, 3, 1, 1).detach().cpu().numpy().tolist(),
        "det_se_b": model.dr_module.b.view(1, 3, 1, 1).detach().cpu().numpy().tolist(),
        "total_parameters": total_params,
    }
    with open(os.path.join(out_dir, "detector_params.json"), "w") as f:
        json.dump(det_params, f, indent=4)

    # filter PGT samples by fold
    sample_dirs = _filter_pgt_samples(pgt_root, eval_specimens)
    if not sample_dirs:
        print(f"[eval] No PGT samples found for specimens {eval_specimens}")
        # still write an empty CSV
        df = pd.DataFrame(columns=["sample", "condition", "sigma", "realization",
                                    "detector", "baseline_psnr", "baseline_ssim",
                                    "recon_psnr", "recon_ssim"])
        csv_path = os.path.join(out_dir, "evaluation_metrics.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    print(f"[eval] {len(sample_dirs)} PGT samples to evaluate")
    results = []
    perf_records = []  # inference_time_s and peak_gpu_mem_mb per sample

    for sample_dir in tqdm(sample_dirs, desc="Evaluating PGT samples"):
        sample_id = os.path.basename(sample_dir)
        clean_dir = os.path.join(sample_dir, "clean")

        clean_imgs = {}
        for det in DETECTOR_NAMES:
            clean_imgs[det] = tifffile.imread(os.path.join(clean_dir, f"{det}.tiff")).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE

        # clean inference
        clean_recons, latents, clean_time_s, clean_mem_mb = _infer_and_reconstruct(model, clean_imgs['ETD'], clean_imgs['T1'], clean_imgs['T2'])
        perf_records.append({
            'sample': sample_id, 'condition': 'clean', 'sigma': 0.0, 'realization': 0,
            'inference_time_s': clean_time_s, 'peak_gpu_mem_mb': clean_mem_mb,
        })

        # save latents for clean input
        c, nx, ny, nz, r_low, r_high, _, g = latents
        np.savez_compressed(
            os.path.join(latents_dir, f"{sample_id}_clean_latents.npz"),
            comp=c.squeeze(0).cpu().numpy(),
            normals=torch.cat([nx, ny, nz], dim=1).squeeze(0).cpu().numpy(),
            r_low=r_low.squeeze(0).cpu().numpy(),
            r_high=r_high.squeeze(0).cpu().numpy(),
            gain=g.squeeze(0).cpu().numpy(),
            f_img=clean_recons['ETD'],  # one reconstruction for reference
        )

        # save clean reconstructions
        sample_img_dir = os.path.join(images_dir, sample_id)
        os.makedirs(sample_img_dir, exist_ok=True)
        for det in DETECTOR_NAMES:
            tifffile.imwrite(os.path.join(sample_img_dir, f"recon_clean_{det}.tiff"),
                             (clean_recons[det] * TIFF_MAX_INTENSITY_VALUE).astype(np.uint16))
            p = psnr(clean_imgs[det], clean_recons[det], data_range=1.0)
            s = ssim(clean_imgs[det], clean_recons[det], data_range=1.0)
            results.append({
                'sample': sample_id, 'condition': 'clean', 'sigma': 0.0,
                'realization': 0, 'detector': det,
                'baseline_psnr': float('inf'), 'baseline_ssim': 1.0,
                'recon_psnr': p, 'recon_ssim': s,
            })

        # noisy inference
        noisy_dirs = [d for d in glob.glob(os.path.join(sample_dir, "noisy_sigma*")) if os.path.isdir(d)]
        if not noisy_dirs:
            continue

        unique_sigmas = sorted(set(float(os.path.basename(d).split('_')[1].replace('sigma', '')) for d in noisy_dirs))
        medium_sigma = unique_sigmas[len(unique_sigmas) // 2]
        worst_sigma = unique_sigmas[-1]

        for noisy_dir in noisy_dirs:
            dirname = os.path.basename(noisy_dir)
            parts = dirname.split('_')
            sigma_val = float(parts[1].replace('sigma', ''))
            rel_val = int(parts[2].replace('rel', ''))

            noisy_imgs = {}
            for det in DETECTOR_NAMES:
                noisy_imgs[det] = tifffile.imread(os.path.join(noisy_dir, f"{det}.tiff")).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE

            noisy_recons, _, noisy_time_s, noisy_mem_mb = _infer_and_reconstruct(model, noisy_imgs['ETD'], noisy_imgs['T1'], noisy_imgs['T2'])
            perf_records.append({
                'sample': sample_id, 'condition': 'noisy', 'sigma': sigma_val,
                'realization': rel_val,
                'inference_time_s': noisy_time_s, 'peak_gpu_mem_mb': noisy_mem_mb,
            })

            # save medium and worst sigma reconstructions (realization 0 only)
            if rel_val == 0 and sigma_val in (medium_sigma, worst_sigma):
                label = "worst" if sigma_val == worst_sigma else "medium"
                for det in DETECTOR_NAMES:
                    tifffile.imwrite(os.path.join(sample_img_dir, f"recon_{label}_{det}.tiff"),
                                     (noisy_recons[det] * TIFF_MAX_INTENSITY_VALUE).astype(np.uint16))

            for det in DETECTOR_NAMES:
                base_p = psnr(clean_imgs[det], noisy_imgs[det], data_range=1.0)
                base_s = ssim(clean_imgs[det], noisy_imgs[det], data_range=1.0)
                recon_p = psnr(clean_imgs[det], noisy_recons[det], data_range=1.0)
                recon_s = ssim(clean_imgs[det], noisy_recons[det], data_range=1.0)
                results.append({
                    'sample': sample_id, 'condition': 'noisy', 'sigma': sigma_val,
                    'realization': rel_val, 'detector': det,
                    'baseline_psnr': base_p, 'baseline_ssim': base_s,
                    'recon_psnr': recon_p, 'recon_ssim': recon_s,
                })

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[eval] Saved {len(df)} metric rows to {csv_path}")

    # save performance (inference time + GPU memory) records
    perf_df = pd.DataFrame(perf_records)
    perf_csv_path = os.path.join(out_dir, "inference_performance.csv")
    perf_df.to_csv(perf_csv_path, index=False)
    print(f"[eval] Saved {len(perf_df)} performance rows to {perf_csv_path}")

    return csv_path


# __ main orchestrator _____________________________________________________

def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation orchestrator")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Root of the experimental dataset (trinity_sem_dataset/)")
    parser.add_argument('--time_limit_hours', type=float, default=11.5,
                        help="Soft time limit in hours")
    args = parser.parse_args()

    print(f"pytorch device: {PYTORCH_DEVICE}")
    start_time = time.time()
    time_limit_s = args.time_limit_hours * 3600

    # ensure evaluation data artefacts exist
    print("[orchestrator] Ensuring evaluation data artefacts…")
    pgt_root = ensure_evaluation_data(args.data_root, REGISTRATION_SHIFTS_CSV)

    # load / init state
    state = init_state()
    print_progress(state)

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit_s:
            print("[orchestrator] Time limit reached. Exiting gracefully.")
            break

        task = claim_next_eval()
        if task is None:
            print("[orchestrator] All evaluation tasks are complete!")
            state = load_state()
            print_progress(state)
            break

        rid = task["rid"]
        phase = task["phase"]
        cfg = task["config"]
        fk = task["fold_key"]
        eval_spec = task["eval_specimens"]

        # determine checkpoint
        if phase == PH_EVAL_PRETRAIN:
            checkpoint = task["pretrain"]["checkpoint"]
        else:
            checkpoint = task["finetune"]["checkpoint"]

        checkpoint = resolve_state_path(checkpoint)

        if checkpoint is None or not os.path.exists(checkpoint):
            print(f"[orchestrator] WARNING: checkpoint missing for {cfg['id']} | {fk} | {phase}, skipping")
            continue

        out_dir = eval_dir(cfg["id"], fk, phase)

        print(f"\n{'='*60}")
        print(f"[orchestrator] EVAL: {cfg['id']} | {fk} | {phase}")
        print(f"{'='*60}")

        try:
            csv_path = run_evaluation(cfg, eval_spec, checkpoint, pgt_root, out_dir)
            state = load_state()
            mark_eval_completed(state, rid, phase, csv_path)
            print(f"[orchestrator] Evaluation completed: {cfg['id']} | {fk} | {phase}")
        except Exception as e:
            print(f"[orchestrator] Evaluation FAILED: {e}")
            raise

    print_progress(state)
    print("[orchestrator] Done.")


if __name__ == "__main__":
    main()
