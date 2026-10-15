#!/usr/bin/env python
"""
Trinity-Inverse: Run Cross-Validation Inference for Ablation Model Architectures.

This script runs inference using trained ablation weights, automatically mapping
each evaluation specimen to its corresponding cross-validation fold model.
"""

import os
import sys
import json
import argparse
import tifffile
import torch
import numpy as np
from tqdm import tqdm

# Add src to sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from _1_constants import (
    TIFF_MAX_INTENSITY_VALUE,
    PYTORCH_DEVICE,
    DEFAULT_EVALUATION_SPECIMEN_NAMES,
)
from _2_data_loading_helpers import parse_data_structure
from _4_model_architecture import TrinityInverseModel
from ablation_study.ablation_config import generate_all_configs

ABLATION_CONFIGS = generate_all_configs()
ABLATION_CONFIGS_COUNT = len(ABLATION_CONFIGS)


def get_config_map():
    return {c["id"]: c for c in ABLATION_CONFIGS}


def list_available_models():
    configs = ABLATION_CONFIGS
    groups = {
        "Encoder + ASPP": [],
        "UNet": [],
        "Encoder + UNet": [],
    }
    for c in configs:
        arch = c["arch_type"]
        if arch == "encoder_aspp":
            groups["Encoder + ASPP"].append(c["id"])
        elif arch == "unet":
            groups["UNet"].append(c["id"])
        elif arch == "encoder_unet":
            groups["Encoder + UNet"].append(c["id"])

    print(f"\nAvailable Ablation Model Codenames ({ABLATION_CONFIGS_COUNT} variants):")
    print("=" * 60)
    for group_name, model_ids in groups.items():
        print(f"\n{group_name}:")
        for mid in model_ids:
            print(f"  {mid}")
    print("=" * 60 + "\n")


def resolve_checkpoint_path(runs_dir, model_id, fold_num, phase):
    fold_dir = os.path.join(runs_dir, model_id, f"fold{fold_num}")
    if phase == "finetune":
        ft_ckpt = os.path.join(fold_dir, "finetune", "best_model_finetune.pth")
        if os.path.exists(ft_ckpt):
            return ft_ckpt, "finetune"
        # Fallback to pretrain stage 3
        pt_ckpt = os.path.join(fold_dir, "pretrain", "best_model_stage3.pth")
        if os.path.exists(pt_ckpt):
            print(f"  [Notice] Fine-tune checkpoint not found for fold {fold_num}, falling back to pretrain stage 3.")
            return pt_ckpt, "pretrain"
        return ft_ckpt, "missing"
    else:
        pt_ckpt = os.path.join(fold_dir, "pretrain", "best_model_stage3.pth")
        if os.path.exists(pt_ckpt):
            return pt_ckpt, "pretrain"
        return pt_ckpt, "missing"


def infer_single_triplet(model, triplet_info, output_dir, device):
    img_etd = tifffile.imread(triplet_info["ETD"]).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
    img_t1 = tifffile.imread(triplet_info["T1"]).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
    img_t2 = tifffile.imread(triplet_info["T2"]).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
    stack = np.stack([img_etd, img_t1, img_t2], axis=0)
    inp_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device)

    specimen_folder_name = os.path.basename(triplet_info["specimen_dir"])
    triplet_dir = os.path.join(output_dir, f"{specimen_folder_name}_{triplet_info['ID']}")
    os.makedirs(triplet_dir, exist_ok=True)

    with torch.inference_mode():
        f_img, c, nx, ny, nz, r_low, r_high, eta, g = model(inp_tensor)

        detector_params_dict = {
            "det_p_vectors": model.dr_module.p.cpu().numpy().tolist(),
            "det_alpha_logits": model.dr_module.alpha_logits.view(1, 3, 1, 1).cpu().numpy().tolist(),
            "se_beta": model.dr_module.beta.view(1, 1, 1, 1).cpu().numpy().tolist(),
            "det_se_w": model.dr_module.w.view(1, 3, 1, 1).cpu().numpy().tolist(),
            "det_se_b": model.dr_module.b.view(1, 3, 1, 1).cpu().numpy().tolist(),
        }
        with open(os.path.join(triplet_dir, "params.json"), "w") as f:
            json.dump(detector_params_dict, f, indent=4)

        np.savez_compressed(
            os.path.join(triplet_dir, "latents.npz"),
            comp=c.squeeze(0).cpu().numpy(),
            normals=torch.cat([nx, ny, nz], dim=1).squeeze(0).cpu().numpy(),
            r_low=r_low.squeeze(0).cpu().numpy(),
            r_high=r_high.squeeze(0).cpu().numpy(),
            gain=g.squeeze(0).cpu().numpy(),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-validation inference for a Trinity-Inverse ablation architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_codename",
        nargs="?",
        default="enc4L_ch96_aspp",
        type=str,
        help=f"Ablation model codename (default: enc4L_ch96_aspp). Use --list to see all {ABLATION_CONFIGS_COUNT} variants.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=f"List all {ABLATION_CONFIGS_COUNT} available ablation model codenames and exit.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["finetune", "pretrain"],
        default="finetune",
        help="Model training phase to use (default: finetune).",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Specific fold to run (1-7), or 'all' to run all 7 folds (default: all).",
    )
    parser.add_argument(
        "--input_data_root",
        type=str,
        default=os.path.join(SRC_DIR, "..", "trinity_sem_dataset"),
        help="Root path to the Trinity SEM Dataset (default: trinity_sem_dataset).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory for saving inferred latents (default: inferred_latents/<model_codename>_<phase>).",
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=os.path.join(SRC_DIR, "ablation_study", "ablation_results", "runs"),
        help="Path to the ablation runs directory containing trained checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch compute device ('cuda', 'mps', or 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--include_sial1",
        action="store_true",
        help="Also run inference on the SiAl1 specimen (using fold 1 model). Default: False.",
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    config_map = get_config_map()
    if args.model_codename not in config_map:
        print(f"\n[Error] Unknown model codename: '{args.model_codename}'.")
        list_available_models()
        sys.exit(1)

    config = config_map[args.model_codename]
    model_id = config["id"]

    device = torch.device(args.device) if args.device else PYTORCH_DEVICE
    print(f"\n=======================================================")
    print(f" Trinity-Inverse Cross-Validation Inference")
    print(f"=======================================================")
    print(f"  Model Codename : {model_id} ({config['arch_type']})")
    print(f"  Phase          : {args.phase}")
    print(f"  Compute Device : {device}")
    print(f"  Dataset Root   : {os.path.abspath(args.input_data_root)}")

    out_dir = args.output_dir or os.path.join("inferred_latents", f"{model_id}_{args.phase}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output Root    : {os.path.abspath(out_dir)}")
    print(f"=======================================================\n")

    # Parse dataset
    all_triplets = parse_data_structure(args.input_data_root)
    if not all_triplets:
        print(f"[Error] No image triplets found in {args.input_data_root}!")
        sys.exit(1)
    print(f"Found {len(all_triplets)} total triplets across {len(set(t['specimen_dir'] for t in all_triplets))} specimens.\n")

    # Determine folds to run
    if args.fold.lower() == "all":
        fold_nums = list(range(1, 8))
    else:
        try:
            f_val = int(args.fold.replace("fold", "").strip())
            if f_val < 1 or f_val > 7:
                raise ValueError()
            fold_nums = [f_val]
        except ValueError:
            print(f"[Error] Invalid fold specification: '{args.fold}'. Expected 1-7 or 'all'.")
            sys.exit(1)

    total_inferred = 0

    for f_num in fold_nums:
        fold_key = f"fold {f_num}"
        eval_specimens = [s.strip() for s in DEFAULT_EVALUATION_SPECIMEN_NAMES[fold_key].split(",")]

        ckpt_path, status = resolve_checkpoint_path(args.runs_dir, model_id, f_num, args.phase)
        if status == "missing":
            print(f"  [Warning] [Fold {f_num}] Checkpoint not found at: {ckpt_path}")
            print(f"    Skipping evaluation for specimens: {eval_specimens}")
            continue

        print(f"── Fold {f_num} [Evaluation specimens: {', '.join(eval_specimens)}] ──")
        print(f"   Loading weights: {os.path.relpath(ckpt_path, os.getcwd())}")

        # Instantiate model for this architecture configuration
        model = TrinityInverseModel(
            latent_channels=config["latent_channels"],
            r_high_skip_connection=config["r_high_skip_connection"],
            arch_config=config["arch_config"],
        ).to(device)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Find matching triplets
        matching_triplets = [
            t for t in all_triplets
            if any(spec in t["specimen_dir"] for spec in eval_specimens)
        ]

        if not matching_triplets:
            print(f"   No triplets found for specimens: {eval_specimens}")
            continue

        for triplet_info in tqdm(matching_triplets, desc=f"   Inferring fold {f_num}"):
            infer_single_triplet(model, triplet_info, out_dir, device)
            total_inferred += 1

        print()

    # Optional: infer SiAl1
    if args.include_sial1:
        sial1_triplets = [t for t in all_triplets if "SiAl1" in t["specimen_dir"]]
        if sial1_triplets:
            ckpt_path, _ = resolve_checkpoint_path(args.runs_dir, model_id, 1, args.phase)
            if os.path.exists(ckpt_path):
                print(f"── Auxiliary: Inferring SiAl1 (using Fold 1 weights) ──")
                model = TrinityInverseModel(
                    latent_channels=config["latent_channels"],
                    r_high_skip_connection=config["r_high_skip_connection"],
                    arch_config=config["arch_config"],
                ).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()

                for triplet_info in tqdm(sial1_triplets, desc="   Inferring SiAl1"):
                    infer_single_triplet(model, triplet_info, out_dir, device)
                    total_inferred += 1
                print()

    print(f"=======================================================")
    print(f" Inference Complete!")
    print(f"   Total Triplets Inferred : {total_inferred}")
    print(f"   Output Directory        : {os.path.abspath(out_dir)}")
    print(f"\n To inspect the inferred components in the GUI Viewer:")
    print(f"   python src/gui/viewer.py --latents_root_dir {out_dir}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
