if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import os
import sys
from typing import List, Dict, Any

#
# add src/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from _1_constants import DEFAULT_EVALUATION_SPECIMEN_NAMES

# __ directory layout (results live under ablation_study/) ________________
ABLATION_ROOT = os.path.join(os.path.dirname(__file__), "ablation_results")
DATA_CACHE_DIR = os.path.join(ABLATION_ROOT, "data_cache")
PGT_CACHE_DIR = os.path.join(ABLATION_ROOT, "pgt_cache")
RUNS_DIR = os.path.join(ABLATION_ROOT, "runs")
STATE_FILE = os.path.join(ABLATION_ROOT, "ablation_state.json")

SPHERICAL_PARTICLES_CSV = os.path.join(DATA_CACHE_DIR, "spherical_particles.csv")
REGISTRATION_SHIFTS_CSV = os.path.join(DATA_CACHE_DIR, "registration_shifts.csv")
PGT_DIR = os.path.join(PGT_CACHE_DIR, "pseudo_ground_truth")
NOISE_COEFFS_JSON = os.path.join(PGT_DIR, "noise_model_coefficients.json")

# specimen used for spherical-particle detection (never in evaluation folds)
DETECTION_SPECIMEN = "SiAl1"

# __ Ablation grid ________________________________________________________
ENCODER_LAYERS = [1, 2, 4, 8]
LATENT_CHANNELS = [8, 16, 48, 96]
ASPP_DILATIONS = [6, 12, 18]

FOLDS = DEFAULT_EVALUATION_SPECIMEN_NAMES

R_HIGH_SKIP_CONNECTION = True


def _config_id(arch_type: str, latent_channels: int, encoder_layers: int = None) -> str:
    """Short, filesystem-safe identifier for a configuration."""
    if arch_type == "encoder_aspp":
        return f"enc{encoder_layers}L_ch{latent_channels}_aspp"
    elif arch_type == "unet":
        return f"unet_ch{latent_channels}"
    elif arch_type == "encoder_unet":
        return f"enc{encoder_layers}L_ch{latent_channels}_unet"
    raise ValueError(arch_type)


def generate_all_configs() -> List[Dict[str, Any]]:
    """Return a list of every ablation configuration dict."""
    configs: List[Dict[str, Any]] = []

    # 1. Encoder + ASPP  (layers × channels)
    for layers in ENCODER_LAYERS:
        for ch in LATENT_CHANNELS:
            cid = _config_id("encoder_aspp", ch, layers)
            configs.append({
                "id": cid,
                "arch_type": "encoder_aspp",
                "latent_channels": ch,
                "encoder_layers": layers,
                "aspp_dilations": ASPP_DILATIONS,
                "r_high_skip_connection": R_HIGH_SKIP_CONNECTION,
                "arch_config": {"type": "encoder_aspp", "encoder_layers": layers, "aspp_dilations": ASPP_DILATIONS},
            })

    # 2. UNet only  (channels)
    for ch in LATENT_CHANNELS:
        cid = _config_id("unet", ch)
        configs.append({
            "id": cid,
            "arch_type": "unet",
            "latent_channels": ch,
            "encoder_layers": None,
            "aspp_dilations": None,
            "r_high_skip_connection": R_HIGH_SKIP_CONNECTION,
            "arch_config": {"type": "unet"},
        })

    # 3. Encoder + UNet  (layers × channels, same channels for both)
    for layers in ENCODER_LAYERS:
        for ch in LATENT_CHANNELS:
            cid = _config_id("encoder_unet", ch, layers)
            configs.append({
                "id": cid,
                "arch_type": "encoder_unet",
                "latent_channels": ch,
                "encoder_layers": layers,
                "aspp_dilations": None,
                "r_high_skip_connection": R_HIGH_SKIP_CONNECTION,
                "arch_config": {"type": "encoder_unet", "encoder_layers": layers},
            })

    return configs


def get_fold_specimens(fold_key: str) -> List[str]:
    """Return the comma-delimited specimen names for a fold key."""
    return [s.strip() for s in FOLDS[fold_key].split(",")]


def get_all_fold_keys() -> List[str]:
    return sorted(FOLDS.keys(), key=lambda k: int(k.split()[1]))


def run_dir(config_id: str, fold_key: str) -> str:
    """Directory for a single config × fold run."""
    fold_num = fold_key.split()[1]
    return os.path.join(RUNS_DIR, config_id, f"fold{fold_num}")


def pretrain_dir(config_id: str, fold_key: str) -> str:
    return os.path.join(run_dir(config_id, fold_key), "pretrain")


def finetune_dir(config_id: str, fold_key: str) -> str:
    return os.path.join(run_dir(config_id, fold_key), "finetune")


def eval_dir(config_id: str, fold_key: str, phase: str) -> str:
    """phase = 'eval_pretrain' or 'eval_finetune'"""
    return os.path.join(run_dir(config_id, fold_key), phase)


def pretrain_checkpoint(config_id: str, fold_key: str) -> str:
    return os.path.join(pretrain_dir(config_id, fold_key), "best_model_stage3.pth")


def finetune_checkpoint(config_id: str, fold_key: str) -> str:
    return os.path.join(finetune_dir(config_id, fold_key), "best_model_finetune.pth")
