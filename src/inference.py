if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import json
import torch
import tifffile
import argparse
import numpy as np
from tqdm import tqdm
from _1_constants import TIFF_MAX_INTENSITY_VALUE, PYTORCH_DEVICE, DEFAULT_EVALUATION_SPECIMEN_NAMES
from _2_data_loading_helpers import parse_data_structure
from _4_model_architecture import TrinityInverseModel

DEFAULT_INFERENCE_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "ablation_study", "ablation_results", "runs", "enc4L_ch96_aspp", "fold2", "finetune", "best_model_finetune.pth"
)

# 0. CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--output_dir', type=str, default=".")
parser.add_argument('--inference_checkpoint_path', type=str, default=DEFAULT_INFERENCE_CHECKPOINT,
    help="Path to model checkpoint (default: enc4L_ch96_aspp fold2 best_model_finetune.pth)"
)
parser.add_argument('--evaluation_specimen_names', type=str, default=DEFAULT_EVALUATION_SPECIMEN_NAMES["fold 2"],
    help="Comma-delimited list of evaluation specimen directory names. Inference will only be run on these specimens."
)
args = parser.parse_args()
args.evaluation_specimen_names = [d.strip() for d in args.evaluation_specimen_names.split(',')]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; torch.backends.cudnn.benchmark = True
print(f"pytorch device: {PYTORCH_DEVICE}")


# 1. SETUP
# ==========================================
print("Running setup")

all_triplets = parse_data_structure(args.input_data_root)
if len(all_triplets) == 0: raise ValueError("no triplets found")

model = TrinityInverseModel().to(PYTORCH_DEVICE)


# 2. INFERENCE
# ==========================================
print("Running triplet inference")
model.load_state_dict(torch.load(args.inference_checkpoint_path, map_location=PYTORCH_DEVICE))
model.eval()

def triplet_infer_and_save(triplet_info):
    img_etd = tifffile.imread(triplet_info['ETD']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
    img_t1  = tifffile.imread(triplet_info['T1']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
    img_t2  = tifffile.imread(triplet_info['T2']).astype(np.float32) / TIFF_MAX_INTENSITY_VALUE
    stack = np.stack([img_etd, img_t1, img_t2], axis=0)
    inp_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(PYTORCH_DEVICE)

    triplet_dir = os.path.join(args.output_dir, f"{triplet_info['specimen_dir'].split(os.sep)[-1]}_{triplet_info['ID']}")
    os.makedirs(triplet_dir, exist_ok=True)
    
    with torch.inference_mode():
        f_img, c, nx, ny, nz, r_low, r_high, eta, g = model(inp_tensor)

        detector_params_dict = {
            "det_p_vectors": model.dr_module.p.cpu().numpy().tolist(),
            "det_alpha_logits": model.dr_module.alpha_logits.view(1, 3, 1, 1).cpu().numpy().tolist(),
            "se_beta": model.dr_module.beta.view(1, 1, 1, 1).cpu().numpy().tolist(),
            "det_se_w": model.dr_module.w.view(1, 3, 1, 1).cpu().numpy().tolist(),
            "det_se_b": model.dr_module.b.view(1, 3, 1, 1).cpu().numpy().tolist()
        }
        with open(os.path.join(triplet_dir, "params.json"), "w") as f:
            json.dump(detector_params_dict, f, indent=4)

        np.savez_compressed(
            os.path.join(triplet_dir, "latents.npz"),
            comp=c.squeeze(0).cpu().numpy(),
            normals=torch.cat([nx, ny, nz], dim=1).squeeze(0).cpu().numpy(),
            r_low=r_low.squeeze(0).cpu().numpy(),
            r_high=r_high.squeeze(0).cpu().numpy(),
            gain=g.squeeze(0).cpu().numpy()
        )

for triplet_info in tqdm([triplet_info for triplet_info in all_triplets if any(eval_dir in triplet_info['specimen_dir'] for eval_dir in args.evaluation_specimen_names)], desc="Testing triplet inference"):
    triplet_infer_and_save(triplet_info)
