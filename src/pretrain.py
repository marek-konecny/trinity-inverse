if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import re
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from _1_constants import PYTORCH_DEVICE, PATCHES_PER_BATCH, PATCH_SIZE_PX, BATCHES_PER_EPOCH, START_LEARNING_RATE, TOTAL_EPOCHS, DEFAULT_EVALUATION_SPECIMEN_NAMES, LOSS_COMPONENT_WEIGHTS_PRETRAINING
from _2_data_loading_helpers import load_spherical_particles_data, parse_data_structure
from _3_dataset_classes import ScanTripletPatchSampler
from _4_model_architecture import TrinityInverseModel
from _5_loss_helpers import dirichlet_energy, sphere_normal_loss, uniform_composition_loss, inter_channel_correlation, autocorrelation


# 0. CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--spherical_particles_data_csv_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default=".")
parser.add_argument('--evaluation_specimen_names', type=str, default=DEFAULT_EVALUATION_SPECIMEN_NAMES["fold 2"],
    help="Comma-delimited list of evaluation specimen directory names. These will be excluded from the training split."
)
parser.add_argument('--continue_from_checkpoint_path', type=str, default=None)
args = parser.parse_args()
args.evaluation_specimen_names = [d.strip() for d in args.evaluation_specimen_names.split(',')]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; torch.backends.cudnn.benchmark = True
print(f"pytorch device: {PYTORCH_DEVICE}")


# 1. SETUP
# ==========================================
print("Running setup")

spherical_particles_data = load_spherical_particles_data(args.spherical_particles_data_csv_path)
all_triplets = parse_data_structure(args.input_data_root, spherical_particles_data, None)
if len(all_triplets) == 0: raise ValueError("no triplets found")

model = TrinityInverseModel().to(PYTORCH_DEVICE)


# 2. PRE-TRAINING
# ==========================================
print("Running pre-training")

train_triplets = [triplet_info for triplet_info in all_triplets if not any(eval_dir in triplet_info['specimen_dir'] for eval_dir in args.evaluation_specimen_names)]
dataset = ScanTripletPatchSampler(train_triplets, PATCH_SIZE_PX, BATCHES_PER_EPOCH * PATCHES_PER_BATCH)
loader = DataLoader(dataset, PATCHES_PER_BATCH, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

optimizer = optim.AdamW(model.parameters(), lr=START_LEARNING_RATE, fused=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300)
scaler = torch.amp.GradScaler('cuda')

start_epoch = 0
if args.continue_from_checkpoint_path:
    print(f"Continuing from checkpoint: {args.continue_from_checkpoint_path}")
    model.load_state_dict(torch.load(args.continue_from_checkpoint_path, map_location=PYTORCH_DEVICE))
    match = re.search(r'epoch_(\d+)', args.continue_from_checkpoint_path)
    if match: start_epoch = int(match.group(1))

# logging setup
csv_filename = os.path.join(args.output_dir, "training_log.csv")
# set up CSV headers
det_names = ['ETD', 'T1', 'T2']
param_headers = []
for d in det_names:
    param_headers.extend([f"Px_{d}", f"Py_{d}", f"Pmag_{d}", f"Alpha_{d}"])
headers = ["Epoch", "Stage", "LR", "Total", "Rec", "TV_Geo", "TV_C", "Sphere", "Comp_Uni", "Eta_Inter_Det_Corr", "Eta_Whiteness", "Rl_Reg"] + param_headers

file_mode = 'a' if start_epoch > 0 and os.path.exists(csv_filename) else 'w'
with open(csv_filename, file_mode) as f:
    if file_mode == 'w': csv.writer(f).writerow(headers)

best_stage_loss = float('inf')
stage_prev_epoch = 0

# main training loop
for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()

    # stage-per-epoch scheduling
    if epoch < 500:
        stage_now = 1
    elif epoch < 1500:
        stage_now = 2
    else:
        stage_now = 3

    if stage_now != stage_prev_epoch:
        print(f"\nTransitioning to Stage {stage_now}")
        best_stage_loss = float('inf')
        stage_prev_epoch = stage_now
        for param_group in optimizer.param_groups:
            param_group['lr'] = START_LEARNING_RATE
        optimizer.state.clear()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500)
    current_lr = optimizer.param_groups[0]['lr']

    # detector params unfreezing
    if epoch < 10:
        model.dr_module.p.requires_grad = False
        model.dr_module.alpha_logits.requires_grad = False
        model.dr_module.beta.requires_grad = False
        model.dr_module.w.requires_grad = False
        model.dr_module.b.requires_grad = False
    else:
        model.dr_module.p.requires_grad = True
        model.dr_module.alpha_logits.requires_grad = True
        model.dr_module.beta.requires_grad = True
        model.dr_module.w.requires_grad = True
        model.dr_module.b.requires_grad = True

    acc_stats = {k: 0.0 for k in ["Total", "Rec", "TV_Geo", "TV_C", "Sphere", "Comp_Uni", "Eta_Inter_Det_Corr", "Eta_Whiteness", "Rl_Reg"]}
    sphere_loss_count = 0
    comp_uni_loss_count = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [S{stage_now}]")
    for input_stack, normals, mask in loop:
        input_stack = input_stack.to(PYTORCH_DEVICE, non_blocking=True)
        normals = normals.to(PYTORCH_DEVICE, non_blocking=True)
        mask = mask.to(PYTORCH_DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            f_img, c, nx, ny, nz, r_low, r_high, eta, g = model(input_stack)

            loss_rec = F.l1_loss(f_img, input_stack)
            loss_tv_geo = dirichlet_energy(nx) + dirichlet_energy(ny)
            loss_tv_c = dirichlet_energy(c)
            loss_sphere = sphere_normal_loss(nx, ny, nz, normals, mask) if LOSS_COMPONENT_WEIGHTS_PRETRAINING["sphere"][stage_now] > 0 else None
            loss_c_uni = uniform_composition_loss(c, mask) if LOSS_COMPONENT_WEIGHTS_PRETRAINING["comp_uni"][stage_now] > 0 else None
            loss_eta_inter_det_corr = inter_channel_correlation(eta)
            loss_eta_whiteness = autocorrelation(eta)
            loss_r_low_reg = torch.abs(r_low).mean()
            loss_t1_p_mag = (model.dr_module.p[1, 0]**2 + model.dr_module.p[1, 1]**2)
            loss_t1_alpha = torch.sigmoid(model.dr_module.alpha_logits.view(3)[1])
            loss_t2_alpha = 1.0 - torch.sigmoid(model.dr_module.alpha_logits.view(3)[2])

            loss = (
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["rec"][stage_now]                * loss_rec) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["tv_geo"][stage_now]             * loss_tv_geo) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["tv_c"][stage_now]               * loss_tv_c) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["eta_inter_det_corr"][stage_now] * loss_eta_inter_det_corr) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["white"][stage_now]              * loss_eta_whiteness) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["r_low_reg"][stage_now]          * loss_r_low_reg) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["t1_p_mag"][stage_now]           * loss_t1_p_mag) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["t1_alpha"][stage_now]           * loss_t1_alpha) +
                (LOSS_COMPONENT_WEIGHTS_PRETRAINING["t2_alpha"][stage_now]           * loss_t2_alpha) +
                # minimize overall surface tilt
                (torch.abs(torch.mean(nx, dim=(0,2,3))) + torch.abs(torch.mean(ny, dim=(0,2,3)))).mean()
            )

            if loss_sphere is not None:
                loss += (LOSS_COMPONENT_WEIGHTS_PRETRAINING["sphere"][stage_now] * loss_sphere)
                acc_stats["Sphere"] += loss_sphere.item()
                sphere_loss_count += 1

            if loss_c_uni is not None:
                loss += (LOSS_COMPONENT_WEIGHTS_PRETRAINING["comp_uni"][stage_now] * loss_c_uni)
                acc_stats["Comp_Uni"] += loss_c_uni.item()
                comp_uni_loss_count += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc_stats["Total"] += loss.item()
        acc_stats["Rec"] += loss_rec.item()
        acc_stats["TV_Geo"] += loss_tv_geo.item()
        acc_stats["TV_C"] += loss_tv_c.item()
        acc_stats["Eta_Inter_Det_Corr"] += loss_eta_inter_det_corr.item()
        acc_stats["Eta_Whiteness"] += loss_eta_whiteness.item()
        acc_stats["Rl_Reg"] += loss_r_low_reg.item()

        loop.set_postfix(l_tot=loss.item(), l_rec=loss_rec.item(), l_noise=loss_eta_inter_det_corr.item(), l_rl_reg=loss_r_low_reg.item(), l_sphere=loss_sphere.item() if loss_sphere is not None else None, l_c_uni=loss_c_uni.item() if loss_c_uni is not None else None, lr=current_lr)

    loss_total_avg = acc_stats["Total"] / len(loader)
    scheduler.step(loss_total_avg)

    # logging
    with torch.no_grad():
        vecs = model.dr_module.p.detach().cpu().numpy()
        mags = np.sqrt(vecs[:,0]**2 + vecs[:,1]**2)
        mix_p = torch.sigmoid(model.dr_module.alpha_logits).view(3).detach().cpu().numpy()

        vec_log = []
        for i in range(3):
            vec_log.extend([vecs[i,0], vecs[i,1], mags[i], mix_p[i]])

        param_vals = vec_log

    for k in ["Total", "Rec", "TV_Geo", "TV_C", "Eta_Inter_Det_Corr", "Eta_Whiteness", "Rl_Reg"]: acc_stats[k] /= len(loader)
    if sphere_loss_count > 0: acc_stats["Sphere"] /= sphere_loss_count
    if comp_uni_loss_count > 0: acc_stats["Comp_Uni"] /= comp_uni_loss_count

    csv_row = [epoch+1, stage_now, current_lr] + [acc_stats[k] for k in headers[3:12]] + param_vals
    with open(csv_filename, 'a') as f:
        csv.writer(f).writerow(csv_row)

    # checkpoint saving
    if loss_total_avg < best_stage_loss and not np.isnan(loss_total_avg) and not sphere_loss_count == 0 and not comp_uni_loss_count == 0:
        best_stage_loss = loss_total_avg
        save_name = f"best_model_stage{stage_now}.pth"
        torch.save(model.state_dict(), os.path.join(args.output_dir, save_name))
        print(f"New best (S{stage_now}): {epoch+1} (loss: {best_stage_loss:.5f}); Saved to {save_name}")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "latest_model.pth"))
    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
