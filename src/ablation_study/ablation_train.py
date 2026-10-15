if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
"""
Ablation training orchestrator: claims pending (pretrain → finetune) tasks
from the shared state file, trains them, and saves checkpoints.  Resumes
from the latest checkpoint on restart; exits on the soft time limit so the
next submission continues where this one left off.
"""
import os
import sys
import re
import csv
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

#
# add src/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from _1_constants import (
    PYTORCH_DEVICE, PATCHES_PER_BATCH, PATCH_SIZE_PX, START_LEARNING_RATE,
    TOTAL_EPOCHS, BATCHES_PER_EPOCH, LOSS_COMPONENT_WEIGHTS_PRETRAINING,
    LOSS_COMPONENT_WEIGHTS_FINETUNING,
)
from _2_data_loading_helpers import (
    load_spherical_particles_data, load_intra_sample_registration_data,
    parse_data_structure,
)
from _3_dataset_classes import ScanTripletPatchSampler, ScanTripletWithFinetuningDataPatchSampler
from _4_model_architecture import TrinityInverseModel, ProposedModelArchitecture
from _5_loss_helpers import (
    dirichlet_energy, sphere_normal_loss, uniform_composition_loss,
    inter_channel_correlation, autocorrelation,
)
from ablation_config import (
    pretrain_dir, finetune_dir, pretrain_checkpoint,
)
from ablation_state import (
    init_state, load_state, print_progress,
    claim_next_pretrain, claim_next_finetune,
    mark_pretrain_completed, mark_finetune_completed,
    resolve_state_path,
)
from ablation_data_prep import ensure_training_data

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True


# __ pretrain ______________________________________________________________

def run_pretrain(config: dict, eval_specimens: list,
                 data_root: str, particles_csv: str, out_dir: str,
                 time_limit_s: float, start_time: float) -> str:
    """Pretrain a single config × fold.  Returns the best checkpoint path."""
    os.makedirs(out_dir, exist_ok=True)
    latent_ch = config["latent_channels"]
    arch_config = config["arch_config"]
    r_high_skip = config["r_high_skip_connection"]

    model = ProposedModelArchitecture(
        latent_channels=latent_ch,
        r_high_skip_connection=r_high_skip,
        arch_config=arch_config,
    ).to(PYTORCH_DEVICE)

    # data
    spherical_particles_data = load_spherical_particles_data(particles_csv)
    all_triplets = parse_data_structure(data_root, spherical_particles_data, None)
    train_triplets = [t for t in all_triplets
                      if not any(e in t['specimen_dir'] for e in eval_specimens)]
    if not train_triplets:
        raise ValueError("No training triplets found")

    epoch_len = BATCHES_PER_EPOCH * PATCHES_PER_BATCH
    dataset = ScanTripletPatchSampler(train_triplets, PATCH_SIZE_PX, epoch_len)
    loader = DataLoader(dataset, PATCHES_PER_BATCH, shuffle=True,
                        num_workers=8, persistent_workers=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=START_LEARNING_RATE, fused=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300)
    scaler = torch.amp.GradScaler('cuda')

    # resume from latest checkpoint
    start_epoch = 0
    latest_path = os.path.join(out_dir, "latest_model.pth")
    if os.path.exists(latest_path):
        try:
            print(f"[pretrain] Resuming from {latest_path}")
            model.load_state_dict(torch.load(latest_path, map_location=PYTORCH_DEVICE))
            ckpts = sorted(glob_ckpt_epochs(out_dir))
            if ckpts:
                start_epoch = ckpts[-1][0]
            print(f"[pretrain] Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"[pretrain] Checkpoint corrupted ({e}), starting from scratch")
            os.remove(latest_path)
            start_epoch = 0

    # csv log file
    csv_filename = os.path.join(out_dir, "training_log.csv")
    det_names = ['ETD', 'T1', 'T2']
    param_headers = []
    for d in det_names:
        param_headers.extend([f"Px_{d}", f"Py_{d}", f"Pmag_{d}", f"Alpha_{d}"])
    headers = ["Epoch", "Stage", "LR", "Total", "Rec", "TV_Geo", "TV_C",
               "Sphere", "Comp_Uni", "Eta_Inter_Det_Corr", "Eta_Whiteness",
               "Rl_Reg"] + param_headers
    file_mode = 'a' if start_epoch > 0 and os.path.exists(csv_filename) else 'w'
    with open(csv_filename, file_mode) as f:
        if file_mode == 'w':
            csv.writer(f).writerow(headers)

    best_stage_loss = float('inf')
    stage_prev_epoch = 0

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        # time check
        elapsed = time.time() - start_time
        if elapsed > time_limit_s:
            print(f"[pretrain] Time limit reached at epoch {epoch}. Saving and exiting.")
            torch.save(model.state_dict(), latest_path)
            return os.path.join(out_dir, "best_model_stage3.pth") if os.path.exists(os.path.join(out_dir, "best_model_stage3.pth")) else latest_path

        model.train()

        # stage scheduling
        if epoch < 500:
            stage_now = 1
        elif epoch < 1500:
            stage_now = 2
        else:
            stage_now = 3

        if stage_now != stage_prev_epoch:
            print(f"\n[pretrain] Transitioning to Stage {stage_now}")
            best_stage_loss = float('inf')
            stage_prev_epoch = stage_now
            for pg in optimizer.param_groups:
                pg['lr'] = START_LEARNING_RATE
            optimizer.state.clear()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500)
        current_lr = optimizer.param_groups[0]['lr']

        # detector param unfreezing
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

        acc = {k: 0.0 for k in ["Total", "Rec", "TV_Geo", "TV_C", "Sphere",
                                 "Comp_Uni", "Eta_Inter_Det_Corr", "Eta_Whiteness", "Rl_Reg"]}
        sphere_count = 0
        comp_count = 0

        loop = tqdm(loader, desc=f"PT E{epoch+1}/{TOTAL_EPOCHS} [S{stage_now}]")
        for input_stack, normals, mask in loop:
            input_stack = input_stack.to(PYTORCH_DEVICE, non_blocking=True)
            normals = normals.to(PYTORCH_DEVICE, non_blocking=True)
            mask = mask.to(PYTORCH_DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                f_img, c, nx, ny, nz, r_low, _, eta, _ = model(input_stack)

                loss_rec = F.l1_loss(f_img, input_stack)
                loss_tv_geo = dirichlet_energy(nx) + dirichlet_energy(ny)
                loss_tv_c = dirichlet_energy(c)
                loss_sphere = sphere_normal_loss(nx, ny, nz, normals, mask) if LOSS_COMPONENT_WEIGHTS_PRETRAINING["sphere"][stage_now] > 0 else None
                loss_c_uni = uniform_composition_loss(c, mask) if LOSS_COMPONENT_WEIGHTS_PRETRAINING["comp_uni"][stage_now] > 0 else None
                loss_eta_inter = inter_channel_correlation(eta)
                loss_eta_white = autocorrelation(eta)
                loss_r_low_reg = torch.abs(r_low).mean()
                loss_t1_p_mag = (model.dr_module.p[1, 0]**2 + model.dr_module.p[1, 1]**2)
                loss_t1_alpha = torch.sigmoid(model.dr_module.alpha_logits.view(3)[1])
                loss_t2_alpha = 1.0 - torch.sigmoid(model.dr_module.alpha_logits.view(3)[2])

                loss = (
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["rec"][stage_now] * loss_rec +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["tv_geo"][stage_now] * loss_tv_geo +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["tv_c"][stage_now] * loss_tv_c +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["eta_inter_det_corr"][stage_now] * loss_eta_inter +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["white"][stage_now] * loss_eta_white +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["r_low_reg"][stage_now] * loss_r_low_reg +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["t1_p_mag"][stage_now] * loss_t1_p_mag +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["t1_alpha"][stage_now] * loss_t1_alpha +
                    LOSS_COMPONENT_WEIGHTS_PRETRAINING["t2_alpha"][stage_now] * loss_t2_alpha +
                    (torch.abs(torch.mean(nx, dim=(0, 2, 3))) + torch.abs(torch.mean(ny, dim=(0, 2, 3)))).mean()
                )

                if loss_sphere is not None:
                    loss += LOSS_COMPONENT_WEIGHTS_PRETRAINING["sphere"][stage_now] * loss_sphere
                    acc["Sphere"] += loss_sphere.item()
                    sphere_count += 1
                if loss_c_uni is not None:
                    loss += LOSS_COMPONENT_WEIGHTS_PRETRAINING["comp_uni"][stage_now] * loss_c_uni
                    acc["Comp_Uni"] += loss_c_uni.item()
                    comp_count += 1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc["Total"] += loss.item()
            acc["Rec"] += loss_rec.item()
            acc["TV_Geo"] += loss_tv_geo.item()
            acc["TV_C"] += loss_tv_c.item()
            acc["Eta_Inter_Det_Corr"] += loss_eta_inter.item()
            acc["Eta_Whiteness"] += loss_eta_white.item()
            acc["Rl_Reg"] += loss_r_low_reg.item()

            loop.set_postfix(l_tot=loss.item(), l_rec=loss_rec.item(),
                             lr=current_lr)

        loss_total_avg = acc["Total"] / len(loader)
        scheduler.step(loss_total_avg)

        # logging
        with torch.no_grad():
            vecs = model.dr_module.p.detach().cpu().numpy()
            mags = np.sqrt(vecs[:, 0]**2 + vecs[:, 1]**2)
            mix_p = torch.sigmoid(model.dr_module.alpha_logits).view(3).detach().cpu().numpy()
            param_vals = []
            for i in range(3):
                param_vals.extend([vecs[i, 0], vecs[i, 1], mags[i], mix_p[i]])

        for k in ["Total", "Rec", "TV_Geo", "TV_C", "Eta_Inter_Det_Corr", "Eta_Whiteness", "Rl_Reg"]:
            acc[k] /= len(loader)
        if sphere_count > 0:
            acc["Sphere"] /= sphere_count
        if comp_count > 0:
            acc["Comp_Uni"] /= comp_count

        csv_row = [epoch + 1, stage_now, current_lr] + [acc[k] for k in headers[3:12]] + param_vals
        with open(csv_filename, 'a') as f:
            csv.writer(f).writerow(csv_row)

        # checkpoint saving
        if loss_total_avg < best_stage_loss and not np.isnan(loss_total_avg) and sphere_count > 0 and comp_count > 0:
            best_stage_loss = loss_total_avg
            save_name = f"best_model_stage{stage_now}.pth"
            torch.save(model.state_dict(), os.path.join(out_dir, save_name))
            print(f"[pretrain] New best (S{stage_now}): epoch {epoch+1} loss {best_stage_loss:.5f}")

        torch.save(model.state_dict(), latest_path)
        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    best_path = os.path.join(out_dir, "best_model_stage3.pth")
    if not os.path.exists(best_path):
        best_path = latest_path
    return best_path


# __ finetune ______________________________________________________________

def run_finetune(config: dict, eval_specimens: list,
                 data_root: str, reg_csv: str, pretrained_ckpt: str,
                 out_dir: str, time_limit_s: float, start_time: float) -> str:
    """Finetune a single config × fold.  Returns the best checkpoint path."""
    os.makedirs(out_dir, exist_ok=True)
    latent_ch = config["latent_channels"]
    arch_config = config["arch_config"]
    r_high_skip = config["r_high_skip_connection"]

    model = ProposedModelArchitecture(
        latent_channels=latent_ch,
        r_high_skip_connection=r_high_skip,
        arch_config=arch_config,
    ).to(PYTORCH_DEVICE)

    reg_data = load_intra_sample_registration_data(reg_csv)
    all_triplets = parse_data_structure(data_root, None, reg_data)
    train_triplets = [t for t in all_triplets
                      if not any(e in t['specimen_dir'] for e in eval_specimens)]
    if not train_triplets:
        raise ValueError("No finetuning triplets found")

    print(f"[finetune] Loading pretrained checkpoint: {pretrained_ckpt}")
    model.load_state_dict(torch.load(pretrained_ckpt, map_location=PYTORCH_DEVICE))

    epoch_len = BATCHES_PER_EPOCH * PATCHES_PER_BATCH
    ft_dataset = ScanTripletWithFinetuningDataPatchSampler(
        train_triplets, reg_data, model, PATCH_SIZE_PX, epoch_len)
    ft_loader = DataLoader(ft_dataset, PATCHES_PER_BATCH, shuffle=True,
                           num_workers=8, persistent_workers=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=START_LEARNING_RATE, fused=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
    scaler = torch.amp.GradScaler('cuda')

    # freeze DR module
    model.dr_module.p.requires_grad = False
    model.dr_module.alpha_logits.requires_grad = False
    model.dr_module.beta.requires_grad = False
    model.dr_module.w.requires_grad = False
    model.dr_module.b.requires_grad = False

    # resume from latest finetune checkpoint
    start_epoch = 0
    latest_ft_path = os.path.join(out_dir, "latest_model_finetune.pth")
    if os.path.exists(latest_ft_path):
        try:
            print(f"[finetune] Resuming from {latest_ft_path}")
            model.load_state_dict(torch.load(latest_ft_path, map_location=PYTORCH_DEVICE))
        except Exception as e:
            print(f"[finetune] Checkpoint corrupted ({e}), starting from scratch")
            os.remove(latest_ft_path)

    # csv log file
    csv_filename = os.path.join(out_dir, "finetune_log.csv")
    ft_headers = ["Epoch", "LR", "Total", "Rec", "C_Match", "N_Match", "Rl_Reg", "Eta_Whiteness"]
    file_mode = 'a' if start_epoch > 0 and os.path.exists(csv_filename) else 'w'
    with open(csv_filename, file_mode) as f:
        if file_mode == 'w':
            csv.writer(f).writerow(ft_headers)

    best_ft_loss = float('inf')

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > time_limit_s:
            print(f"[finetune] Time limit reached at epoch {epoch}. Saving and exiting.")
            torch.save(model.state_dict(), latest_ft_path)
            best_path = os.path.join(out_dir, "best_model_finetune.pth")
            return best_path if os.path.exists(best_path) else latest_ft_path

        model.train()
        acc = {"Total": 0.0, "Rec": 0.0, "C_Match": 0.0, "N_Match": 0.0, "Rl_Reg": 0.0, "Eta_Whiteness": 0.0}

        loop = tqdm(ft_loader, desc=f"FT E{epoch+1}/{TOTAL_EPOCHS}")
        for input_stack, target_c, target_normals in loop:
            input_stack = input_stack.to(PYTORCH_DEVICE, non_blocking=True)
            target_c = target_c.to(PYTORCH_DEVICE, non_blocking=True)
            target_normals = target_normals.to(PYTORCH_DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                f_img, c, nx, ny, nz, r_low, _, eta, _ = model(input_stack)
                n = torch.cat([nx, ny, nz], dim=1)

                loss_rec = F.l1_loss(f_img, input_stack)
                loss_c_match = F.l1_loss(c, target_c)
                loss_n_match = F.l1_loss(n, target_normals)
                loss_r_low_reg = torch.abs(r_low).mean()
                loss_eta_white = autocorrelation(eta)

                loss = (
                    LOSS_COMPONENT_WEIGHTS_FINETUNING["rec"] * loss_rec +
                    LOSS_COMPONENT_WEIGHTS_FINETUNING["c_match"] * loss_c_match +
                    LOSS_COMPONENT_WEIGHTS_FINETUNING["n_match"] * loss_n_match +
                    LOSS_COMPONENT_WEIGHTS_FINETUNING["r_low_reg"] * loss_r_low_reg +
                    LOSS_COMPONENT_WEIGHTS_FINETUNING["white"] * loss_eta_white
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc["Total"] += loss.item()
            acc["Rec"] += loss_rec.item()
            acc["C_Match"] += loss_c_match.item()
            acc["N_Match"] += loss_n_match.item()
            acc["Rl_Reg"] += loss_r_low_reg.item()
            acc["Eta_Whiteness"] += loss_eta_white.item()

            loop.set_postfix(l_tot=loss.item(), l_c=loss_c_match.item(), l_n=loss_n_match.item())

        loss_total_avg = acc["Total"] / len(ft_loader)
        scheduler.step(loss_total_avg)

        csv_row = [epoch + 1, optimizer.param_groups[0]['lr']] + [acc[k] / len(ft_loader) for k in ft_headers[2:]]
        with open(csv_filename, 'a') as f:
            csv.writer(f).writerow(csv_row)

        if loss_total_avg < best_ft_loss:
            best_ft_loss = loss_total_avg
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model_finetune.pth"))
            print(f"[finetune] New best at epoch {epoch+1} (loss: {best_ft_loss:.5f})")

        torch.save(model.state_dict(), latest_ft_path)

    best_path = os.path.join(out_dir, "best_model_finetune.pth")
    if not os.path.exists(best_path):
        best_path = latest_ft_path
    return best_path


# __ helpers _______________________________________________________________

def glob_ckpt_epochs(out_dir: str):
    """Return list of (epoch_num, path) sorted by epoch."""
    results = []
    for f in os.listdir(out_dir):
        m = re.match(r'checkpoint_epoch_(\d+)\.pth', f)
        if m:
            results.append((int(m.group(1)), os.path.join(out_dir, f)))
    results.sort(key=lambda x: x[0])
    return results


# __ main orchestrator _____________________________________________________

def main():
    parser = argparse.ArgumentParser(description="Ablation training orchestrator")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Root of the experimental dataset (trinity_sem_dataset/)")
    parser.add_argument('--time_limit_hours', type=float, default=47.0,
                        help="Soft time limit in hours")
    args = parser.parse_args()

    print(f"pytorch device: {PYTORCH_DEVICE}")
    start_time = time.time()
    time_limit_s = args.time_limit_hours * 3600

    # ensure data artefacts exist
    print("[orchestrator] Ensuring training data artefacts…")
    particles_csv, reg_csv = ensure_training_data(args.data_root)

    # load / init state
    state = init_state()

    # __ main loop: pretrain→finetune cycles _____________________________

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit_s:
            print("[orchestrator] Time limit reached. Exiting gracefully.")
            break

        # claim next pending pretrain
        task = claim_next_pretrain()
        if task is not None:
            rid = task["rid"]
            cfg = task["config"]
            fk = task["fold_key"]
            eval_spec = task["eval_specimens"]
            out_dir = pretrain_dir(cfg["id"], fk)

            print(f"\n{'='*60}")
            print(f"[orchestrator] PRETRAIN: {cfg['id']} | {fk}")
            print(f"{'='*60}")

            try:
                best_ckpt = run_pretrain(cfg, eval_spec, args.data_root,
                                         particles_csv, out_dir, time_limit_s, start_time)
                state = load_state()
                mark_pretrain_completed(state, rid, best_ckpt)
                print(f"[orchestrator] Pretrain completed: {cfg['id']} | {fk}")
            except Exception as e:
                print(f"[orchestrator] Pretrain FAILED: {e}")
                raise

            continue

        # claim next pending finetune
        task = claim_next_finetune()
        if task is not None:
            rid = task["rid"]
            cfg = task["config"]
            fk = task["fold_key"]
            eval_spec = task["eval_specimens"]
            out_dir = finetune_dir(cfg["id"], fk)
            pretrained_ckpt = resolve_state_path(task["pretrain"]["checkpoint"]) or pretrain_checkpoint(cfg["id"], fk)


            print(f"\n{'='*60}")
            print(f"[orchestrator] FINETUNE: {cfg['id']} | {fk}")
            print(f"{'='*60}")

            try:
                best_ckpt = run_finetune(cfg, eval_spec, args.data_root,
                                         reg_csv, pretrained_ckpt, out_dir,
                                         time_limit_s, start_time)
                state = load_state()
                mark_finetune_completed(state, rid, best_ckpt)
                print(f"[orchestrator] Finetune completed: {cfg['id']} | {fk}")
            except Exception as e:
                print(f"[orchestrator] Finetune FAILED: {e}")
                raise

            continue

        # nothing left to train
        print("[orchestrator] All training tasks are complete!")
        state = load_state()
        print_progress(state)
        break

    print_progress(state)
    print("[orchestrator] Done.")


if __name__ == "__main__":
    main()
