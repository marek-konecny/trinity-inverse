if __name__ != "__main__": raise RuntimeError("Only run this directly as the main entry point.")
import os
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from _1_constants import PYTORCH_DEVICE, PATCHES_PER_BATCH, PATCH_SIZE_PX, BATCHES_PER_EPOCH, START_LEARNING_RATE, TOTAL_EPOCHS, DEFAULT_EVALUATION_SPECIMEN_NAMES, LOSS_COMPONENT_WEIGHTS_FINETUNING
from _2_data_loading_helpers import load_intra_sample_registration_data, parse_data_structure
from _3_dataset_classes import ScanTripletWithFinetuningDataPatchSampler
from _4_model_architecture import TrinityInverseModel
from _5_loss_helpers import autocorrelation


# 0. CONFIG
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_root', type=str, required=True)
parser.add_argument('--intra_sample_registration_data_csv_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default=".")
parser.add_argument('--evaluation_specimen_names', type=str, default=DEFAULT_EVALUATION_SPECIMEN_NAMES["fold 2"],
    help="Comma-delimited list of evaluation specimen directory names. These will be excluded from the training split."
)
parser.add_argument('--finetune_pretrained_checkpoint_path', type=str, required=True)
args = parser.parse_args()
args.evaluation_specimen_names = [d.strip() for d in args.evaluation_specimen_names.split(',')]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; torch.backends.cudnn.benchmark = True
print(f"pytorch device: {PYTORCH_DEVICE}")


# 1. SETUP
# ==========================================
print("Running setup")

intra_sample_registration_data = load_intra_sample_registration_data(args.intra_sample_registration_data_csv_path)
all_triplets = parse_data_structure(args.input_data_root, None, intra_sample_registration_data)
if len(all_triplets) == 0: raise ValueError("no triplets found")

model = TrinityInverseModel().to(PYTORCH_DEVICE)


# 2. FINETUNING
# ==========================================
print(f"\nStarting finetuning from pre-trained checkpoint: {args.finetune_pretrained_checkpoint_path}")
model.load_state_dict(torch.load(args.finetune_pretrained_checkpoint_path, map_location=PYTORCH_DEVICE))

finetuning_triplets = [triplet_info for triplet_info in all_triplets if not any(eval_dir in triplet_info['specimen_dir'] for eval_dir in args.evaluation_specimen_names)]
ft_dataset = ScanTripletWithFinetuningDataPatchSampler(finetuning_triplets, intra_sample_registration_data, model, PATCH_SIZE_PX, BATCHES_PER_EPOCH * PATCHES_PER_BATCH)
ft_loader = DataLoader(ft_dataset, PATCHES_PER_BATCH, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

optimizer = optim.AdamW(model.parameters(), lr=START_LEARNING_RATE, fused=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
scaler = torch.amp.GradScaler('cuda')

model.dr_module.p.requires_grad = False
model.dr_module.alpha_logits.requires_grad = False
model.dr_module.beta.requires_grad = False
model.dr_module.w.requires_grad = False
model.dr_module.b.requires_grad = False

best_ft_loss = float('inf')

for epoch in range(TOTAL_EPOCHS):
    model.train()
    acc_stats = {"Total": 0.0, "Rec": 0.0, "C_Match": 0.0, "N_Match": 0.0, "Eta_Inter_Det_Corr": 0.0, "Eta_Whiteness": 0.0, "Rl_Reg": 0.0}

    loop = tqdm(ft_loader, desc=f"FT Epoch {epoch+1}/{TOTAL_EPOCHS}")
    for input_stack, target_c, target_normals in loop:
        input_stack = input_stack.to(PYTORCH_DEVICE, non_blocking=True)
        target_c = target_c.to(PYTORCH_DEVICE, non_blocking=True)
        target_normals = target_normals.to(PYTORCH_DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            f_img, c, nx, ny, nz, r_low, r_high, eta, g = model(input_stack)
            n = torch.cat([nx, ny, nz], dim=1)

            loss_rec = F.l1_loss(f_img, input_stack)
            loss_c_match = F.l1_loss(c, target_c)
            loss_n_match = F.l1_loss(n, target_normals)
            loss_r_low_reg = torch.abs(r_low).mean()
            loss_eta_whiteness = autocorrelation(eta)

            loss = (
                (LOSS_COMPONENT_WEIGHTS_FINETUNING["rec"]       * loss_rec) +
                (LOSS_COMPONENT_WEIGHTS_FINETUNING["c_match"]   * loss_c_match) +
                (LOSS_COMPONENT_WEIGHTS_FINETUNING["n_match"]   * loss_n_match) +
                (LOSS_COMPONENT_WEIGHTS_FINETUNING["r_low_reg"] * loss_r_low_reg) +
                (LOSS_COMPONENT_WEIGHTS_FINETUNING["white"]     * loss_eta_whiteness)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc_stats["Total"] += loss.item()
        acc_stats["Rec"] += loss_rec.item()
        acc_stats["C_Match"] += loss_c_match.item()
        acc_stats["N_Match"] += loss_n_match.item()
        acc_stats["Rl_Reg"] += loss_r_low_reg.item()
        acc_stats["Eta_Whiteness"] += loss_eta_whiteness.item()

        loop.set_postfix(l_tot=loss.item(), l_rec=loss_rec.item(), l_c=loss_c_match.item(), l_n=loss_n_match.item(), l_rl=loss_r_low_reg.item(), l_white=loss_eta_whiteness.item(), lr=optimizer.param_groups[0]['lr'])

    loss_total_avg = acc_stats["Total"] / len(ft_loader)
    scheduler.step(loss_total_avg)

    if loss_total_avg < best_ft_loss:
        best_ft_loss = loss_total_avg
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_finetune.pth"))
        print(f"New best finetuned model saved at epoch {epoch+1} (loss: {best_ft_loss:.5f})")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "latest_model_finetune.pth"))
