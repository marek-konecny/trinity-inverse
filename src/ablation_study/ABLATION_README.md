# Ablation Study Pipeline

This pipeline evaluates different IID network architectures for the
proposed multi-detector SEM image decomposition method. It is designed
to run on a PBS cluster and supports **save/resume**
across multiple job submissions.

## Ablation Grid

| Group             | Configurations                           | Count |
|-------------------|------------------------------------------|-------|
| Encoder + ASPP    | 1/2/4/8 layers × 8/16/48/96 channels     | 16    |
| UNet only         | 8/16/48/96 output channels               | 4     |
| Encoder + UNet    | 1/2/4/8 layers × 8/16/48/96 channels     | 16    |
| **Total**         |                                          | **36** |

Each configuration is evaluated on **7 evaluation folds**, and each run
goes through **4 phases**:

1. **Pretrain** - self-supervised training with sparse spherical particle supervision
2. **Eval (pretrain)** - objective evaluation on held-out fold
3. **Finetune** - finetuning with pseudo-labels from pretrained model
4. **Eval (finetune)** - objective evaluation after finetuning

**Total: 36 × 7 = 252 runs × 4 phases = 1008 sub-tasks**



## Quick Start

### 1. Data preparation (run once)

```bash
qsub src/ablation_study/submit_ablation_data_prep.sh
```

This creates:
- `spherical_particles.csv` (Cellpose detection on SiAl1)
- `registration_shifts.csv` (phase cross-correlation registration estimates)
- `pseudo_ground_truth/` (SNR-weighted registered averages)
- `noise_model_coefficients.json` (per-detector noise scaling)
- Noisy PGT samples (30 σ levels × 15 realizations)

All artefacts are cached under `src/ablation_study/ablation_results/data_cache/`
and `src/ablation_study/ablation_results/pgt_cache/`.

### 2. Training jobs (repeat as needed)

```bash
qsub src/ablation_study/submit_ablation_train.sh
```

Each job trains as many configurations as fit within the 48-hour walltime, then
exits gracefully. Re-submitting continues from where the previous job stopped.

### 3. Evaluation jobs (repeat as needed)

```bash
qsub src/ablation_study/submit_ablation_evaluate.sh
```

Each job evaluates as many completed training runs as fit within the 12-hour
walltime. Re-submitting continues with the remaining tasks.

### 4. Progress monitoring

The state file at `src/ablation_study/ablation_results/ablation_state.json` tracks
all 1008 sub-tasks. Progress can be inspected:

```bash
cd src/ablation_study && python -c "from ablation_state import load_state, print_progress; print_progress(load_state())"
```



## Directory Structure

```
src/
  ablation_study/                      # All ablation pipeline scripts
    ablation_config.py
    ablation_state.py
    ablation_data_prep.py
    ablation_train.py
    ablation_evaluate.py
    submit_ablation_data_prep.sh
    submit_ablation_train.sh
    submit_ablation_evaluate.sh
    ABLATION_README.md
    ablation_results/                  # All pipeline outputs (persistent)
    ablation_state.json                # Global state file (tracks all tasks)
    data_cache/
      spherical_particles.csv          # Cached particle detections
      registration_shifts.csv          # Cached registration shifts
    pgt_cache/
      pseudo_ground_truth/
        14MoSiO2 2025_sample00/
          clean/
            ETD.tiff, T1.tiff, T2.tiff
          noisy_sigma0.123_rel00/
            ETD.tiff, T1.tiff, T2.tiff
          ...
        noise_model_coefficients.json
    runs/
      enc4L_ch96_aspp/                 # One directory per config
        fold1/
          pretrain/
            best_model_stage3.pth
            latest_model.pth
            training_log.csv
          finetune/
            best_model_finetune.pth
            latest_model_finetune.pth
            finetune_log.csv
          eval_pretrain/
            evaluation_metrics.csv
            detector_params.json
            images/
            latents/
          eval_finetune/
            evaluation_metrics.csv
            detector_params.json
            images/
            latents/
        fold2/
          ...
      unet_ch48/
        fold1/
          ...
```



## Job Resumability

1. **State file** (`ablation_state.json`): A JSON file that tracks the status
   of every sub-task (`pending` → `in_progress` → `completed`). It is written
   atomically after each status change.

2. **Checkpoint saving**: During pretraining, `latest_model.pth` is saved after
   every epoch, and `checkpoint_epoch_N.pth` every 500 epochs. During
   finetuning, `latest_model_finetune.pth` is saved after every epoch.

3. **Automatic resume**: When a job is re-submitted, the orchestrator loads the
   state file, finds the next `pending` or `in_progress` task, and resumes:
   - For `in_progress` pretrain: loads `latest_model.pth` and continues training
   - For `in_progress` finetune: loads `latest_model_finetune.pth` and continues
   - For `pending` tasks: starts fresh

4. **Time management**: Each job has a soft time limit (47h for training, 11.5h
   for evaluation). The orchestrator checks elapsed time before each epoch and
   exits gracefully when the limit is reached, saving the current checkpoint.



## Evaluation Output

For each evaluated run, the following is saved:

- `evaluation_metrics.csv`: Per-sample, per-detector, per-noise-level PSNR/SSIM
  metrics (both baseline and reconstruction)
- `detector_params.json`: Learned detector parameters (p vectors, alpha, beta, w, b)
- `images/`: Reconstructed TIFF images (clean, medium noise, worst noise)
- `latents/`: Saved latent representations (composition, normals, r_low, r_high, gain)



## Scripts

| File | Description |
|------|-------------|
| `ablation_config.py` | Defines all ablation configurations, folds, and paths |
| `ablation_state.py` | State file management and task selection |
| `ablation_data_prep.py` | Creates/caches CSVs, PGT, and noise-injected data |
| `ablation_train.py` | Training orchestrator (pretrain + finetune) |
| `ablation_evaluate.py` | Evaluation orchestrator (inference + metrics) |