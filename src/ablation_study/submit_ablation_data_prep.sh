#!/bin/bash
#PBS -N AblDataPrep
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=10:mem=64gb:scratch_local=256gb:ngpus=1:gpu_mem=32gb
#PBS -l walltime=12:00:00
#PBS -m ae

# __ Ablation data preparation submission script ________________________
# Submit from the project root directory:
#   qsub proposed_method/submit_ablation_data_prep.sh
#
# Creates / verifies all cached data artefacts:
#   1. spherical_particles.csv  (Cellpose detection on SiAl1)
#   2. registration_shifts.csv   (phase cross-correlation)
#   3. pseudo_ground_truth/      (SNR-weighted registered averages)
#   4. noise_model_coefficients.json
#   5. noisy PGT samples
#
# Only needs to be run once.  Subsequent training/eval jobs will reuse
# the cached artefacts automatically.

SINGULARITY_IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF"

PROJECT_ROOT="$PBS_O_WORKDIR"
ABLATION_DIR="$PROJECT_ROOT/src/ablation_study"
DATA_ROOT="$PROJECT_ROOT/trinity_sem_dataset"

cd "$ABLATION_DIR"

singularity exec --nv \
    -B "$SCRATCHDIR" \
    -B /storage \
    -B "$PROJECT_ROOT:$PROJECT_ROOT" \
    "$SINGULARITY_IMAGE" \
    bash -c 'cd "$0" && pip install --quiet tifffile cellpose 2>/dev/null; python ablation_data_prep.py --data_root "$1"' "$ABLATION_DIR" "$DATA_ROOT"
