#!/bin/bash
#PBS -N AblTrain
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=10:mem=32gb:scratch_local=256gb:ngpus=1:gpu_mem=32gb
#PBS -l walltime=48:00:00
#PBS -m ae

# __ Ablation training submission script _________________________________
# Submit from the project root directory:
#   qsub proposed_method/submit_ablation_train.sh
#
# This script trains as many ablation configurations as fit within the
# 48-hour walltime, then exits.  Re-submit to continue from where it
# left off — the state file tracks all progress.

SINGULARITY_IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF"

# PBS_O_WORKDIR = directory from which qsub was called (project root)
PROJECT_ROOT="$PBS_O_WORKDIR"
ABLATION_DIR="$PROJECT_ROOT/src/ablation_study"

# Data root relative to project root
DATA_ROOT="$PROJECT_ROOT/trinity_sem_dataset"

cd "$ABLATION_DIR"

singularity exec --nv \
    -B "$SCRATCHDIR" \
    -B /storage \
    -B "$PROJECT_ROOT:$PROJECT_ROOT" \
    "$SINGULARITY_IMAGE" \
    bash -c 'cd "$0" && pip install --quiet tifffile cellpose 2>/dev/null; python ablation_train.py --data_root "$1" --time_limit_hours 47.0' "$ABLATION_DIR" "$DATA_ROOT"
