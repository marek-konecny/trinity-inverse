#!/bin/bash
#PBS -N AblEval
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=10:mem=64gb:scratch_local=256gb:ngpus=1:gpu_mem=32gb
#PBS -l walltime=12:00:00
#PBS -m ae

# __ Ablation evaluation submission script ______________________________
# Submit from the project root directory:
#   qsub proposed_method/submit_ablation_evaluate.sh
#
# This script evaluates as many completed training runs as fit within the
# 12-hour walltime, then exits.  Re-submit to continue.

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
    bash -c 'cd "$0" && pip install --quiet tifffile 2>/dev/null; python ablation_evaluate.py --data_root "$1" --time_limit_hours 11.5' "$ABLATION_DIR" "$DATA_ROOT"
