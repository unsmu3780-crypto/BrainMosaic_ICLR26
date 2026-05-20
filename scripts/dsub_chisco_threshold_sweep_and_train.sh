#!/bin/bash
# DSUB job script for Chisco token-bank threshold sweep and optional full training.
#
# Usage examples:
#   dsub -s scripts/dsub_chisco_threshold_sweep_and_train.sh
#
# Override defaults at submit time if needed:
#   RUN_FULL=0 dsub -s scripts/dsub_chisco_threshold_sweep_and_train.sh
#   THRESHOLDS="0.78 0.85 0.90 0.95" FULL_EPOCHS=50 FULL_BATCH_SIZE=16 dsub -s scripts/dsub_chisco_threshold_sweep_and_train.sh
#
# This job assumes:
#   1. Chisco EEG splits already exist under the configured data root.
#   2. Chisco text assets and Qwen embeddings have already been generated.
#   3. configs/train.chisco.json and configs/token_bank.chisco.json exist.
#   4. The local Qwen model is ignored by git and stored under models/Qwen3-Embedding-8B.

#DSUB -n chisco_threshold_sweep
#DSUB -N 1
#DSUB -A root.project.P23Z10200N0876_tmp
#DSUB -R "cpu=8;gpu=1;mem=80000"
#DSUB -oo /home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/log/submit/%J.chisco_threshold_sweep.out
#DSUB -eo /home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/log/submit/%J.chisco_threshold_sweep.err

set -euo pipefail

# -----------------------------
# 1. Project paths and logging
# -----------------------------

JOB_PATH="${JOB_PATH:-/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26}"
LOG_DIR="${LOG_DIR:-$JOB_PATH/log/chisco_threshold_sweep}"
SUBMIT_LOG_DIR="$JOB_PATH/log/submit"
mkdir -p "$LOG_DIR" "$SUBMIT_LOG_DIR"

HOST_SHORT="$(hostname -s)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/chisco_threshold_sweep_${RUN_TS}_${HOST_SHORT}.log"

# -----------------------------
# 2. HPC module environment
# -----------------------------
#
# These modules mirror the interactive environment that made torch/Qwen run:
#   - gcc/9.3.0
#   - cuda/11.8
#   - nccl/cuda11.8
#   - cudnn/cuda11
#   - openblas, needed for libopenblas.so.0

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge

module load "${GCC_MODULE:-compilers/gcc/9.3.0}"
module load "${CUDA_MODULE:-compilers/cuda/11.8}"
module load "${NCCL_MODULE:-libs/nccl/2.16.5-1_cuda11.8}"
module load "${CUDNN_MODULE:-libs/cudnn/8.8.1_cuda11}"
module load "${OPENBLAS_MODULE:-libs/openblas/0.3.26_gcc9.3.0}"

# Keep CPU thread use aligned with the requested CPU count.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

# -----------------------------
# 3. Python environment
# -----------------------------
#
# Use the known BrainMosaic conda environment directly. This avoids depending on
# the shell's conda activation setup inside a batch job.

PYTHON_BIN="${PYTHON_BIN:-/home/share/huadjyin/home/sunmengmeng/.conda/envs/BrainMosaic/bin/python}"
PIPELINE_SCRIPT="$JOB_PATH/scripts/run_chisco_threshold_sweep_and_train.sh"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN" | tee -a "$RUN_LOG"
  exit 1
fi

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "[ERROR] missing pipeline script: $PIPELINE_SCRIPT" | tee -a "$RUN_LOG"
  exit 1
fi

# Make sure subprocesses use the same Python as the conda environment.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

# -----------------------------
# 4. Sweep/full-training controls
# -----------------------------
#
# Defaults:
#   - Sweep four thresholds, including the author's public default 0.78.
#   - Run 1-epoch smoke validation for each threshold.
#   - Then run full training with the best smoke threshold.
#
# Set RUN_FULL=0 if you only want the sweep summary and no full training.

export THRESHOLDS="${THRESHOLDS:-0.78 0.85 0.90 0.95}"
export SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
export SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
export RUN_FULL="${RUN_FULL:-1}"
export FULL_EPOCHS="${FULL_EPOCHS:-50}"
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
export FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-4}"

export BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-configs/train.chisco.json}"
export BASE_TOKEN_CONFIG="${BASE_TOKEN_CONFIG:-configs/token_bank.chisco.json}"
export SWEEP_ROOT="${SWEEP_ROOT:-outputs/chisco_threshold_sweep}"
export FULL_ROOT="${FULL_ROOT:-outputs/chisco_full}"

# Optional manual final-threshold override. Leave empty to auto-select from
# sweep_summary.json by matching_acc, then mean_cosine.
export BEST_THRESHOLD="${BEST_THRESHOLD:-}"

# -----------------------------
# 5. Record environment and run
# -----------------------------

{
  echo "[INFO] start time: $(date '+%F %T')"
  echo "[INFO] host: $HOST_SHORT"
  echo "[INFO] job path: $JOB_PATH"
  echo "[INFO] python: $PYTHON_BIN"
  echo "[INFO] pipeline script: $PIPELINE_SCRIPT"
  echo "[INFO] thresholds: $THRESHOLDS"
  echo "[INFO] smoke: epochs=$SMOKE_EPOCHS batch_size=$SMOKE_BATCH_SIZE num_workers=$SMOKE_NUM_WORKERS"
  echo "[INFO] full: run_full=$RUN_FULL epochs=$FULL_EPOCHS batch_size=$FULL_BATCH_SIZE num_workers=$FULL_NUM_WORKERS best_threshold=${BEST_THRESHOLD:-auto}"
  echo "[INFO] sweep root: $SWEEP_ROOT"
  echo "[INFO] full root: $FULL_ROOT"
  echo "[INFO] loaded modules:"
  module list
  echo "[INFO] python check:"
  "$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY
} 2>&1 | tee -a "$RUN_LOG"

(
  cd "$JOB_PATH"
  bash "$PIPELINE_SCRIPT"
) 2>&1 | tee -a "$RUN_LOG"

echo "[INFO] end time: $(date '+%F %T')" | tee -a "$RUN_LOG"
echo "[INFO] run log: $RUN_LOG" | tee -a "$RUN_LOG"
