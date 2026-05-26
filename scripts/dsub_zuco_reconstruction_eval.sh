#!/bin/bash
# DSUB job script for paper-style ZuCo sentence reconstruction + evaluation.
#
# Usage:
#   TASK=ZuCoNR SENT_EMB_PATH=/abs/path/sentence_embeddings.pt EMBED_MODEL_PATH=/abs/path/model \
#   LLM_ENDPOINT=https://.../v1/chat/completions LLM_MODEL=gpt-5.4 LLM_API_KEY=... \
#   dsub -s scripts/dsub_zuco_reconstruction_eval.sh

#DSUB -n zuco_reconstruction_eval
#DSUB -N 1
#DSUB -A root.project.P23Z10200N0876
#DSUB -R "cpu=8;gpu=1;mem=64000"
#DSUB -oo /home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/log/submit/%J.zuco_reconstruction_eval.out
#DSUB -eo /home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/log/submit/%J.zuco_reconstruction_eval.err

set -euo pipefail

TASK="${TASK:?Set TASK=ZuCoSR|ZuCoNR|ZuCoTSR before submit}"
SENT_EMB_PATH="${SENT_EMB_PATH:?Set SENT_EMB_PATH to sentence_embeddings.pt}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:?Set EMBED_MODEL_PATH to embedding model dir/name}"
LLM_ENDPOINT="${LLM_ENDPOINT:?Set LLM_ENDPOINT}"
LLM_MODEL="${LLM_MODEL:?Set LLM_MODEL}"
LLM_API_KEY="${LLM_API_KEY:?Set LLM_API_KEY}"

JOB_PATH="${JOB_PATH:-/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26}"
LOG_DIR="${LOG_DIR:-$JOB_PATH/log/${TASK}_reconstruction_eval}"
SUBMIT_LOG_DIR="$JOB_PATH/log/submit"
mkdir -p "$LOG_DIR" "$SUBMIT_LOG_DIR"

HOST_SHORT="$(hostname -s)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/${TASK}_reconstruction_eval_${RUN_TS}_${HOST_SHORT}.log"

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module load "${GCC_MODULE:-compilers/gcc/9.3.0}"
module load "${CUDA_MODULE:-compilers/cuda/11.8}"
module load "${OPENBLAS_MODULE:-libs/openblas/0.3.26_gcc9.3.0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

CONDA_SH="${CONDA_SH:-/home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-BrainMosaic}"
PYTHON_BIN="${PYTHON_BIN:-/home/share/huadjyin/home/sunmengmeng/.conda/envs/BrainMosaic/bin/python}"
PIPELINE_SCRIPT="$JOB_PATH/scripts/run_zuco_reconstruction_eval.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

export THRESHOLD_TAG="${THRESHOLD_TAG:-t0p78}"
export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-180}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-5}"
export LLM_RETRY_BACKOFF_SEC="${LLM_RETRY_BACKOFF_SEC:-3}"
export NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
export MAX_TOKENS="${MAX_TOKENS:-120}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export RECON_FLUSH_EVERY="${RECON_FLUSH_EVERY:-20}"
export EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-128}"
export EVAL_TRUNCATE_DIM="${EVAL_TRUNCATE_DIM:-256}"

{
  echo "[INFO] start time: $(date '+%F %T')"
  echo "[INFO] host: $HOST_SHORT"
  echo "[INFO] task: $TASK"
  echo "[INFO] threshold tag: $THRESHOLD_TAG"
  echo "[INFO] job path: $JOB_PATH"
  echo "[INFO] pipeline script: $PIPELINE_SCRIPT"
  echo "[INFO] sentence embeddings: $SENT_EMB_PATH"
  echo "[INFO] embedding model: $EMBED_MODEL_PATH"
  echo "[INFO] llm endpoint: $LLM_ENDPOINT"
  echo "[INFO] llm model: $LLM_MODEL"
  echo "[INFO] num candidates: $NUM_CANDIDATES"
  echo "[INFO] recon flush every: $RECON_FLUSH_EVERY"
  module list
} 2>&1 | tee -a "$RUN_LOG"

(
  cd "$JOB_PATH"
  bash "$PIPELINE_SCRIPT" "$TASK"
) 2>&1 | tee -a "$RUN_LOG"

echo "[INFO] end time: $(date '+%F %T')" | tee -a "$RUN_LOG"
echo "[INFO] run log: $RUN_LOG" | tee -a "$RUN_LOG"
