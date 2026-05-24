#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: run_zuco_text_pipeline.sh ZuCoSR|ZuCoNR|ZuCoTSR [template|openai-compatible]}"
BACKEND="${2:-template}"
EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"
TAG="$(echo "${TASK}" | tr '[:upper:]' '[:lower:]')"
ZUCO_REAL_DATA_ROOT="${ZUCO_REAL_DATA_ROOT:-/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data}"
TEXT_ROOT="${TEXT_ROOT:-${ZUCO_REAL_DATA_ROOT}/${TASK}/text_assets}"

python scripts/build_zuco_text_assets_inputs.py --task "${TASK}"

if [[ "${BACKEND}" == "openai-compatible" ]]; then
  python scripts/expand_chisco_tokens.py \
    --text-root "${TEXT_ROOT}" \
    --backend openai-compatible \
    --model "${EXPANSION_MODEL}" \
    --resume
else
  python scripts/expand_chisco_tokens.py \
    --text-root "${TEXT_ROOT}" \
    --backend template \
    --resume
fi

python labels/gen_embedding.py --config "configs/text_embedding.${TAG}.json"
python labels/emb_preprocessing.py --config "configs/token_bank.${TAG}.json"
