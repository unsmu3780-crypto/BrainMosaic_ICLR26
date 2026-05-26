#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?usage: run_zuco_reconstruction_eval.sh ZuCoSR|ZuCoNR|ZuCoTSR}"
TAG="$(echo "${TASK}" | tr '[:upper:]' '[:lower:]')"
THRESHOLD_TAG="${THRESHOLD_TAG:-t0p78}"
RUN_DIR="${RUN_DIR:-outputs/${TAG}_full/${THRESHOLD_TAG}}"
RECON_CONFIG="${RECON_CONFIG:-configs/reconstruct.${TAG}.${THRESHOLD_TAG}.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/evaluate_reconstruction.${TAG}.${THRESHOLD_TAG}.json}"
SENT_EMB_PATH="${SENT_EMB_PATH:-}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-}"

if [[ -z "${SENT_EMB_PATH}" ]]; then
  echo "[ERR] SENT_EMB_PATH is required"
  exit 1
fi

if [[ -z "${EMBED_MODEL_PATH}" ]]; then
  echo "[ERR] EMBED_MODEL_PATH is required"
  exit 1
fi

mkdir -p "$(dirname "${RECON_CONFIG}")" "$(dirname "${EVAL_CONFIG}")" "${RUN_DIR}/reconstruction"

python - "${TASK}" "${TAG}" "${THRESHOLD_TAG}" "${RUN_DIR}" "${RECON_CONFIG}" "${EVAL_CONFIG}" "${SENT_EMB_PATH}" "${EMBED_MODEL_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

task, tag, threshold_tag, run_dir, recon_config, eval_config, sent_emb_path, embed_model_path = sys.argv[1:9]
run_dir = Path(run_dir)

llm_endpoint = os.environ["LLM_ENDPOINT"]
llm_model = os.environ["LLM_MODEL"]
llm_api_key = os.environ["LLM_API_KEY"]
llm_timeout = int(os.environ.get("LLM_TIMEOUT_SEC", "180"))
llm_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
llm_backoff = float(os.environ.get("LLM_RETRY_BACKOFF_SEC", "3"))
num_candidates = int(os.environ.get("NUM_CANDIDATES", "3"))
max_tokens = int(os.environ.get("MAX_TOKENS", "120"))
temperature = float(os.environ.get("TEMPERATURE", "0.7"))
flush_every = int(os.environ.get("RECON_FLUSH_EVERY", "20"))

is_zh = "chisco" in tag or "chineseeeg" in tag
language = "zh" if is_zh else "en"
bert_lang = "zh" if is_zh else "en"

recon = {
    "input": {
        "topk_json": str(run_dir / "eval_embeddings" / "best_by_matching_acc.topk.json")
    },
    "output": {
        "reconstruction_json": str(run_dir / "reconstruction" / "reconstructed_sentences.json"),
        "partial_json": str(run_dir / "reconstruction" / "reconstructed_sentences.partial.json"),
        "resume": True,
        "flush_every": flush_every,
        "include_prompt": False
    },
    "llm": {
        "endpoint": llm_endpoint,
        "model": llm_model,
        "api_key": llm_api_key,
        "timeout_sec": llm_timeout,
        "max_retries": llm_retries,
        "retry_backoff_sec": llm_backoff
    },
    "generation": {
        "num_candidates": num_candidates,
        "temperature": temperature,
        "max_tokens": max_tokens
    },
    "prompt": {
        "language": language,
        "hint_threshold": 0.7,
        "hint_thresholds": {
            "sentence_mode": 0.7,
            "subjectivity": 0.8,
            "semantic_focus": 0.7
        },
        "hint_labels": {
            "zh": {
                "sentence_mode": ["declarative", "interrogative", "negative", "imperative"],
                "subjectivity": ["objective", "subjective"],
                "semantic_focus": ["first-person", "second-person", "third-person", "thing-description", "event-action"]
            },
            "en": {
                "sentence_mode": ["declarative", "interrogative", "negative", "imperative"],
                "subjectivity": ["objective", "subjective"],
                "semantic_focus": ["first-person", "second-person", "third-person", "thing-description", "event-action"]
            }
        },
        "max_keywords": 12,
        "max_chars": 20
    }
}

evaluation = {
    "input": {
        "run_dir": str(run_dir),
        "reconstruction_json": str(run_dir / "reconstruction" / "reconstructed_sentences.json"),
        "sentence_embeddings_pt": sent_emb_path
    },
    "embedding_model": {
        "name_or_path": embed_model_path
    },
    "runtime": {
        "device": os.environ.get("EVAL_DEVICE", "cuda"),
        "batch_size": int(os.environ.get("EVAL_BATCH_SIZE", "32")),
        "max_length": int(os.environ.get("EVAL_MAX_LENGTH", "128")),
        "truncate_dim": int(os.environ.get("EVAL_TRUNCATE_DIM", "256"))
    },
    "metrics": {
        "expected_candidates": num_candidates,
        "bert_score_lang": bert_lang,
        "bert_score_model_type": os.environ.get("BERT_SCORE_MODEL_TYPE") or None
    },
    "output": {
        "summary_json": str(run_dir / "reconstruction" / "eval_summary.json"),
        "per_sample_json": str(run_dir / "reconstruction" / "eval_per_sample.json")
    }
}

Path(recon_config).write_text(json.dumps(recon, ensure_ascii=False, indent=2), encoding="utf-8")
Path(eval_config).write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
print(recon_config)
print(eval_config)
PY

python semantic_guided_decoder/sen_llm.py --config "${RECON_CONFIG}"
python scripts/evaluate_reconstruction.py --config "${EVAL_CONFIG}"

echo "[OK] reconstruction + evaluation finished for ${TASK} @ ${RUN_DIR}"
