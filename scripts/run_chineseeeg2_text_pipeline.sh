#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-raw}"
EXPANSION_MODEL="${EXPANSION_MODEL:-gpt-4o-mini}"
CHINESEEEG2_EEG_ROOT="${CHINESEEEG2_EEG_ROOT:-/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/ChineseEEG2}"
TEXT_ROOT="${TEXT_ROOT:-${CHINESEEEG2_EEG_ROOT}/text_assets}"

if [[ "${BACKEND}" == "raw" ]]; then
  python scripts/build_chineseeeg2_text_assets_inputs.py
else
  python scripts/build_chineseeeg2_text_assets_inputs.py --use-expanded-tokens
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
fi

python labels/gen_embedding.py --config configs/text_embedding.chineseeeg2.json
python labels/emb_preprocessing.py --config configs/token_bank.chineseeeg2.json

python - <<'PY'
import json
from pathlib import Path

import torch

cfg = json.load(open("configs/text_embedding.chineseeeg2.json", "r", encoding="utf-8"))
root = Path(cfg["output_dir"])
s = torch.load(root / "sentence_embeddings.pt", map_location="cpu")
w = torch.load(root / "word_embeddings.pt", map_location="cpu")
tb = torch.load(root / "token_bank" / "embeddings.pt", map_location="cpu")
print("[CHECK] sentences:", len(s["sentences"]), tuple(s["embeddings"].shape))
print("[CHECK] words:", len(w["keys"]), tuple(w["embeddings"].shape))
print("[CHECK] token_bank:", tuple(tb["embeddings"].shape))
print("[CHECK] sample word texts:", w["texts"][:3])
PY
