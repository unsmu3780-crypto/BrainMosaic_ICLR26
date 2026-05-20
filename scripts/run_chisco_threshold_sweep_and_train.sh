#!/usr/bin/env bash
set -euo pipefail

THRESHOLDS="${THRESHOLDS:-0.78 0.85 0.90 0.95}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
FULL_EPOCHS="${FULL_EPOCHS:-50}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-4}"
RUN_FULL="${RUN_FULL:-1}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-configs/train.chisco.json}"
BASE_TOKEN_CONFIG="${BASE_TOKEN_CONFIG:-configs/token_bank.chisco.json}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/chisco_threshold_sweep}"
FULL_ROOT="${FULL_ROOT:-outputs/chisco_full}"

mkdir -p "${SWEEP_ROOT}" "${FULL_ROOT}"

echo "[INFO] thresholds: ${THRESHOLDS}"
echo "[INFO] smoke epochs=${SMOKE_EPOCHS} batch_size=${SMOKE_BATCH_SIZE} num_workers=${SMOKE_NUM_WORKERS}"
echo "[INFO] full epochs=${FULL_EPOCHS} batch_size=${FULL_BATCH_SIZE} num_workers=${FULL_NUM_WORKERS} run_full=${RUN_FULL}"

for threshold in ${THRESHOLDS}; do
  tag="t${threshold//./p}"
  token_cfg="configs/token_bank.chisco.${tag}.json"
  train_cfg="configs/train.chisco.smoke.${tag}.json"
  log_file="${SWEEP_ROOT}/${tag}.log"

  echo "[SWEEP] threshold=${threshold} tag=${tag}"

  THRESHOLD="${threshold}" TAG="${tag}" BASE_TOKEN_CONFIG="${BASE_TOKEN_CONFIG}" \
  python - <<'PY'
import json
import os
from pathlib import Path

threshold = float(os.environ["THRESHOLD"])
tag = os.environ["TAG"]
base_path = Path(os.environ["BASE_TOKEN_CONFIG"])
cfg = json.load(base_path.open("r", encoding="utf-8"))
text_root = Path(cfg["input"]["word_embeddings_pt"]).parent
cfg["cluster_sim_threshold"] = threshold
cfg["output"]["token_bank_dir"] = str(text_root / f"token_bank_{tag}")
out_path = Path("configs") / f"token_bank.chisco.{tag}.json"
json.dump(cfg, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OUT] {out_path}")
print(f"[OUT] token_bank_dir={cfg['output']['token_bank_dir']}")
PY

  python labels/emb_preprocessing.py --config "${token_cfg}"

  THRESHOLD="${threshold}" TAG="${tag}" BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG}" \
  SMOKE_EPOCHS="${SMOKE_EPOCHS}" SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
  SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS}" SWEEP_ROOT="${SWEEP_ROOT}" \
  python - <<'PY'
import json
import os
from pathlib import Path

threshold = float(os.environ["THRESHOLD"])
tag = os.environ["TAG"]
base_path = Path(os.environ["BASE_TRAIN_CONFIG"])
cfg = json.load(base_path.open("r", encoding="utf-8"))
token_cfg = json.load(open(Path("configs") / f"token_bank.chisco.{tag}.json", "r", encoding="utf-8"))
cfg["data"]["token_path"] = token_cfg["output"]["token_bank_dir"]
cfg["runtime"]["output_dir"] = str(Path(os.environ["SWEEP_ROOT"]) / tag)
cfg["runtime"]["batch_size"] = int(os.environ["SMOKE_BATCH_SIZE"])
cfg["runtime"]["num_workers"] = int(os.environ["SMOKE_NUM_WORKERS"])
cfg["train"]["epochs"] = int(os.environ["SMOKE_EPOCHS"])
cfg["train"]["lr_drop"] = max(int(os.environ["SMOKE_EPOCHS"]), 1)
cfg.setdefault("notes", {})
cfg["notes"]["token_bank_threshold"] = threshold
cfg["notes"]["run_type"] = "threshold_sweep_smoke"
out_path = Path("configs") / f"train.chisco.smoke.{tag}.json"
json.dump(cfg, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OUT] {out_path}")
print(f"[OUT] output_dir={cfg['runtime']['output_dir']}")
PY

  python main.py --config "${train_cfg}" 2>&1 | tee "${log_file}"

  THRESHOLD="${threshold}" TAG="${tag}" SWEEP_ROOT="${SWEEP_ROOT}" \
  python - <<'PY'
import json
import os
from pathlib import Path

import torch

threshold = float(os.environ["THRESHOLD"])
tag = os.environ["TAG"]
sweep_root = Path(os.environ["SWEEP_ROOT"])
token_cfg = json.load(open(Path("configs") / f"token_bank.chisco.{tag}.json", "r", encoding="utf-8"))
train_cfg = json.load(open(Path("configs") / f"train.chisco.smoke.{tag}.json", "r", encoding="utf-8"))
emb = torch.load(Path(token_cfg["output"]["token_bank_dir"]) / "embeddings.pt", map_location="cpu")["embeddings"]
best_path = Path(train_cfg["runtime"]["output_dir"]) / "best_summary.json"
hist_path = Path(train_cfg["runtime"]["output_dir"]) / "epoch_history.json"
best = json.load(best_path.open("r", encoding="utf-8")) if best_path.exists() else {}
history = json.load(hist_path.open("r", encoding="utf-8")) if hist_path.exists() else []
last_val = history[-1].get("val", {}) if history else {}
record = {
    "threshold": threshold,
    "tag": tag,
    "clusters": int(emb.shape[0]),
    "embedding_dim": int(emb.shape[1]),
    "best_acc": best.get("best_acc"),
    "best_epoch": best.get("best_epoch"),
    "mean_cosine": last_val.get("mean_cosine"),
    "output_dir": train_cfg["runtime"]["output_dir"],
    "token_bank_dir": token_cfg["output"]["token_bank_dir"],
}
out_path = sweep_root / f"{tag}.summary.json"
json.dump(record, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[SUMMARY] {json.dumps(record, ensure_ascii=False)}")
PY
done

SWEEP_ROOT="${SWEEP_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SWEEP_ROOT"])
records = []
for path in sorted(root.glob("t*.summary.json")):
    records.append(json.load(path.open("r", encoding="utf-8")))
records.sort(key=lambda x: (x.get("best_acc") is not None, x.get("best_acc") or -1, x.get("mean_cosine") or -1), reverse=True)
json.dump(records, (root / "sweep_summary.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
with (root / "sweep_summary.tsv").open("w", encoding="utf-8") as f:
    f.write("threshold\ttag\tclusters\tbest_acc\tmean_cosine\toutput_dir\ttoken_bank_dir\n")
    for r in records:
        f.write(
            f"{r['threshold']}\t{r['tag']}\t{r['clusters']}\t{r.get('best_acc')}\t"
            f"{r.get('mean_cosine')}\t{r['output_dir']}\t{r['token_bank_dir']}\n"
        )
if records:
    print(f"[BEST] {records[0]}")
else:
    print("[WARN] no sweep records found")
PY

if [[ "${RUN_FULL}" != "1" ]]; then
  echo "[DONE] threshold sweep complete; RUN_FULL=${RUN_FULL}, skipping full training"
  exit 0
fi

python - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ.get("SWEEP_ROOT", "outputs/chisco_threshold_sweep")) / "sweep_summary.json"
records = json.load(summary_path.open("r", encoding="utf-8"))
if not records:
    raise SystemExit("No sweep records available for full training")

chosen = os.environ.get("BEST_THRESHOLD", "").strip()
if chosen:
    selected = next((r for r in records if str(r["threshold"]) == chosen), None)
    if selected is None:
        raise SystemExit(f"BEST_THRESHOLD={chosen} not found in sweep records")
else:
    selected = records[0]

tag = selected["tag"]
base_cfg = json.load(open(os.environ.get("BASE_TRAIN_CONFIG", "configs/train.chisco.json"), "r", encoding="utf-8"))
base_cfg["data"]["token_path"] = selected["token_bank_dir"]
base_cfg["runtime"]["output_dir"] = str(Path(os.environ.get("FULL_ROOT", "outputs/chisco_full")) / tag)
base_cfg["runtime"]["batch_size"] = int(os.environ.get("FULL_BATCH_SIZE", "32"))
base_cfg["runtime"]["num_workers"] = int(os.environ.get("FULL_NUM_WORKERS", "4"))
base_cfg["train"]["epochs"] = int(os.environ.get("FULL_EPOCHS", "50"))
base_cfg["train"]["lr_drop"] = int(os.environ.get("FULL_EPOCHS", "50"))
base_cfg.setdefault("notes", {})
base_cfg["notes"]["token_bank_threshold"] = selected["threshold"]
base_cfg["notes"]["run_type"] = "full_training_after_threshold_sweep"
out_path = Path("configs") / f"train.chisco.full.{tag}.json"
json.dump(base_cfg, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(out_path)
PY

best_tag="$(python - <<'PY'
import json
import os
from pathlib import Path

records = json.load(open(Path(os.environ.get("SWEEP_ROOT", "outputs/chisco_threshold_sweep")) / "sweep_summary.json", "r", encoding="utf-8"))
chosen = os.environ.get("BEST_THRESHOLD", "").strip()
if chosen:
    selected = next(r for r in records if str(r["threshold"]) == chosen)
else:
    selected = records[0]
print(selected["tag"])
PY
)"

full_cfg="configs/train.chisco.full.${best_tag}.json"
full_log="${FULL_ROOT}/${best_tag}.log"
mkdir -p "${FULL_ROOT}"
echo "[FULL] config=${full_cfg}"
python main.py --config "${full_cfg}" 2>&1 | tee "${full_log}"
