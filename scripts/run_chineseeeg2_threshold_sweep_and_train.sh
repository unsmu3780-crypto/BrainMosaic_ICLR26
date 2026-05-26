#!/usr/bin/env bash
set -euo pipefail

DATASET_TAG="chineseeeg2"

THRESHOLDS="${THRESHOLDS:-0.78 0.85 0.90 0.95}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
FULL_EPOCHS="${FULL_EPOCHS:-50}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-16}"
FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-4}"
RUN_FULL="${RUN_FULL:-1}"
FULL_MODE="${FULL_MODE:-baseline_and_best}"
BASELINE_THRESHOLD="${BASELINE_THRESHOLD:-0.78}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-configs/train.${DATASET_TAG}.json}"
BASE_TOKEN_CONFIG="${BASE_TOKEN_CONFIG:-configs/token_bank.${DATASET_TAG}.json}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/${DATASET_TAG}_threshold_sweep}"
FULL_ROOT="${FULL_ROOT:-outputs/${DATASET_TAG}_full}"

mkdir -p "${SWEEP_ROOT}" "${FULL_ROOT}"

echo "[INFO] dataset=${DATASET_TAG}"
echo "[INFO] thresholds: ${THRESHOLDS}"
echo "[INFO] smoke epochs=${SMOKE_EPOCHS} batch_size=${SMOKE_BATCH_SIZE} num_workers=${SMOKE_NUM_WORKERS}"
echo "[INFO] full epochs=${FULL_EPOCHS} batch_size=${FULL_BATCH_SIZE} num_workers=${FULL_NUM_WORKERS} run_full=${RUN_FULL} full_mode=${FULL_MODE}"

if [[ ! -f "${BASE_TRAIN_CONFIG}" ]]; then
  echo "[ERROR] missing base train config: ${BASE_TRAIN_CONFIG}"
  exit 1
fi

if [[ ! -f "${BASE_TOKEN_CONFIG}" ]]; then
  echo "[ERROR] missing base token config: ${BASE_TOKEN_CONFIG}"
  exit 1
fi

for threshold in ${THRESHOLDS}; do
  tag="t${threshold//./p}"
  token_cfg="configs/token_bank.${DATASET_TAG}.${tag}.json"
  train_cfg="configs/train.${DATASET_TAG}.smoke.${tag}.json"
  log_file="${SWEEP_ROOT}/${tag}.log"

  echo "[SWEEP] threshold=${threshold} tag=${tag}"

  THRESHOLD="${threshold}" TAG="${tag}" DATASET_TAG="${DATASET_TAG}" BASE_TOKEN_CONFIG="${BASE_TOKEN_CONFIG}" \
  python - <<'PY'
import json
import os
from pathlib import Path

threshold = float(os.environ["THRESHOLD"])
tag = os.environ["TAG"]
dataset_tag = os.environ["DATASET_TAG"]
base_path = Path(os.environ["BASE_TOKEN_CONFIG"])
cfg = json.load(base_path.open("r", encoding="utf-8"))
text_root = Path(cfg["input"]["word_embeddings_pt"]).parent
cfg["cluster_sim_threshold"] = threshold
cfg["output"]["token_bank_dir"] = str(text_root / f"token_bank_{tag}")
out_path = Path("configs") / f"token_bank.{dataset_tag}.{tag}.json"
json.dump(cfg, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OUT] {out_path}")
print(f"[OUT] token_bank_dir={cfg['output']['token_bank_dir']}")
PY

  python labels/emb_preprocessing.py --config "${token_cfg}"

  THRESHOLD="${threshold}" TAG="${tag}" DATASET_TAG="${DATASET_TAG}" BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG}" \
  SMOKE_EPOCHS="${SMOKE_EPOCHS}" SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
  SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS}" SWEEP_ROOT="${SWEEP_ROOT}" \
  python - <<'PY'
import json
import os
from pathlib import Path

threshold = float(os.environ["THRESHOLD"])
tag = os.environ["TAG"]
dataset_tag = os.environ["DATASET_TAG"]
cfg = json.load(open(os.environ["BASE_TRAIN_CONFIG"], "r", encoding="utf-8"))
token_cfg = json.load(open(Path("configs") / f"token_bank.{dataset_tag}.{tag}.json", "r", encoding="utf-8"))
cfg["data"]["token_path"] = token_cfg["output"]["token_bank_dir"]
cfg["runtime"]["output_dir"] = str(Path(os.environ["SWEEP_ROOT"]) / tag)
cfg["runtime"]["batch_size"] = int(os.environ["SMOKE_BATCH_SIZE"])
cfg["runtime"]["num_workers"] = int(os.environ["SMOKE_NUM_WORKERS"])
cfg["train"]["epochs"] = int(os.environ["SMOKE_EPOCHS"])
cfg["train"]["lr_drop"] = max(int(os.environ["SMOKE_EPOCHS"]), 1)
cfg.setdefault("notes", {})
cfg["notes"]["dataset"] = dataset_tag
cfg["notes"]["token_bank_threshold"] = threshold
cfg["notes"]["run_type"] = "threshold_sweep_smoke"
out_path = Path("configs") / f"train.{dataset_tag}.smoke.{tag}.json"
json.dump(cfg, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OUT] {out_path}")
print(f"[OUT] output_dir={cfg['runtime']['output_dir']}")
PY

  python main.py --config "${train_cfg}" 2>&1 | tee "${log_file}"

  THRESHOLD="${threshold}" TAG="${tag}" DATASET_TAG="${DATASET_TAG}" SWEEP_ROOT="${SWEEP_ROOT}" \
  python - <<'PY'
import json
import os
from pathlib import Path

import torch

threshold = float(os.environ["THRESHOLD"])
tag = os.environ["TAG"]
dataset_tag = os.environ["DATASET_TAG"]
sweep_root = Path(os.environ["SWEEP_ROOT"])
token_cfg = json.load(open(Path("configs") / f"token_bank.{dataset_tag}.{tag}.json", "r", encoding="utf-8"))
train_cfg = json.load(open(Path("configs") / f"train.{dataset_tag}.smoke.{tag}.json", "r", encoding="utf-8"))
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
records.sort(
    key=lambda x: (
        x.get("best_acc") is not None,
        x.get("best_acc") or -1,
        x.get("mean_cosine") or -1,
    ),
    reverse=True,
)
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

FULL_MODE="${FULL_MODE}" BASELINE_THRESHOLD="${BASELINE_THRESHOLD}" SWEEP_ROOT="${SWEEP_ROOT}" \
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG}" FULL_ROOT="${FULL_ROOT}" FULL_BATCH_SIZE="${FULL_BATCH_SIZE}" \
FULL_NUM_WORKERS="${FULL_NUM_WORKERS}" FULL_EPOCHS="${FULL_EPOCHS}" DATASET_TAG="${DATASET_TAG}" python - <<'PY'
import json
import os
from pathlib import Path

records = json.load((Path(os.environ["SWEEP_ROOT"]) / "sweep_summary.json").open("r", encoding="utf-8"))
if not records:
    raise SystemExit("No sweep records available for full training")

chosen = os.environ.get("BEST_THRESHOLD", "").strip()
best = records[0]
if chosen:
    best = next((r for r in records if str(r["threshold"]) == chosen), None)
    if best is None:
        raise SystemExit(f"BEST_THRESHOLD={chosen} not found")

selected = []
full_mode = os.environ["FULL_MODE"]
baseline_threshold = os.environ["BASELINE_THRESHOLD"]
if full_mode == "best_only":
    selected = [best]
elif full_mode == "baseline_and_best":
    baseline = next((r for r in records if str(r["threshold"]) == baseline_threshold), None)
    if baseline is None:
        raise SystemExit(f"BASELINE_THRESHOLD={baseline_threshold} not found")
    selected = [baseline]
    if str(best["threshold"]) != str(baseline["threshold"]):
        selected.append(best)
else:
    raise SystemExit(f"Unsupported FULL_MODE={full_mode}")

paths = []
for record in selected:
    tag = record["tag"]
    dataset_tag = os.environ["DATASET_TAG"]
    base_cfg = json.load(open(os.environ["BASE_TRAIN_CONFIG"], "r", encoding="utf-8"))
    base_cfg["data"]["token_path"] = record["token_bank_dir"]
    base_cfg["runtime"]["output_dir"] = str(Path(os.environ["FULL_ROOT"]) / tag)
    base_cfg["runtime"]["batch_size"] = int(os.environ["FULL_BATCH_SIZE"])
    base_cfg["runtime"]["num_workers"] = int(os.environ["FULL_NUM_WORKERS"])
    base_cfg["train"]["epochs"] = int(os.environ["FULL_EPOCHS"])
    base_cfg["train"]["lr_drop"] = int(os.environ["FULL_EPOCHS"])
    base_cfg.setdefault("notes", {})
    base_cfg["notes"]["dataset"] = dataset_tag
    base_cfg["notes"]["token_bank_threshold"] = record["threshold"]
    base_cfg["notes"]["run_type"] = "full_training_after_threshold_sweep"
    base_cfg["notes"]["full_mode"] = full_mode
    base_cfg["notes"]["baseline_threshold"] = baseline_threshold
    base_cfg["notes"]["selected_by"] = "baseline" if str(record["threshold"]) == baseline_threshold else "best_sweep"
    out_path = Path("configs") / f"train.{dataset_tag}.full.{tag}.json"
    json.dump(base_cfg, out_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    paths.append(str(out_path))

Path(os.environ["FULL_ROOT"]).mkdir(parents=True, exist_ok=True)
with open(Path(os.environ["FULL_ROOT"]) / "selected_full_configs.txt", "w", encoding="utf-8") as f:
    for p in paths:
        f.write(p + "\n")
print("\n".join(paths))
PY

while IFS= read -r full_cfg; do
  [[ -n "${full_cfg}" ]] || continue
  full_tag="$(basename "${full_cfg}" .json)"
  full_tag="${full_tag#train.${DATASET_TAG}.full.}"
  full_log="${FULL_ROOT}/${full_tag}.log"
  echo "[FULL] config=${full_cfg}"
  python main.py --config "${full_cfg}" 2>&1 | tee "${full_log}"
done < "${FULL_ROOT}/selected_full_configs.txt"
