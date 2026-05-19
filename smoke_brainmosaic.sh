#!/bin/bash
set -e

REPO_DIR="/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26"
cd "$REPO_DIR"

echo "== Environment check =="
python - <<'PY'
import torch, numpy
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("numpy:", numpy.__version__)
PY

echo "== Create synthetic smoke-test assets =="
python - <<'PY'
import json
from pathlib import Path
import torch
import torch.nn.functional as F

root = Path("smoke_assets")
data_dir = root / "eeg"
token_dir = root / "token_bank"
out_dir = root / "outputs"
cfg_dir = root / "configs"

for p in [data_dir, token_dir, out_dir, cfg_dir]:
    p.mkdir(parents=True, exist_ok=True)

hidden_dim = 256
in_channels = 122
time_len = 250

sentences = [
    "I eat apples every day.",
    "She likes orange juice.",
    "I am ready to sleep.",
    "Today we go library.",
]

tokens = [
    "I", "eat", "apples", "daily",
    "she", "likes", "orange", "juice",
    "ready", "sleep", "today", "we", "go", "library"
]

torch.manual_seed(42)

# token bank
token_emb = F.normalize(torch.randn(len(tokens), hidden_dim), dim=-1)
torch.save({"embeddings": token_emb}, token_dir / "embeddings.pt")

word2cid = {w: i for i, w in enumerate(tokens)}
with open(token_dir / "map.json", "w", encoding="utf-8") as f:
    json.dump(word2cid, f, ensure_ascii=False, indent=2)

info = [
    {"cluster2_id": i, "size": 1, "members": [w]}
    for i, w in enumerate(tokens)
]
with open(token_dir / "info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

# sentence embeddings
sent_emb = F.normalize(torch.randn(len(sentences), hidden_dim), dim=-1)
torch.save(
    {"sentences": sentences, "embeddings": sent_emb},
    root / "sentence_embeddings.pt"
)

word_sets = [
    ["I", "eat", "apples", "daily"],
    ["she", "likes", "orange", "juice"],
    ["I", "ready", "sleep"],
    ["today", "we", "go", "library"],
]

def make_records(n):
    records = []
    for i in range(n):
        j = i % len(sentences)
        records.append({
            "eeg": torch.randn(in_channels, time_len),
            "sentence": sentences[j],
            "words": word_sets[j],
            "sentence_mode": 0,
            "subjectivity": 0,
            "semantic_focus": 0,
        })
    return records

torch.save(make_records(4), data_dir / "train.pt")
torch.save(make_records(2), data_dir / "val.pt")

config = {
    "data": {
        "in_channels": in_channels,
        "eeg_split_pattern": "{split}.pt",
        "eeg_scale": 1.0,
        "normalize_token_emb": True,
        "token_path": str(token_dir),
        "sent_emb_path": str(root / "sentence_embeddings.pt"),
        "segmentation_path": "",
        "eeg_path": str(data_dir)
    },
    "runtime": {
        "output_dir": str(out_dir),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "num_workers": 0,
        "batch_size": 2,
        "resume": "",
        "eval": False,
        "world_size": 1,
        "dist_url": "env://"
    },
    "train": {
        "lr": 0.0001,
        "lr_backbone": 0.00001,
        "weight_decay": 0.0001,
        "epochs": 1,
        "lr_drop": 1,
        "clip_max_norm": 0.1
    },
    "model": {
        "encoder": "moderntcn",
        "tcn_blocks_per_stage": [1],
        "tcn_large_kernel_per_stage": [5],
        "tcn_small_kernel_per_stage": [3],
        "tcn_ffn_ratio": 2.0,
        "tcn_downsample_ratio": 1,
        "tcn_stem_dim": in_channels,
        "tcn_size": 64,
        "tcn_use_revin": False,
        "tcn_dropout": 0.0,
        "enc_layers": 1,
        "dec_layers": 1,
        "hidden_dim": hidden_dim,
        "dropout": 0.1,
        "num_queries": 8,
        "slot_dropout_p": 0.0
    },
    "retrieval": {
        "top_k": 3,
        "exist_threshold": 0.1,
        "cos_threshold": -1.0
    },
    "loss": {
        "embed_loss": "both",
        "tau": 0.07,
        "lambda_infonce": 0.2,
        "lambda_cos": 1.0,
        "lambda_sent": 0.2,
        "lambda_cls": 1.0,
        "eos_coef": 0.3,
        "cost_class": 1.0,
        "cost_emb": 2.0,
        "lambda_sentence_mode": 0.2,
        "lambda_subjectivity": 0.2,
        "lambda_semantic_focus": 0.2,
        "sentence_mode_class_counts": None,
        "subjectivity_class_counts": None,
        "semantic_focus_class_counts": None
    }
}

with open(cfg_dir / "smoke_train.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("Created:", cfg_dir / "smoke_train.json")
PY

echo "== Run BrainMosaic smoke training =="
python main.py --config smoke_assets/configs/smoke_train.json

echo "== Check output files =="
ls -lh smoke_assets/outputs
find smoke_assets/outputs -maxdepth 3 -type f | sort

echo "== Smoke test finished successfully =="

