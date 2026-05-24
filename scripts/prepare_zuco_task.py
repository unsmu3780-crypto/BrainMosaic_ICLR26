from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


RAW_ROOT = Path(
    os.environ.get(
        "ZUCO_RAW_ROOT",
        "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/Zuco-1",
    )
)
OUT_BASE = Path(
    os.environ.get(
        "ZUCO_OUT_BASE",
        "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data",
    )
)
SAVE_DTYPE = torch.float16
SEED = 42
VAL_RATIO = 0.2

TASKS = {
    "ZuCoSR": {
        "task_dir": "task1- SR",
        "sentence_csv": ("Raw data", "sentiment_normal_reading.csv"),
        "label_field": "control",
        "label_kind": "sr_control",
        "sentence_mode": 0,
        "subjectivity": 0,
        "semantic_focus": 3,
    },
    "ZuCoNR": {
        "task_dir": "task2 - NR",
        "sentence_csv": ("Raw data", "relations_normal_reading.csv"),
        "label_field": "control",
        "label_kind": "nr_control",
        "sentence_mode": 0,
        "subjectivity": 0,
        "semantic_focus": 3,
    },
    "ZuCoTSR": {
        "task_dir": "task3 - TSR",
        "sentence_csv": ("Preprocessed", "relations_task_specific.csv"),
        "label_field": "relation_type",
        "label_kind": "tsr_relation_type",
        "sentence_mode": 0,
        "subjectivity": 0,
        "semantic_focus": 3,
    },
}

EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "there",
    "they", "this", "to", "was", "were", "will", "with", "you", "your",
}

EN_CLOSED_CLASS = {
    "AUX", "CCONJ", "ADP", "CC", "IN", "TO", "DT", "PDT", "WDT", "PRP", "PRP$", "WP", "WP$", "MD",
}

TOKEN_FIXES = {
    "emp11111ty": "empty",
}


def parse_args():
    parser = argparse.ArgumentParser("Prepare a single ZuCo task into unified BrainMosaic splits")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-dtype", default="float16", choices=["float16", "float32"])
    return parser.parse_args()


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip()).replace("\uFFFD", "'")


def normalize_token(token):
    token = normalize_text(token)
    token = token.replace("`", "'")
    token = re.sub(r"[“”]", '"', token)
    token = re.sub(r"[‘’]", "'", token)
    token = token.strip("()[]{}\"'")
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
    token = TOKEN_FIXES.get(token, token)
    token = re.sub(r"([A-Za-z])\d{2,}([A-Za-z])", r"\1\2", token)
    token = re.sub(r"\d{4,}", "", token) if not re.search(r"[A-Za-z]", token) else token
    return token.strip()


def is_noise_token(token):
    if not token:
        return True
    if re.fullmatch(r"[\W_]+", token):
        return True
    if not re.search(r"[A-Za-z0-9]", token):
        return True
    if re.fullmatch(r"[A-Za-z]?", token):
        return True
    if re.search(r"\d{5,}", token):
        return True
    return False


def load_semicolon_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            clean = {str(k).strip(): normalize_text(v) for k, v in row.items() if k is not None}
            if any(clean.values()):
                rows.append(clean)
    return rows


def build_sentence_label_lookup(task_name, task_cfg, raw_root):
    csv_path = raw_root / task_cfg["task_dir"] / task_cfg["sentence_csv"][0] / task_cfg["sentence_csv"][1]
    rows = load_semicolon_csv(csv_path)
    lookup = {}
    collisions = Counter()
    for row in rows:
        sentence = normalize_text(row.get("sentence", ""))
        if not sentence:
            continue
        label = normalize_text(row.get(task_cfg["label_field"], ""))
        key = sentence.lower()
        if key in lookup and lookup[key] != label:
            collisions[key] += 1
        lookup[key] = label
    return lookup, csv_path, collisions


def maybe_pos_tag(words):
    try:
        import nltk

        try:
            tagged = nltk.pos_tag(words)
        except LookupError:
            return None
        return [(w, tag) for w, tag in tagged]
    except Exception:
        return None


def select_words(word_entries):
    raw_words = []
    token_stats = Counter()
    for item in word_entries or []:
        if isinstance(item, dict):
            token = normalize_text(item.get("content", ""))
        else:
            token = normalize_text(item)
        if not token:
            token_stats["empty_raw"] += 1
            continue
        token = normalize_token(token)
        if is_noise_token(token):
            token_stats["noise_filtered"] += 1
            continue
        raw_words.append(token)

    if not raw_words:
        return [], {"mode": "empty", "stats": dict(token_stats)}

    tagged = maybe_pos_tag(raw_words)
    if tagged:
        kept = [w for w, tag in tagged if tag not in EN_CLOSED_CLASS and w.lower() not in EN_STOPWORDS]
        if kept:
            token_stats["closed_class_filtered"] = len(raw_words) - len(kept)
            return kept[:8], {"mode": "nltk_pos", "num_input": len(raw_words), "num_kept": len(kept), "stats": dict(token_stats)}

    kept = [w for w in raw_words if w.lower() not in EN_STOPWORDS and re.search(r"[A-Za-z0-9]", w)]
    token_stats["stopword_filtered"] = len(raw_words) - len(kept)
    deduped = []
    seen = set()
    for w in kept:
        key = w.lower()
        if key in seen:
            token_stats["dedup_filtered"] += 1
            continue
        seen.add(key)
        deduped.append(w)
    return deduped[:8], {"mode": "fallback_stopwords", "num_input": len(raw_words), "num_kept": len(deduped), "stats": dict(token_stats)}


def stable_split_indices(length, val_ratio, seed, subject, task):
    import random

    idx = list(range(length))
    rnd = random.Random(f"{seed}:{task}:{subject}")
    rnd.shuffle(idx)
    n_val = max(1, int(length * val_ratio))
    val_idx = set(idx[:n_val])
    train_idx = [i for i in range(length) if i not in val_idx]
    val_idx_sorted = [i for i in range(length) if i in val_idx]
    return train_idx, val_idx_sorted


def extract_subject(mat_path):
    m = re.search(r"results([A-Z0-9]+)_(SR|NR|TSR)\.mat$", mat_path.name)
    if not m:
        raise ValueError(f"Cannot parse subject from {mat_path}")
    return m.group(1)


def load_sentence_data(mat_path):
    data = sio.loadmat(str(mat_path), simplify_cells=True)
    rows = data.get("sentenceData", [])
    if not isinstance(rows, list):
        rows = [rows]
    return rows


def build_record(task_name, task_cfg, subject, trial_idx, row, sentence_lookup, save_dtype):
    sentence = normalize_text(row.get("content", ""))
    if not sentence:
        return None, {"skip": "empty_sentence"}

    eeg = row.get("rawData", None)
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2:
        return None, {"skip": "invalid_eeg"}

    eeg_tensor = torch.as_tensor(eeg, dtype=getattr(torch, save_dtype))
    words, token_meta = select_words(row.get("word", []))
    if not words:
        return None, {"skip": "empty_words", "token_meta": token_meta}

    label = sentence_lookup.get(sentence.lower(), "")
    record = {
        "eeg": eeg_tensor,
        "sentence": sentence,
        "words": words,
        "label": label,
        "sentence_mode": task_cfg["sentence_mode"],
        "subjectivity": task_cfg["subjectivity"],
        "semantic_focus": task_cfg["semantic_focus"],
        "source": {
            "dataset": task_name,
            "subject": subject,
            "trial": trial_idx,
            "mat_file": str(row.get("_mat_file", "")),
            "task_dir": task_cfg["task_dir"],
            "label_kind": task_cfg["label_kind"],
            "token_filter": token_meta,
            "raw_shape": list(eeg.shape),
        },
    }
    return record, None


def main():
    args = parse_args()
    task_cfg = TASKS[args.task]
    raw_root = args.raw_root.resolve()
    out_root = (args.out_root or (OUT_BASE / args.task)).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sentence_lookup, csv_path, collisions = build_sentence_label_lookup(args.task, task_cfg, raw_root)
    mat_dir = raw_root / task_cfg["task_dir"] / "Matlab files"
    mat_files = sorted(mat_dir.glob("results*_" + args.task.replace("ZuCo", "") + ".mat"))
    if not mat_files:
        suffix = args.task.replace("ZuCo", "")
        mat_files = sorted(mat_dir.glob(f"results*_{suffix}.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No mat files found under {mat_dir}")

    train_records = []
    val_records = []
    subject_stats = []
    skips = Counter()

    for mat_path in mat_files:
        subject = extract_subject(mat_path)
        rows = load_sentence_data(mat_path)
        for row in rows:
            if isinstance(row, dict):
                row["_mat_file"] = str(mat_path)
        train_idx, val_idx = stable_split_indices(len(rows), args.val_ratio, args.seed, subject, args.task)
        subject_train = 0
        subject_val = 0
        for split_name, split_idx in (("train", train_idx), ("val", val_idx)):
            target = train_records if split_name == "train" else val_records
            for idx in split_idx:
                record, err = build_record(
                    args.task,
                    task_cfg,
                    subject,
                    idx,
                    rows[idx],
                    sentence_lookup,
                    args.save_dtype,
                )
                if record is None:
                    skips[err["skip"]] += 1
                    continue
                target.append(record)
                if split_name == "train":
                    subject_train += 1
                else:
                    subject_val += 1
        subject_stats.append(
            {
                "subject": subject,
                "mat_file": str(mat_path),
                "raw_trials": len(rows),
                "train_records": subject_train,
                "val_records": subject_val,
            }
        )

    if not train_records or not val_records:
        raise RuntimeError(f"{args.task} produced empty split(s)")

    torch.save(train_records, out_root / "train.pt")
    torch.save(val_records, out_root / "val.pt")
    meta = {
        "dataset": args.task,
        "raw_root": str(raw_root),
        "task_dir": task_cfg["task_dir"],
        "sentence_csv": str(csv_path),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "subjects": subject_stats,
        "label_field": task_cfg["label_field"],
        "label_collisions": sum(collisions.values()),
        "split_policy": {
            "mode": "in_subject_deterministic_shuffle",
            "seed": args.seed,
            "val_ratio": args.val_ratio,
        },
        "save_dtype": args.save_dtype,
        "skips": dict(skips),
        "notes": "Unified public reproduction unit aligned to paper task naming.",
    }
    torch.save(meta, out_root / "meta.pt")
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {args.task}")
    print(f"  mat files: {len(mat_files)}")
    print(f"  train: {len(train_records)}")
    print(f"  val: {len(val_records)}")
    print(f"  out: {out_root}")
    print(f"  sentence csv: {csv_path}")
    print(f"  skips: {dict(skips)}")


if __name__ == "__main__":
    main()
