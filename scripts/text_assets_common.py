import csv
import json
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path


DEFAULT_MODEL_PATH = os.environ.get(
    "BRAINMOSAIC_EMBEDDING_MODEL",
    "/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/models/Qwen3-Embedding-8B",
)


def load_records(path):
    import torch

    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain list[dict] or {{'samples': list[dict]}}")
    return data


def clean_text(value):
    return str(value).strip()


def as_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dataset_tag(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def collect_assets(eeg_root):
    split_files = {
        "train": eeg_root / "train.pt",
        "val": eeg_root / "val.pt",
    }
    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split: {path}")

    sentence_counts = Counter()
    token_counts = Counter()
    segmentation = OrderedDict()
    split_sizes = {}

    for split, path in split_files.items():
        records = load_records(path)
        split_sizes[split] = len(records)
        for rec in records:
            if not isinstance(rec, dict):
                continue

            sentence = clean_text(rec.get("sentence", ""))
            if not sentence:
                continue

            words = [clean_text(w) for w in (rec.get("words") or []) if clean_text(w)]
            if not words:
                continue

            sentence_counts[sentence] += 1
            token_counts.update(words)

            if sentence not in segmentation:
                segmentation[sentence] = {
                    "sentence": sentence,
                    "tokens": words,
                    "sentence_mode": as_int(rec.get("sentence_mode", rec.get("te", 0))),
                    "subjectivity": as_int(rec.get("subjectivity", rec.get("oors", 0))),
                    "semantic_focus": as_int(rec.get("semantic_focus", rec.get("su", 0))),
                    "count": 0,
                }
            segmentation[sentence]["count"] += 1

    return sentence_counts, token_counts, list(segmentation.values()), split_sizes


def write_sentences_csv(path, sentence_counts):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "count"])
        writer.writeheader()
        for sentence, count in sentence_counts.most_common():
            writer.writerow({"sentence": sentence, "count": count})


def write_token_json(path, token_counts):
    rows = [{"key": token, "explanation": token, "count": count} for token, count in token_counts.most_common()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_segmentation_json(path, segmentation):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segmentation, f, ensure_ascii=False, indent=2)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_text_embedding_config(text_root, model_name, device, batch_size, truncate_dim, use_expanded_tokens):
    tokens_file = text_root / ("token_explanations.expanded.json" if use_expanded_tokens else "token_explanations.json")
    return {
        "model": {"name_or_path": model_name},
        "runtime": {
            "device": device,
            "batch_size": batch_size,
            "max_length": 128,
            "truncate_dim": truncate_dim,
        },
        "inputs": {
            "sentences_file": str(text_root / "sentences.csv"),
            "sentence_col": "sentence",
            "tokens_file": str(tokens_file),
            "token_key_col": "key",
            "token_explanation_col": "explanation",
            "use_expansion_text": True,
        },
        "output_dir": str(text_root),
    }


def build_token_bank_config(text_root, truncate_dim, cluster_sim_threshold):
    return {
        "input": {"word_embeddings_pt": str(text_root / "word_embeddings.pt")},
        "output": {"token_bank_dir": str(text_root / "token_bank")},
        "truncate_dim": truncate_dim,
        "cluster_sim_threshold": cluster_sim_threshold,
    }


def build_train_config(eeg_root, text_root, output_dir, device, in_channels):
    return {
        "data": {
            "in_channels": in_channels,
            "eeg_split_pattern": "{split}.pt",
            "eeg_scale": 1000000.0,
            "normalize_token_emb": True,
            "token_path": str(text_root / "token_bank"),
            "sent_emb_path": str(text_root / "sentence_embeddings.pt"),
            "segmentation_path": str(text_root / "segmentation.json"),
            "eeg_path": str(eeg_root),
        },
        "runtime": {
            "output_dir": str(output_dir),
            "device": device,
            "seed": 42,
            "num_workers": 4,
            "batch_size": 32,
            "resume": "",
            "eval": False,
            "world_size": 1,
            "dist_url": "env://",
        },
        "train": {
            "lr": 0.0001,
            "lr_backbone": 0.00001,
            "weight_decay": 0.0001,
            "epochs": 50,
            "lr_drop": 50,
            "clip_max_norm": 0.1,
        },
        "model": {
            "encoder": "moderntcn",
            "tcn_blocks_per_stage": [2],
            "tcn_large_kernel_per_stage": [25],
            "tcn_small_kernel_per_stage": [5],
            "tcn_ffn_ratio": 2.0,
            "tcn_downsample_ratio": 1,
            "tcn_stem_dim": in_channels,
            "tcn_size": 64,
            "tcn_use_revin": False,
            "tcn_dropout": 0.0,
            "enc_layers": 3,
            "dec_layers": 6,
            "hidden_dim": 256,
            "dropout": 0.1,
            "num_queries": 8,
            "slot_dropout_p": 0.2,
        },
        "retrieval": {"top_k": 5, "exist_threshold": 0.7, "cos_threshold": 0.7},
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
            "semantic_focus_class_counts": None,
        },
    }


def build_dataset_assets(
    eeg_root,
    text_root,
    config_dir,
    output_dir,
    dataset_name,
    model_name,
    device,
    batch_size,
    truncate_dim,
    cluster_sim_threshold,
    in_channels,
    raw_token_config,
):
    sentence_counts, token_counts, segmentation, split_sizes = collect_assets(eeg_root)

    text_root.mkdir(parents=True, exist_ok=True)
    write_sentences_csv(text_root / "sentences.csv", sentence_counts)
    write_token_json(text_root / "token_explanations.json", token_counts)
    write_segmentation_json(text_root / "segmentation.json", segmentation)

    tag = dataset_tag(dataset_name)
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config_dir / f"text_embedding.{tag}.json",
        build_text_embedding_config(
            text_root=text_root,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            truncate_dim=truncate_dim,
            use_expanded_tokens=not raw_token_config,
        ),
    )
    write_json(
        config_dir / f"token_bank.{tag}.json",
        build_token_bank_config(
            text_root=text_root,
            truncate_dim=truncate_dim,
            cluster_sim_threshold=cluster_sim_threshold,
        ),
    )
    write_json(
        config_dir / f"train.{tag}.json",
        build_train_config(
            eeg_root=eeg_root,
            text_root=text_root,
            output_dir=output_dir,
            device=device,
            in_channels=in_channels,
        ),
    )
    return sentence_counts, token_counts, split_sizes, tag
