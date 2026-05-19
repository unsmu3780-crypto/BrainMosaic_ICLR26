from pathlib import Path
import re
import random

import jieba
import mne
import pandas as pd
import torch

RAW_ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/Chiso")
OUT_ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco")

WINDOW_SEC = 4.0
TARGET_SFREQ = 250
SEED = 42
VAL_RATIO = 0.2
SAVE_DTYPE = torch.float16

def split_words(sentence):
    words = []
    for w in jieba.lcut(str(sentence)):
        w = str(w).strip()
        if not w:
            continue
        if re.fullmatch(r"[\s，。！？、,.!?;；:：]+", w):
            continue
        words.append(w)
    return words[:8]

def sentence_mode(sentence):
    s = str(sentence)
    if "?" in s or "？" in s or "吗" in s or "呢" in s:
        return 1
    if "别" in s or "不要" in s or "不" in s or "没" in s:
        return 2
    return 0

def subjectivity(sentence):
    s = str(sentence)
    markers = ["我觉得", "喜欢", "讨厌", "害怕", "担心", "希望", "想", "怕", "爱", "开心", "难过"]
    return 1 if any(x in s for x in markers) else 0

def semantic_focus(sentence):
    s = str(sentence)
    if any(x in s for x in ["我", "我们", "咱们"]):
        return 0
    if any(x in s for x in ["你", "您"]):
        return 1
    if any(x in s for x in ["他", "她", "他们", "她们"]):
        return 2
    if any(x in s for x in ["做", "去", "来", "走", "跑", "吃", "喝", "订", "买", "工作", "旅行"]):
        return 4
    return 3

def parse_run_id(path):
    m = re.search(r"run-0*(\d+)_eeg\.edf$", path.name)
    return int(m.group(1)) if m else None

def parse_subject(path):
    m = re.search(r"(sub-\d+)", str(path))
    return m.group(1) if m else "unknown"

def load_sentences(run_id):
    p = RAW_ROOT / "textdataset" / f"split_data_{run_id}.xlsx"
    if not p.exists():
        return None
    df = pd.read_excel(p)
    if "句子" not in df.columns:
        raise ValueError(f"{p} missing column: 句子")
    return df

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(RAW_ROOT.rglob("*.edf"))
    print("edf files:", len(edf_files))

    records = []
    skipped = []

    for edf_path in edf_files:
        run_id = parse_run_id(edf_path)
        subject = parse_subject(edf_path)

        if run_id is None:
            skipped.append((str(edf_path), "no run id"))
            continue

        df = load_sentences(run_id)
        if df is None:
            skipped.append((str(edf_path), f"missing split_data_{run_id}.xlsx"))
            continue

        print(f"\nreading {subject} run {run_id}: {edf_path}")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.pick_types(eeg=True, exclude=[])
        raw.resample(TARGET_SFREQ, verbose=False)

        anns = raw.annotations
        n = min(len(anns), len(df))
        print("annotations:", len(anns), "sentences:", len(df), "use:", n)

        for i in range(n):
            onset = float(anns.onset[i])
            start = int(onset * raw.info["sfreq"])
            stop = start + int(WINDOW_SEC * raw.info["sfreq"])

            if stop > raw.n_times:
                continue

            sentence = str(df.iloc[i]["句子"]).strip()
            label = str(df.iloc[i]["标签"]).strip() if "标签" in df.columns else ""

            if not sentence:
                continue

            words = split_words(sentence)
            if not words:
                continue

            eeg = raw.get_data(start=start, stop=stop)

            records.append({
                "eeg": torch.tensor(eeg, dtype=SAVE_DTYPE),
                "sentence": sentence,
                "words": words,
                "label": label,
                "sentence_mode": sentence_mode(sentence),
                "subjectivity": subjectivity(sentence),
                "semantic_focus": semantic_focus(sentence),
                "source": {
                    "dataset": "Chisco",
                    "subject": subject,
                    "run": run_id,
                    "trial": i,
                    "edf": str(edf_path),
                    "onset": onset,
                    "window_sec": WINDOW_SEC,
                    "sfreq": TARGET_SFREQ,
                },
            })

        del raw

    print("\nrecords:", len(records))
    print("skipped:", len(skipped))
    for item in skipped[:20]:
        print("skip:", item)

    if not records:
        raise RuntimeError("No records generated")

    random.seed(SEED)
    random.shuffle(records)

    n_val = max(1, int(len(records) * VAL_RATIO))
    val = records[:n_val]
    train = records[n_val:]

    torch.save(train, OUT_ROOT / "train.pt")
    torch.save(val, OUT_ROOT / "val.pt")

    meta = {
        "dataset": "Chisco",
        "raw_root": str(RAW_ROOT),
        "out_root": str(OUT_ROOT),
        "num_records": len(records),
        "num_train": len(train),
        "num_val": len(val),
        "window_sec": WINDOW_SEC,
        "target_sfreq": TARGET_SFREQ,
        "save_dtype": str(SAVE_DTYPE),
        "val_ratio": VAL_RATIO,
        "seed": SEED,
        "note": "Words are generated by jieba; sentence attributes are heuristic labels.",
    }
    torch.save(meta, OUT_ROOT / "meta.pt")

    print("\nsaved:")
    print("train:", len(train), OUT_ROOT / "train.pt")
    print("val:", len(val), OUT_ROOT / "val.pt")
    print("meta:", OUT_ROOT / "meta.pt")
    print("example eeg:", tuple(train[0]["eeg"].shape), train[0]["eeg"].dtype)
    print("example sentence:", train[0]["sentence"])
    print("example words:", train[0]["words"])

if __name__ == "__main__":
    main()
