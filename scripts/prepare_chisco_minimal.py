from pathlib import Path
import re
import random

import jieba
import mne
import pandas as pd
import torch

RAW_ROOT = Path("/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/Chiso")
OUT_ROOT = Path("/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/real_data/Chisco_minimal")

SUBJECT = "sub-01"
MAX_RUNS = 5
WINDOW_SEC = 8.0
TARGET_SFREQ = 250
SEED = 42

TEXTUAL_ID = {"statement": 0, "question": 1, "negative": 2, "imperative": 3}
OS_ID = {"objective": 0, "subjective": 1}
SUBJECT_ID = {"I/we": 0, "you": 1, "others": 2, "thing": 3, "event": 4}

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
    subjective_markers = ["我觉得", "喜欢", "讨厌", "害怕", "担心", "希望", "想", "怕", "爱", "开心", "难过"]
    return 1 if any(x in s for x in subjective_markers) else 0

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

def run_id_from_edf(path):
    m = re.search(r"run-0*(\d+)_eeg\.edf$", path.name)
    return int(m.group(1)) if m else None

def find_edf(subject, run_id):
    candidates = list((RAW_ROOT / subject).rglob(f"*run-{run_id:02d}_eeg.edf"))
    candidates += list((RAW_ROOT / subject).rglob(f"*run-{run_id:03d}_eeg.edf"))
    candidates = sorted(set(candidates))
    return candidates[0] if candidates else None

def load_sentences(run_id):
    path = RAW_ROOT / "textdataset" / f"split_data_{run_id}.xlsx"
    if not path.exists():
        return None
    df = pd.read_excel(path)
    if "句子" not in df.columns:
        raise ValueError(f"{path} missing column 句子")
    return df

def make_records():
    records = []
    run_ids = []
    for p in sorted((RAW_ROOT / SUBJECT).rglob("*.edf")):
        rid = run_id_from_edf(p)
        if rid is not None:
            run_ids.append(rid)
    run_ids = sorted(set(run_ids))[:MAX_RUNS]
    print("selected run_ids:", run_ids)

    for rid in run_ids:
        edf_path = find_edf(SUBJECT, rid)
        df = load_sentences(rid)
        if edf_path is None or df is None:
            print("skip run", rid, "edf", edf_path, "text", df is not None)
            continue

        print("reading", edf_path)
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        raw.pick_types(eeg=True, exclude=[])
        raw.resample(TARGET_SFREQ, verbose=False)

        anns = raw.annotations
        n = min(len(anns), len(df))
        print("run", rid, "annotations", len(anns), "sentences", len(df), "use", n)

        for i in range(n):
            onset = float(anns.onset[i])
            start = int(onset * raw.info["sfreq"])
            stop = start + int(WINDOW_SEC * raw.info["sfreq"])
            if stop > raw.n_times:
                continue

            eeg = raw.get_data(start=start, stop=stop)
            sentence = str(df.iloc[i]["句子"]).strip()
            label = str(df.iloc[i]["标签"]).strip() if "标签" in df.columns else ""

            if not sentence:
                continue

            words = split_words(sentence)
            if not words:
                continue

            records.append({
                "eeg": torch.tensor(eeg, dtype=torch.float32),
                "sentence": sentence,
                "words": words,
                "label": label,
                "sentence_mode": sentence_mode(sentence),
                "subjectivity": subjectivity(sentence),
                "semantic_focus": semantic_focus(sentence),
                "source": {
                    "dataset": "Chisco",
                    "subject": SUBJECT,
                    "run": rid,
                    "trial": i,
                    "edf": str(edf_path),
                    "onset": onset,
                }
            })

    return records

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = make_records()
    print("records:", len(records))
    if not records:
        raise RuntimeError("No records generated")

    random.seed(SEED)
    random.shuffle(records)

    n_val = max(1, int(len(records) * 0.2))
    val = records[:n_val]
    train = records[n_val:]

    torch.save(train, OUT_ROOT / "train.pt")
    torch.save(val, OUT_ROOT / "val.pt")

    print("saved train:", len(train), OUT_ROOT / "train.pt")
    print("saved val:", len(val), OUT_ROOT / "val.pt")
    print("example keys:", train[0].keys())
    print("example eeg shape:", tuple(train[0]["eeg"].shape))
    print("example sentence:", train[0]["sentence"])
    print("example words:", train[0]["words"])

if __name__ == "__main__":
    main()
