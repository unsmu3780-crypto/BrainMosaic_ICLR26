from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import mne
import pandas as pd
import torch


RAW_ROOT = Path(
    "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/ChineseEEG-2"
)
OUT_ROOT = Path(
    "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/ChineseEEG2"
)
MATERIALS_ROOT = RAW_ROOT / "materials&embeddings"
TARGET_SFREQ = 250
VAL_RATIO = 0.2
SEED = 42

TASK_DIRS = {
    "ReadingAloud": {
        "zip_glob": "ReadingAloud/derivatives/preprocessed/*.zip",
    },
    "PassiveListening": {
        "zip_glob": "PassiveListening/derivatives/preprocessed/*.zip",
    },
}

SESSION_RUN_LAYOUT = {
    "littleprince": [14, 13],
    "garnettdream": [5, 4],
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Prepare ChineseEEG-2 into unified BrainMosaic train.pt/val.pt splits"
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--materials-root", type=Path, default=MATERIALS_ROOT)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_DIRS),
        default=["ReadingAloud", "PassiveListening"],
    )
    parser.add_argument("--target-sfreq", type=int, default=TARGET_SFREQ)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text).replace("\ufeff", "").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_words(sentence: str) -> list[str]:
    words = []
    for word in jieba.lcut(str(sentence)):
        word = normalize_text(word)
        if not word:
            continue
        if re.fullmatch(r"[\W_]+", word):
            continue
        words.append(word)
    return words[:8]


def sentence_mode(sentence: str) -> int:
    s = str(sentence)
    if any(ch in s for ch in ["?", "\uff1f"]):
        return 1
    if any(x in s for x in ["\u4e0d", "\u6ca1", "\u65e0", "\u522b", "\u52ff"]):
        return 2
    if any(x in s for x in ["\u8bf7", "\u52a1\u5fc5", "\u8bb0\u5f97", "\u8d76\u7d27", "\u7acb\u5373"]):
        return 3
    return 0


def subjectivity(sentence: str) -> int:
    s = str(sentence)
    markers = [
        "\u6211\u89c9\u5f97",
        "\u559c\u6b22",
        "\u8ba8\u538c",
        "\u62c5\u5fc3",
        "\u5e0c\u671b",
        "\u5bb3\u6015",
        "\u8ba4\u4e3a",
        "\u60f3",
        "\u7231",
        "\u96be\u8fc7",
        "\u5f00\u5fc3",
    ]
    return 1 if any(x in s for x in markers) else 0


def semantic_focus(sentence: str) -> int:
    s = str(sentence)
    if any(x in s for x in ["\u6211", "\u6211\u4eec", "\u54b1\u4eec", "\u81ea\u5df1"]):
        return 0
    if any(x in s for x in ["\u4f60", "\u4f60\u4eec", "\u60a8"]):
        return 1
    if any(x in s for x in ["\u4ed6", "\u5979", "\u4ed6\u4eec", "\u5979\u4eec", "\u5b83", "\u5b83\u4eec"]):
        return 2
    if any(x in s for x in ["\u4e8b\u60c5", "\u4e1c\u897f", "\u6545\u4e8b", "\u97f3\u4e50", "\u58f0\u97f3", "\u98ce\u666f", "\u57ce\u5e02"]):
        return 3
    return 4


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[\u3002\uff01\uff1f!?\uff1b;])\s*", text)
    sentences = [normalize_text(part) for part in parts if normalize_text(part)]
    return [s for s in sentences if re.search(r"\w|[\u4e00-\u9fff]", s)]


def parse_subject_from_zip(path: Path) -> str:
    match = re.search(r"(sub-[A-Za-z0-9]+)\.zip$", path.name)
    return match.group(1) if match else path.stem


def parse_run_number(path: str) -> str | None:
    match = re.search(r"_run-(\d+)_", path)
    return match.group(1) if match else None


def parse_session_name(path: str) -> str | None:
    match = re.search(r"_ses-([A-Za-z0-9]+)_", path)
    return match.group(1).lower() if match else None


def compute_chapter_index(run_number: str) -> tuple[int, int] | None:
    if not run_number or len(run_number) < 2:
        return None
    run_group = int(run_number[0])
    chapter_in_group = int(run_number[1:])
    if run_group <= 0 or chapter_in_group <= 0:
        return None
    return run_group, chapter_in_group


def chapter_from_run(session: str, run_number: str) -> int | None:
    parsed = compute_chapter_index(run_number)
    if parsed is None:
        return None
    run_group, chapter_in_group = parsed
    layout = SESSION_RUN_LAYOUT.get(session)
    if not layout or run_group > len(layout):
        return None
    offset = sum(layout[: run_group - 1])
    return offset + chapter_in_group


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def guess_text_column(columns: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in columns}
    candidates = [
        "text",
        "sentence",
        "content",
        "stimulus",
        "story",
        "paragraph",
        "chapter_text",
        "\u7ae0\u8282\u5185\u5bb9",
        "\u5185\u5bb9",
        "\u53e5\u5b50",
        "\u6587\u672c",
    ]
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    for column in columns:
        name = str(column).strip().lower()
        if any(token in name for token in ["text", "sentence", "content", "stimulus", "story"]):
            return column
    return None


def guess_chapter_column(columns: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in columns}
    candidates = ["chapter", "chapter_id", "chapteridx", "\u7ae0\u8282", "chapter_index"]
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    for column in columns:
        name = str(column).strip().lower()
        if "chapter" in name or "\u7ae0\u8282" in name:
            return column
    return None


def load_materials_from_tables(materials_root: Path) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for session in SESSION_RUN_LAYOUT:
        for path in sorted(materials_root.rglob(f"*{session}*")):
            if path.suffix.lower() not in [".xlsx", ".xls", ".csv", ".tsv"]:
                continue
            try:
                df = read_table(path)
            except Exception:
                continue
            if df.empty:
                continue
            chapter_col = guess_chapter_column(list(df.columns))
            text_col = guess_text_column(list(df.columns))
            if chapter_col is None or text_col is None:
                continue
            for _, row in df.iterrows():
                chapter_value = row.get(chapter_col)
                text_value = normalize_text(row.get(text_col, ""))
                if pd.isna(chapter_value) or not text_value:
                    continue
                try:
                    chapter_index = int(chapter_value)
                except Exception:
                    continue
                out[session][chapter_index] = text_value
            if out[session]:
                break
    return out


def parse_chapter_text_file(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"^\s*Ch(?:apter)?\s*0*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    chapters = {}
    for idx, match in enumerate(matches):
        chapter_index = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = normalize_text(text[start:end])
        if body:
            chapters[chapter_index] = body
    return chapters


def load_materials(materials_root: Path) -> dict[str, dict[int, str]]:
    materials = load_materials_from_tables(materials_root)
    for session in SESSION_RUN_LAYOUT:
        if materials.get(session):
            continue
        for path in sorted(materials_root.rglob(f"*{session}*.txt")):
            chapters = parse_chapter_text_file(path)
            if chapters:
                materials[session] = chapters
                break
    return materials


def extract_sentences_from_events(events_df: pd.DataFrame) -> list[str]:
    candidates = [
        "sentence",
        "text",
        "content",
        "stimulus",
        "stim_text",
        "trial_text",
        "value",
        "annotation",
        "trial_type",
    ]
    columns = {str(col).strip().lower(): col for col in events_df.columns}

    for key in candidates:
        if key not in columns:
            continue
        series = events_df[columns[key]]
        values = []
        seen = set()
        for value in series:
            if pd.isna(value):
                continue
            text = normalize_text(value)
            if not text or text.lower() == "nan":
                continue
            if text in seen:
                continue
            seen.add(text)
            values.append(text)
        if not values:
            continue

        # If the column stores one long chapter string, split it into sentences.
        if len(values) == 1:
            return split_sentences(values[0])

        # If rows already look sentence-like, keep them as units.
        if sum(len(split_sentences(text)) for text in values[:10]) <= len(values[:10]) * 2:
            return values

        joined = " ".join(values)
        split_joined = split_sentences(joined)
        if split_joined:
            return split_joined
    return []


def read_events_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as handle:
        return pd.read_csv(handle, sep="\t")


def event_time_bounds(events_df: pd.DataFrame, full_duration: float) -> tuple[float, float]:
    if "onset" not in events_df.columns:
        return 0.0, full_duration
    onsets = pd.to_numeric(events_df["onset"], errors="coerce").dropna()
    if onsets.empty:
        return 0.0, full_duration
    if "duration" in events_df.columns:
        durations = pd.to_numeric(events_df["duration"], errors="coerce").fillna(0.0)
        ends = (pd.to_numeric(events_df["onset"], errors="coerce").fillna(0.0) + durations).dropna()
        end_sec = float(ends.max()) if not ends.empty else full_duration
    else:
        end_sec = float(onsets.max())
    start_sec = float(onsets.min())
    if end_sec - start_sec < 1.0:
        return 0.0, full_duration
    return max(0.0, start_sec), min(full_duration, end_sec)


def collect_run_assets(zf: zipfile.ZipFile) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for name in zf.namelist():
        if not name.endswith((".vhdr", ".eeg", ".vmrk", "_events.tsv")):
            continue
        if "/eeg/" not in name:
            continue
        stem = Path(name).name
        if "_eeg." in stem:
            key = name.rsplit("_eeg.", 1)[0]
            groups[key][Path(name).suffix.lower().lstrip(".")] = name
        elif stem.endswith("_events.tsv"):
            key = name.rsplit("_events.tsv", 1)[0]
            groups[key]["events"] = name
    return groups


def extract_run_to_temp(zf: zipfile.ZipFile, asset_group: dict[str, str], temp_dir: Path) -> Path:
    needed = ["vhdr", "eeg", "vmrk"]
    for kind in needed:
        if kind not in asset_group:
            raise FileNotFoundError(f"Missing {kind} file in asset group: {asset_group}")
    out_dir = temp_dir / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in needed:
        member = asset_group[kind]
        target = out_dir / Path(member).name
        with zf.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())
    return out_dir / Path(asset_group["vhdr"]).name


def estimate_sentence_windows(
    sentences: list[str],
    start_sec: float,
    end_sec: float,
) -> list[tuple[str, float, float]]:
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0.0 or not sentences:
        return []
    weights = [max(1, len(re.sub(r"\s+", "", sentence))) for sentence in sentences]
    total_weight = float(sum(weights))
    cursor = start_sec
    windows = []
    for idx, sentence in enumerate(sentences):
        portion = duration * (weights[idx] / total_weight)
        next_cursor = end_sec if idx == len(sentences) - 1 else cursor + portion
        windows.append((sentence, cursor, next_cursor))
        cursor = next_cursor
    return windows


def build_record(
    raw: mne.io.BaseRaw,
    sentence: str,
    start_sec: float,
    end_sec: float,
    meta: dict[str, object],
    save_dtype: str,
) -> dict[str, object] | None:
    start = int(start_sec * raw.info["sfreq"])
    stop = int(end_sec * raw.info["sfreq"])
    if stop - start < max(16, int(0.25 * raw.info["sfreq"])):
        return None
    if stop > raw.n_times:
        stop = raw.n_times
    if start < 0 or start >= stop:
        return None

    words = split_words(sentence)
    if not words:
        return None

    eeg = raw.get_data(start=start, stop=stop)
    return {
        "eeg": torch.as_tensor(eeg, dtype=getattr(torch, save_dtype)),
        "sentence": sentence,
        "words": words,
        "sentence_mode": sentence_mode(sentence),
        "subjectivity": subjectivity(sentence),
        "semantic_focus": semantic_focus(sentence),
        "source": meta,
    }


def assign_split(records: list[dict[str, object]], val_ratio: float, seed: int):
    train, val = [], []
    for record in records:
        source = record.get("source", {})
        key = ":".join(
            [
                str(source.get("subject", "")),
                str(source.get("task", "")),
                str(source.get("session", "")),
                str(source.get("run", "")),
                str(source.get("sentence_index", "")),
            ]
        )
        rnd = random.Random(f"{seed}:{key}")
        if rnd.random() < val_ratio:
            val.append(record)
        else:
            train.append(record)
    if not train or not val:
        raise RuntimeError(
            f"Split degenerated: train={len(train)} val={len(val)}. Adjust val_ratio or inputs."
        )
    return train, val


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    materials_root = args.materials_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    materials = load_materials(materials_root) if materials_root.exists() else {}
    records = []
    skipped = []
    stats = Counter()
    channels_seen = Counter()

    for task_name in args.tasks:
        zip_paths = sorted(raw_root.glob(TASK_DIRS[task_name]["zip_glob"]))
        if not zip_paths:
            skipped.append((task_name, "missing_zip_inputs"))
            continue

        for zip_path in zip_paths:
            subject = parse_subject_from_zip(zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                run_groups = collect_run_assets(zf)
                for asset_key, asset_group in sorted(run_groups.items()):
                    if "events" not in asset_group:
                        skipped.append((zip_path.name, asset_key, "missing_events"))
                        continue

                    run_number = parse_run_number(asset_group["events"])
                    session = parse_session_name(asset_group["events"])
                    if not run_number or not session:
                        skipped.append((zip_path.name, asset_key, "missing_run_or_session"))
                        continue

                    events_df = read_events_from_zip(zf, asset_group["events"])
                    chapter_index = chapter_from_run(session, run_number)
                    chapter_text = materials.get(session, {}).get(chapter_index) if chapter_index is not None else None
                    sentences = split_sentences(chapter_text) if chapter_text else []
                    if not sentences:
                        sentences = extract_sentences_from_events(events_df)
                    if not sentences:
                        skipped.append(
                            (zip_path.name, asset_key, f"missing_sentence_source:{session}:{run_number}")
                        )
                        continue

                    with tempfile.TemporaryDirectory(prefix="chineseeeg2_") as temp_dir:
                        vhdr_path = extract_run_to_temp(zf, asset_group, Path(temp_dir))
                        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
                        raw.pick_types(eeg=True, exclude=[])
                        raw.resample(args.target_sfreq, verbose=False)
                        full_duration = raw.n_times / raw.info["sfreq"]
                        start_sec, end_sec = event_time_bounds(events_df, full_duration)
                        windows = estimate_sentence_windows(sentences, start_sec, end_sec)
                        if not windows:
                            skipped.append((zip_path.name, asset_key, "no_sentence_windows"))
                            continue

                        channels_seen[raw.info["nchan"]] += 1
                        for sent_idx, (sentence, sent_start, sent_end) in enumerate(windows):
                            meta = {
                                "dataset": "ChineseEEG2",
                                "subject": subject,
                                "task": task_name,
                                "session": session,
                                "run": run_number,
                                "chapter_index": chapter_index,
                                "sentence_index": sent_idx,
                                "zip": str(zip_path),
                                "events_file": asset_group["events"],
                                "window_start_sec": sent_start,
                                "window_end_sec": sent_end,
                                "timing_mode": "chapter_duration_proportional",
                                "text_source": "materials" if chapter_text else "events_tsv",
                                "target_sfreq": args.target_sfreq,
                            }
                            record = build_record(raw, sentence, sent_start, sent_end, meta, args.save_dtype)
                            if record is None:
                                continue
                            records.append(record)
                            stats[f"task:{task_name}"] += 1
                            stats[f"session:{session}"] += 1

                        stats["runs_used"] += 1
                        stats[f"subject:{subject}"] += len(windows)
                        del raw

    if not records:
        raise RuntimeError("No ChineseEEG-2 records generated")

    train, val = assign_split(records, val_ratio=args.val_ratio, seed=args.seed)
    torch.save(train, out_root / "train.pt")
    torch.save(val, out_root / "val.pt")

    meta = {
        "dataset": "ChineseEEG2",
        "raw_root": str(raw_root),
        "materials_root": str(materials_root),
        "out_root": str(out_root),
        "num_records": len(records),
        "num_train": len(train),
        "num_val": len(val),
        "target_sfreq": args.target_sfreq,
        "save_dtype": args.save_dtype,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "tasks": args.tasks,
        "channels_seen": dict(channels_seen),
        "stats": dict(stats),
        "skipped_preview": skipped[:100],
        "notes": [
            "This public reproduction path uses derivative preprocessed BrainVision zips.",
            "Sentence windows are approximated by proportional allocation within each run's event span.",
            "Sentence text is reconstructed from materials chapters and split with punctuation heuristics.",
        ],
    }
    torch.save(meta, out_root / "meta.pt")
    with open(out_root / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print("[OK] ChineseEEG-2 unified splits written")
    print(f"  train: {len(train)} -> {out_root / 'train.pt'}")
    print(f"  val: {len(val)} -> {out_root / 'val.pt'}")
    print(f"  meta: {out_root / 'meta.pt'}")
    print(f"  records: {len(records)}")
    print(f"  channels seen: {dict(channels_seen)}")
    print(f"  tasks: {args.tasks}")
    print(f"  skipped preview count: {len(skipped[:100])}")
    print(f"  example eeg: {tuple(train[0]['eeg'].shape)} {train[0]['eeg'].dtype}")
    print(f"  example sentence: {train[0]['sentence']}")
    print(f"  example words: {train[0]['words']}")


if __name__ == "__main__":
    main()
