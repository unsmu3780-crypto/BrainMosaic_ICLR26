import argparse
import json
import time
from pathlib import Path

import httpx


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_existing_samples(path):
    path = Path(path)
    if not path.exists():
        return [], {}

    data = load_json(path)
    samples = data.get("samples", []) if isinstance(data, dict) else []
    index = {}
    for item in samples:
        sample_index = item.get("sample_index")
        if sample_index is not None:
            index[sample_index] = item
    return samples, index


def _head_hint_text(head_name, probs, cfg, language):
    if not isinstance(probs, list) or len(probs) == 0:
        return None
    threshold_cfg = cfg["prompt"].get("hint_thresholds", {})
    old_key_alias = {
        "sentence_mode": "te",
        "subjectivity": "oors",
        "semantic_focus": "su",
    }
    threshold = float(
        threshold_cfg.get(
            head_name,
            threshold_cfg.get(old_key_alias.get(head_name, ""), cfg["prompt"].get("hint_threshold", 0.7)),
        )
    )

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    best_prob = float(probs[best_idx])
    if best_prob < threshold:
        return None

    label_map = cfg["prompt"].get("hint_labels", {})
    labels = None
    if isinstance(label_map, dict):
        if language in label_map and isinstance(label_map.get(language), dict):
            labels = label_map[language].get(head_name)
            if labels is None:
                labels = label_map[language].get(old_key_alias.get(head_name, ""))
        elif head_name in label_map:
            labels = label_map.get(head_name)
    if not isinstance(labels, list) or best_idx >= len(labels):
        if language == "zh":
            labels = [f"{head_name}_{i}" for i in range(len(probs))]
        else:
            labels = [f"{head_name}_{i}" for i in range(len(probs))]

    label = str(labels[best_idx]).strip()
    if language == "zh":
        return f"这个句子可能是{label}"
    return f"This sentence may be {label}."


def _build_hints(sentence_mode_probs, subjectivity_probs, semantic_focus_probs, cfg, language):
    hints = []
    for head_name, probs in [
        ("sentence_mode", sentence_mode_probs),
        ("subjectivity", subjectivity_probs),
        ("semantic_focus", semantic_focus_probs),
    ]:
        hint = _head_hint_text(head_name, probs, cfg, language)
        if hint:
            hints.append(hint)
    return hints


def build_prompt(item, cfg):
    topk_words = item.get("topk_words", [])
    sentence_mode_probs = item.get("sentence_mode_probs", item.get("te_probs", []))
    subjectivity_probs = item.get("subjectivity_probs", item.get("oors_probs", []))
    semantic_focus_probs = item.get("semantic_focus_probs", item.get("su_probs", []))

    keywords = []
    for slot_words in topk_words:
        for group in slot_words:
            if isinstance(group, list) and group:
                keywords.append("、".join([str(w).strip() for w in group if str(w).strip()]))
    keywords = [k for k in keywords if k]
    keywords = keywords[: cfg["prompt"].get("max_keywords", 12)]

    language = cfg["prompt"].get("language", "en").lower()
    hints = _build_hints(
        sentence_mode_probs, subjectivity_probs, semantic_focus_probs, cfg, language
    )
    if language == "zh":
        hint_block = "\n".join(hints) if hints else "无可靠全局提示"
        return (
            "你是一个句子重构助手。\n"
            f"请生成恰好 {cfg['generation']['num_candidates']} 条候选句子。\n"
            f"每条句子最多 {cfg['prompt']['max_chars']} 个字符。\n"
            "每行仅输出一句。\n"
            f"关键词：{'；'.join(keywords)}\n"
            f"全局提示：\n{hint_block}"
        )
    hint_block = "\n".join(hints) if hints else "No reliable global hints."
    return (
        "You are a sentence reconstructor.\n"
        f"Generate exactly {cfg['generation']['num_candidates']} candidate sentences.\n"
        f"Max length per sentence: {cfg['prompt']['max_chars']} characters.\n"
        "Output one sentence per line.\n"
        f"Keywords: {'；'.join(keywords)}\n"
        f"Global hints:\n{hint_block}"
    )


def call_llm(prompt, cfg):
    endpoint = cfg["llm"]["endpoint"]
    model = cfg["llm"]["model"]
    api_key = cfg["llm"]["api_key"]
    timeout_sec = cfg["llm"].get("timeout_sec", 60)
    max_retries = int(cfg["llm"].get("max_retries", 3))
    retry_backoff_sec = float(cfg["llm"].get("retry_backoff_sec", 2.0))

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["generation"].get("temperature", 0.7),
        "max_tokens": cfg["generation"].get("max_tokens", 200),
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                res = client.post(endpoint, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
            content = data["choices"][0]["message"]["content"]
            lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
            return lines[: cfg["generation"]["num_candidates"]]
        except Exception as err:
            last_err = err
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_sec * attempt)
    raise last_err


def reconstruct(config_path):
    cfg = load_json(config_path)
    topk_items = load_json(cfg["input"]["topk_json"])
    out_path = Path(cfg["output"]["reconstruction_json"])
    partial_path = Path(cfg["output"].get("partial_json", str(out_path) + ".partial"))
    resume = bool(cfg["output"].get("resume", True))
    flush_every = int(cfg["output"].get("flush_every", 20))
    include_prompt = bool(cfg["output"].get("include_prompt", False))

    existing_samples = []
    existing_index = {}
    if resume:
        existing_samples, existing_index = load_existing_samples(out_path)
        if not existing_samples and partial_path.exists():
            existing_samples, existing_index = load_existing_samples(partial_path)

    out = list(existing_samples)
    samples_since_flush = 0

    for row_idx, item in enumerate(topk_items):
        sample_index = item.get("sample_index", row_idx)
        if resume and sample_index in existing_index:
            print(f"[resume] skip sample_index={sample_index}")
            continue

        prompt = build_prompt(item, cfg)
        record = {
            "sample_index": sample_index,
            "gold_sentence": item.get("gold_sentence", ""),
        }
        if include_prompt:
            record["prompt"] = prompt

        try:
            cands = call_llm(prompt, cfg)
            record["candidates"] = cands
            record["status"] = "ok"
            print(f"[ok] sample_index={sample_index} candidates={len(cands)}")
        except Exception as err:
            record["candidates"] = []
            record["status"] = "error"
            record["error"] = f"{type(err).__name__}: {err}"
            print(f"[error] sample_index={sample_index} {record['error']}")

        out.append(record)
        existing_index[sample_index] = record
        samples_since_flush += 1

        if samples_since_flush >= flush_every:
            save_json({"samples": out}, partial_path, indent=None)
            samples_since_flush = 0

    save_json({"samples": out}, out_path)
    save_json({"samples": out}, partial_path, indent=None)
    print(f"Saved: {out_path}")
    print(f"Partial/Resume file: {partial_path}")


def main():
    parser = argparse.ArgumentParser("Standalone semantic-guided sentence reconstruction")
    parser.add_argument("--config", required=True, type=str, help="Path to reconstruction config json")
    args = parser.parse_args()
    reconstruct(args.config)


if __name__ == "__main__":
    main()
