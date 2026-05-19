import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


DEFAULT_TEXT_ROOT = Path(
    os.environ.get(
        "CHISCO_TEXT_ROOT",
        "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets",
    )
)

DEFAULT_SYSTEM_PROMPT = (
    "You are building semantic-unit explanations for an EEG semantic intent decoding dataset. "
    "For each Chinese token, write one concise Chinese explanation phrase that preserves its "
    "meaning in the given sentence contexts. Do not add unrelated information. "
    "Return only the explanation phrase."
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_contexts(segmentation, max_contexts):
    contexts = defaultdict(list)
    for item in segmentation:
        sentence = str(item.get("sentence", "")).strip()
        if not sentence:
            continue
        for token in item.get("tokens", []) or []:
            token = str(token).strip()
            if token and sentence not in contexts[token]:
                contexts[token].append(sentence)
                if len(contexts[token]) >= max_contexts:
                    break
    return contexts


def template_expand(token, contexts):
    if contexts:
        joined = "；".join(contexts[:2])
        return f"在句子语境中表示“{token}”的语义单元，例句：{joined}"
    return f"表示“{token}”的中文语义单元"


def make_user_prompt(token, contexts):
    context_lines = "\n".join(f"- {s}" for s in contexts[:5]) or "- 无"
    return (
        f"中文 token：{token}\n"
        f"出现语境：\n{context_lines}\n\n"
        "请给出一个适合做语义 embedding 的短解释短语。"
    )


def chat_completion(base_url, api_key, model, system_prompt, user_prompt, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 80,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    result = json.loads(raw)
    return str(result["choices"][0]["message"]["content"]).strip()


def parse_args():
    parser = argparse.ArgumentParser("Expand Chisco token explanations before embedding")
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--segmentation-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=["template", "openai-compatible"],
        default="template",
        help="Use template for offline deterministic expansion, or an OpenAI-compatible chat API for LLM expansion.",
    )
    parser.add_argument("--model", default=os.environ.get("EXPANSION_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0, help="For debugging; 0 means all tokens.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing output explanations when present.")
    return parser.parse_args()


def main():
    args = parse_args()
    text_root = args.text_root.resolve()
    tokens_file = args.tokens_file or (text_root / "token_explanations.json")
    segmentation_file = args.segmentation_file or (text_root / "segmentation.json")
    output_file = args.output_file or (text_root / "token_explanations.expanded.json")

    rows = load_json(tokens_file)
    segmentation = load_json(segmentation_file)
    contexts = collect_contexts(segmentation, max_contexts=args.max_contexts)

    existing = {}
    if args.resume and output_file.exists():
        for row in load_json(output_file):
            key = str(row.get("key", "")).strip()
            exp = str(row.get("explanation", "")).strip()
            if key and exp:
                existing[key] = exp

    if args.backend == "openai-compatible" and not args.api_key:
        raise ValueError("OPENAI_API_KEY or --api-key is required for openai-compatible backend")

    out = []
    todo = rows[: args.limit] if args.limit and args.limit > 0 else rows
    for idx, row in enumerate(todo, start=1):
        token = str(row.get("key", "")).strip()
        if not token:
            continue

        if token in existing:
            explanation = existing[token]
            source = "resume"
        elif args.backend == "template":
            explanation = template_expand(token, contexts.get(token, []))
            source = "template"
        else:
            prompt = make_user_prompt(token, contexts.get(token, []))
            try:
                explanation = chat_completion(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_prompt=prompt,
                    timeout=args.timeout,
                )
                source = args.model
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                explanation = template_expand(token, contexts.get(token, []))
                source = f"template_fallback_after_error:{type(exc).__name__}"

        out.append(
            {
                "key": token,
                "explanation": explanation,
                "count": row.get("count", 0),
                "contexts": contexts.get(token, [])[: args.max_contexts],
                "expansion_source": source,
            }
        )

        if idx % 50 == 0 or idx == len(todo):
            write_json(output_file, out)
            print(f"[progress] {idx}/{len(todo)} -> {output_file}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_json(output_file, out)
    print(f"[OK] expanded tokens: {len(out)}")
    print(f"[OUT] {output_file}")
    print("Use this file as inputs.tokens_file in configs/text_embedding.chisco.json")


if __name__ == "__main__":
    main()
