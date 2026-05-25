import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def install_torch_compiler_shim():
    for dtype_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        if not hasattr(torch, dtype_name):
            setattr(torch, dtype_name, torch.uint8)

    if hasattr(torch, "compiler"):
        return

    class _CompilerShim:
        @staticmethod
        def disable(fn=None, recursive=True):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        @staticmethod
        def is_compiling():
            return False

    torch.compiler = _CompilerShim()


def install_load_state_dict_assign_shim():
    try:
        signature = inspect.signature(torch.nn.Module.load_state_dict)
    except (TypeError, ValueError):
        return
    if "assign" in signature.parameters:
        return

    original_load_state_dict = torch.nn.Module.load_state_dict

    def set_tensor_by_name(module, name, tensor):
        parts = name.split(".")
        target = module
        for part in parts[:-1]:
            if part in target._modules:
                target = target._modules[part]
            else:
                target = getattr(target, part)

        leaf = parts[-1]
        if leaf in target._parameters:
            old_param = target._parameters[leaf]
            requires_grad = old_param.requires_grad if old_param is not None else True
            target._parameters[leaf] = torch.nn.Parameter(tensor.detach(), requires_grad=requires_grad)
            return
        if leaf in target._buffers:
            target._buffers[leaf] = tensor.detach()

    def load_state_dict_compat(self, state_dict, strict=True, assign=False):
        if assign:
            for name, tensor in state_dict.items():
                set_tensor_by_name(self, name, tensor)
        return original_load_state_dict(self, state_dict, strict=strict)

    torch.nn.Module.load_state_dict = load_state_dict_compat


install_torch_compiler_shim()
install_load_state_dict_assign_shim()

from bert_score import score as bert_score
from transformers import AutoModel, AutoTokenizer


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    truncate_dim: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        hidden = model(**tokens).last_hidden_state
        pooled = mean_pool(hidden, tokens["attention_mask"])
        if truncate_dim and truncate_dim > 0:
            pooled = pooled[:, :truncate_dim]
        pooled = F.normalize(pooled, dim=-1)
        outputs.append(pooled.cpu())
    if not outputs:
        return torch.empty(0, truncate_dim)
    return torch.cat(outputs, dim=0)


def build_gold_embedding_map(sent_emb_path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(sent_emb_path, map_location="cpu")
    sentences = payload["sentences"]
    embeddings = payload["embeddings"]
    return {str(sentence): embeddings[idx] for idx, sentence in enumerate(sentences)}


def resolve_concept_metrics(run_dir: Path) -> Tuple[float, float, int]:
    best_summary_path = run_dir / "best_summary.json"
    epoch_history_path = run_dir / "epoch_history.json"

    best_summary = load_json(str(best_summary_path))
    best_epoch = int(best_summary["best_epoch"])
    uma = float(best_summary["best_acc"])

    epoch_history = load_json(str(epoch_history_path))
    matched = next((item for item in epoch_history if int(item["epoch"]) == best_epoch), None)
    if matched is None:
        raise ValueError(f"best_epoch={best_epoch} not found in {epoch_history_path}")
    mus = float(matched["val"]["mean_cosine"])
    return uma, mus, best_epoch


def compute_srs_and_candidates(
    reconstruction_samples: List[dict],
    gold_embedding_map: Dict[str, torch.Tensor],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    truncate_dim: int,
) -> Tuple[List[dict], float]:
    flat_candidates = []
    candidate_index = []
    fallback_gold = []
    fallback_gold_indices = []

    per_sample = []
    for sample_idx, item in enumerate(reconstruction_samples):
        gold_sentence = str(item.get("gold_sentence", "")).strip()
        candidates = [str(c).strip() for c in item.get("candidates", []) if str(c).strip()]
        if not candidates:
            candidates = [""]
        gold_embedding = gold_embedding_map.get(gold_sentence)
        if gold_embedding is None:
            fallback_gold.append(gold_sentence)
            fallback_gold_indices.append(sample_idx)

        start = len(flat_candidates)
        flat_candidates.extend(candidates)
        candidate_index.append((start, len(flat_candidates)))
        per_sample.append(
            {
                "sample_index": item.get("sample_index"),
                "gold_sentence": gold_sentence,
                "candidates": candidates,
                "candidate_count": len(candidates),
                "srs_per_candidate": [],
                "srs_mean": None,
            }
        )

    candidate_embeddings = encode_texts(
        flat_candidates,
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )

    if fallback_gold:
        fallback_embeddings = encode_texts(
            fallback_gold,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )
        for idx, emb in zip(fallback_gold_indices, fallback_embeddings):
            gold_embedding_map[per_sample[idx]["gold_sentence"]] = emb

    sample_means = []
    for idx, (start, end) in enumerate(candidate_index):
        gold_embedding = gold_embedding_map[per_sample[idx]["gold_sentence"]]
        cand_embeddings = candidate_embeddings[start:end]
        sims = torch.matmul(cand_embeddings, gold_embedding).tolist()
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        per_sample[idx]["srs_per_candidate"] = [float(x) for x in sims]
        per_sample[idx]["srs_mean"] = float(mean_sim)
        sample_means.append(mean_sim)

    overall_srs = sum(sample_means) / len(sample_means) if sample_means else 0.0
    return per_sample, overall_srs


def compute_bert_f1(per_sample: List[dict], lang: str, model_type: str = None) -> float:
    flat_candidates = []
    flat_golds = []
    sample_spans = []

    for item in per_sample:
        start = len(flat_candidates)
        for candidate in item["candidates"]:
            flat_candidates.append(candidate)
            flat_golds.append(item["gold_sentence"])
        sample_spans.append((start, len(flat_candidates)))

    if not flat_candidates:
        return 0.0

    kwargs = {
        "cands": flat_candidates,
        "refs": flat_golds,
        "lang": lang,
        "rescale_with_baseline": True,
    }
    if model_type:
        kwargs["model_type"] = model_type

    _, _, f1 = bert_score(**kwargs)
    f1_values = f1.cpu().tolist()

    per_sample_means = []
    for item, (start, end) in zip(per_sample, sample_spans):
        values = [float(x) for x in f1_values[start:end]]
        item["bert_f1_per_candidate"] = values
        item["bert_f1_mean"] = float(sum(values) / len(values)) if values else 0.0
        per_sample_means.append(item["bert_f1_mean"])

    return sum(per_sample_means) / len(per_sample_means) if per_sample_means else 0.0


def main():
    parser = argparse.ArgumentParser("Evaluate BrainMosaic reconstruction with paper-style metrics")
    parser.add_argument("--config", required=True, type=str, help="Path to evaluation config JSON")
    args = parser.parse_args()

    cfg = load_json(args.config)
    run_dir = Path(cfg["input"]["run_dir"])
    reconstruction_json = cfg["input"]["reconstruction_json"]
    sentence_embeddings_pt = cfg["input"]["sentence_embeddings_pt"]

    model_name = cfg["embedding_model"]["name_or_path"]
    device = cfg["runtime"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["runtime"].get("batch_size", 32))
    max_length = int(cfg["runtime"].get("max_length", 128))
    truncate_dim = int(cfg["runtime"].get("truncate_dim", 256))
    bert_lang = cfg["metrics"].get("bert_score_lang", "zh")
    bert_model_type = cfg["metrics"].get("bert_score_model_type")

    uma, mus, best_epoch = resolve_concept_metrics(run_dir)
    reconstruction_payload = load_json(reconstruction_json)
    reconstruction_samples = reconstruction_payload.get("samples", reconstruction_payload)
    gold_embedding_map = build_gold_embedding_map(sentence_embeddings_pt)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    ).eval().to(device)

    per_sample, srs = compute_srs_and_candidates(
        reconstruction_samples=reconstruction_samples,
        gold_embedding_map=gold_embedding_map,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        truncate_dim=truncate_dim,
    )
    bert_f1 = compute_bert_f1(per_sample, lang=bert_lang, model_type=bert_model_type)

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "num_samples": len(per_sample),
        "num_candidates_per_sample_expected": cfg["metrics"].get("expected_candidates", 5),
        "metrics": {
            "UMA": float(uma),
            "MUS": float(mus),
            "SRS": float(srs),
            "BERT-F1": float(bert_f1),
        },
    }

    output_json = cfg["output"]["summary_json"]
    output_samples_json = cfg["output"].get("per_sample_json")
    save_json(summary, output_json)
    if output_samples_json:
        save_json({"samples": per_sample}, output_samples_json)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
