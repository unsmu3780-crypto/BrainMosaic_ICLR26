import argparse
import inspect
import json
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F


def install_torch_compiler_shim():
    """Provide the small newer-Torch API surface used while importing Qwen3."""
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


install_torch_compiler_shim()


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


install_load_state_dict_assign_shim()

from transformers import AutoModel, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if p.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def resolve_torch_dtype(device: str, runtime_cfg: dict):
    dtype_name = str(runtime_cfg.get("torch_dtype", "auto")).strip().lower()
    if dtype_name in ("", "auto"):
        if str(device).startswith("cuda"):
            if hasattr(torch, "bfloat16"):
                return torch.bfloat16
            return torch.float16
        return torch.float32

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype={dtype_name}")
    return mapping[dtype_name]


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = 128,
    truncate_dim: int = 256,
):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        hidden = model(**tokens).last_hidden_state
        pooled = mean_pool(hidden, tokens["attention_mask"])
        if truncate_dim is not None and truncate_dim > 0:
            pooled = pooled[:, :truncate_dim]
        pooled = F.normalize(pooled, dim=-1)
        outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, truncate_dim)


def main():
    parser = argparse.ArgumentParser("Generate public text embeddings for BrainMosaic")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name_or_path"]
    device = cfg["runtime"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["runtime"].get("batch_size", 32))
    max_length = int(cfg["runtime"].get("max_length", 128))
    truncate_dim = int(cfg["runtime"].get("truncate_dim", 256))
    torch_dtype = resolve_torch_dtype(device, cfg["runtime"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
    ).eval().to(device)

    sentence_df = read_table(cfg["inputs"]["sentences_file"])
    sentence_col = cfg["inputs"].get("sentence_col", "sentence")
    sentences = sentence_df[sentence_col].dropna().astype(str).str.strip().tolist()
    sentence_emb = encode_texts(
        sentences, tokenizer, model, batch_size=batch_size, max_length=max_length, truncate_dim=truncate_dim
    )
    torch.save(
        {"sentences": sentences, "embeddings": sentence_emb},
        str(out_dir / "sentence_embeddings.pt"),
    )

    token_df = read_table(cfg["inputs"]["tokens_file"])
    token_key_col = cfg["inputs"].get("token_key_col", "key")
    token_exp_col = cfg["inputs"].get("token_explanation_col", "explanation")
    use_expansion = bool(cfg["inputs"].get("use_expansion_text", True))

    keys = token_df[token_key_col].dropna().astype(str).str.strip().tolist()
    key_to_text = {}
    if use_expansion and token_exp_col in token_df.columns:
        for _, row in token_df.iterrows():
            k = str(row.get(token_key_col, "")).strip()
            e = str(row.get(token_exp_col, "")).strip()
            if k:
                key_to_text[k] = e if e else k
    else:
        key_to_text = {k: k for k in keys}

    keys = [k for k in keys if k in key_to_text]
    texts = [key_to_text[k] for k in keys]
    word_emb = encode_texts(
        texts, tokenizer, model, batch_size=batch_size, max_length=max_length, truncate_dim=truncate_dim
    )
    torch.save(
        {"keys": keys, "texts": texts, "embeddings": word_emb},
        str(out_dir / "word_embeddings.pt"),
    )

    print(f"[OK] sentence_embeddings.pt: {len(sentences)} x {sentence_emb.shape[-1]}")
    print(f"[OK] word_embeddings.pt: {len(keys)} x {word_emb.shape[-1]}")
    print(f"[OK] model dtype: {torch_dtype}")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()
