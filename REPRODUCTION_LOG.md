# BrainMosaic Reproduction Log

This log records what data was used, what method/script was run, what was produced, and where outputs are stored.

## 2026-05-19

### Current Objective

Reproduce "Assembling the Mind's Mosaic: Towards EEG Semantic Intent Decoding" with the public datasets. The current active dataset is Chisco.

### Completed: Chisco EEG-Side Conversion

- Data used:
  - Raw Chisco root: `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/Chiso`
  - EEG files: 225 EDF files
  - Text tables: `textdataset/split_data_*.xlsx`
  - Text table columns observed: `句子`, `标签`
- Code used:
  - `scripts/prepare_chisco_full.py`
  - Inspection helpers under `scratch/inspect_logs/`
- Method:
  - Load each Chisco EDF file with MNE.
  - Keep EEG channels.
  - Resample EEG to 250 Hz.
  - Use EDF annotations as trial onsets.
  - For each trial, cut a 4-second EEG window.
  - Align trial index with the matching sentence row in `split_data_<run>.xlsx`.
  - Segment Chinese sentences with jieba.
  - Assign provisional sentence attributes with heuristic rules:
    - `sentence_mode`
    - `subjectivity`
    - `semantic_focus`
  - Save BrainMosaic-compatible PyTorch split files.
- Result:
  - EDF files processed: 225
  - Records generated: 32405
  - Files skipped: 0
  - Train records: 25924
  - Validation records: 6481
  - Example EEG shape: `[132, 1000]`
  - Example dtype: `torch.float16`
- Outputs:
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/train.pt`
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/val.pt`
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/meta.pt`
- Status:
  - Chisco EEG-side conversion is complete.
  - Chisco has not yet been verified through the full BrainMosaic training path.

### Added: Chisco Text-Side Asset Input Builder

- Code added:
  - `scripts/build_chisco_text_assets_inputs.py`
- Purpose:
  - Read the converted Chisco `train.pt` and `val.pt`.
  - Extract unique sentences for sentence embedding generation.
  - Extract token vocabulary from `words` for token embedding generation.
  - Write segmentation metadata for BrainMosaic dataset loading.
  - Generate Chisco-specific configs for the existing author-provided embedding, token-bank, and training scripts.
- Inputs:
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/train.pt`
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/val.pt`
- Expected outputs:
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets/sentences.csv`
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets/token_explanations.json`
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets/segmentation.json`
  - `configs/text_embedding.chisco.json`
  - `configs/token_bank.chisco.json`
  - `configs/train.chisco.json`
- Validation performed locally:
  - `python -m py_compile scripts/build_chisco_text_assets_inputs.py`
  - Full execution was not run locally because the local Windows environment does not have PyTorch installed and does not contain the server-side Chisco `.pt` files.

### Next Commands on Server

Run from the repository root on the server:

```bash
python scripts/build_chisco_text_assets_inputs.py
python scripts/expand_chisco_tokens.py --backend template
python labels/gen_embedding.py --config configs/text_embedding.chisco.json
python labels/emb_preprocessing.py --config configs/token_bank.chisco.json
python main.py --config configs/train.chisco.json
```

After running token expansion, update `configs/text_embedding.chisco.json` so
`inputs.tokens_file` points to:

```text
/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets/token_explanations.expanded.json
```

### Added: Chisco Token Expansion Step

- Code added:
  - `scripts/expand_chisco_tokens.py`
  - `scripts/run_chisco_text_pipeline.sh`
- Motivation:
  - The paper describes expanding semantic units into short explanation phrases before embedding to reduce ambiguity and improve generalization.
  - The first Chisco input builder used the raw token itself as its explanation. That is enough to run the public pipeline, but less close to the paper.
- Method:
  - Read `token_explanations.json` and `segmentation.json`.
  - Collect representative sentence contexts for each token.
  - Produce `token_explanations.expanded.json`, preserving the same `key` field expected by `labels/gen_embedding.py`.
  - Two backends are supported:
    - `template`: offline deterministic context phrase, no external model call.
    - `openai-compatible`: call a chat model through an OpenAI-compatible API, with template fallback on API/network errors.
  - The Chisco config builder now points `configs/text_embedding.chisco.json` to `token_explanations.expanded.json` by default, so regenerated configs follow the paper-like token-expansion path.
  - The Chisco config builder now defaults `model.name_or_path` to the local server model path:
    `/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/models/Qwen3-Embedding-8B`.
- Expected output:
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets/token_explanations.expanded.json`
- Recommended first run:

```bash
bash scripts/run_chisco_text_pipeline.sh template
```

- Optional LLM run, if an OpenAI-compatible endpoint is available:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export EXPANSION_MODEL=MODEL_NAME
bash scripts/run_chisco_text_pipeline.sh openai-compatible
```

- Status:
  - Token expansion script Chinese prompt/template was repaired after a local mojibake check.
  - Full paper-like text-side run should be rerun on the server so `word_embeddings.pt` is generated from expanded token explanations.

### Chisco Basic Text Embedding Run

- Command run on server:
  - `python labels/gen_embedding.py --config configs/text_embedding.chisco.json`
- Data used:
  - `sentences.csv`
  - `token_explanations.json` or the then-current `tokens_file` in `configs/text_embedding.chisco.json`
  - Local Qwen3 model under `models/Qwen3-Embedding-8B`
- Method:
  - Qwen3-Embedding-8B text encoding.
  - Mean pooling over the final hidden states.
  - Truncate vectors to 256 dimensions.
  - L2 normalize embeddings.
- Result:
  - `sentence_embeddings.pt`: `6567 x 256`
  - `word_embeddings.pt`: `5869 x 256`
- Output:
  - `/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco/text_assets`
- Note:
  - For the paper-like reproduction, rerun the text pipeline after generating `token_explanations.expanded.json`, then rebuild the token bank.

### Added: Chisco Token-Bank Threshold Sweep and Training Job

- Code added:
  - `scripts/run_chisco_threshold_sweep_and_train.sh`
  - `scripts/dsub_chisco_threshold_sweep_and_train.sh`
- Motivation:
  - The public repository provides `cluster_sim_threshold=0.78` in `configs/token_bank.example.json` and uses `0.78` as the code default, but no stronger paper-level fixed threshold was found in the public docs.
  - The first Chisco token bank produced `187` clusters from `5869` expanded token embeddings, so threshold sensitivity should be checked before long training.
- Method:
  - Sweep token-bank thresholds with default values: `0.78 0.85 0.90 0.95`.
  - For each threshold:
    - Write a threshold-specific token-bank config under `configs/token_bank.chisco.t*.json`.
    - Build a threshold-specific token bank under `text_assets/token_bank_t*/`.
    - Write a threshold-specific smoke training config under `configs/train.chisco.smoke.t*.json`.
    - Run short training/validation, default `1` epoch with batch size `2`.
    - Save summary JSON/TSV under `outputs/chisco_threshold_sweep/`.
  - If `RUN_FULL=1`, select the best smoke result by matching accuracy and mean cosine, write `configs/train.chisco.full.t*.json`, and start full training.
- Recommended server command:

```bash
bash scripts/run_chisco_threshold_sweep_and_train.sh
```

- Recommended DSUB submission:

```bash
dsub -s scripts/dsub_chisco_threshold_sweep_and_train.sh
```

- Useful overrides:

```bash
RUN_FULL=0 bash scripts/run_chisco_threshold_sweep_and_train.sh
THRESHOLDS="0.78 0.85 0.90 0.95" SMOKE_EPOCHS=1 RUN_FULL=0 bash scripts/run_chisco_threshold_sweep_and_train.sh
BEST_THRESHOLD=0.90 FULL_EPOCHS=50 FULL_BATCH_SIZE=32 bash scripts/run_chisco_threshold_sweep_and_train.sh
RUN_FULL=0 dsub -s scripts/dsub_chisco_threshold_sweep_and_train.sh
BEST_THRESHOLD=0.90 FULL_EPOCHS=50 FULL_BATCH_SIZE=32 dsub -s scripts/dsub_chisco_threshold_sweep_and_train.sh
```

- Expected outputs:
  - `outputs/chisco_threshold_sweep/sweep_summary.json`
  - `outputs/chisco_threshold_sweep/sweep_summary.tsv`
  - `outputs/chisco_threshold_sweep/t*.summary.json`
  - `outputs/chisco_full/t*/epoch_history.json`
  - `outputs/chisco_full/t*/best_summary.json`

Recommended first training check:

- Set `epochs` to `1` in `configs/train.chisco.json`.
- Set `batch_size` to `2` or `4`.
- Confirm that `datasets/eeg.py` can load Chisco and that `main.py` starts training without shape/path errors.

### Code Ownership Notes

- Author-provided code:
  - `datasets/eeg.py`
  - `labels/gen_embedding.py`
  - `labels/emb_preprocessing.py`
  - `main.py`
  - `engine.py`
  - `models/`
  - `semantic_guided_decoder/`
- Reproduction-added code:
  - `scripts/prepare_chisco_full.py`
  - `scripts/prepare_chisco_minimal.py`
  - `scripts/build_chisco_text_assets_inputs.py`
  - `scripts/expand_chisco_tokens.py`
  - `scratch/inspect_logs/*`
  - `smoke_brainmosaic.sh`

### Environment Compatibility: Qwen3 on ARM Torch 2.0

- Context:
  - The remote server is `aarch64`/ARM64, while the initially downloaded PyTorch wheels were `linux_x86_64` and cannot be installed there.
  - The current server environment has `torch 2.0.0+cu118`, which can use CUDA 11.8 but does not expose `torch.compiler`.
  - The upgraded `transformers` version needed by Qwen3 imports model code that expects `torch.compiler`.
- Code changed:
  - `labels/gen_embedding.py`
- Method:
  - Add a minimal `torch.compiler` shim before importing `transformers`.
  - The shim only provides `disable()` and `is_compiling()`, enough for model import paths that check the API.
  - Add missing float8 dtype names expected by newer `transformers` import-time dtype tables, mapped to `torch.uint8` only to allow import on `torch 2.0.0`.
  - Patch `torch.nn.Module.load_state_dict` to emulate the newer `assign` keyword when running on `torch 2.0.0`, replacing meta parameters with checkpoint tensors before the normal strictness checks.
  - Load Qwen3 with `low_cpu_mem_usage=False` to avoid the meta-tensor loading path that requires `assign=True`.
  - Tensor computation, CUDA execution, model loading, mean pooling, 256-d truncation, and L2 normalization are unchanged.
- Expected effect:
  - Allow `python labels/gen_embedding.py --config configs/text_embedding.chisco.json` to proceed on the current ARM server environment without upgrading PyTorch.
- Validation:
  - Local syntax check passed with `python -m py_compile labels/gen_embedding.py`.
