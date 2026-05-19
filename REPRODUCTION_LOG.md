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
python labels/gen_embedding.py --config configs/text_embedding.chisco.json
python labels/emb_preprocessing.py --config configs/token_bank.chisco.json
python main.py --config configs/train.chisco.json
```

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
  - `scratch/inspect_logs/*`
  - `smoke_brainmosaic.sh`

