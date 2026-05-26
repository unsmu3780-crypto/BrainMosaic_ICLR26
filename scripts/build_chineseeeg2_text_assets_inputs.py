import argparse
import os
from pathlib import Path

import torch

from text_assets_common import DEFAULT_MODEL_PATH, build_dataset_assets


DEFAULT_EEG_ROOT = Path(
    os.environ.get(
        "CHINESEEEG2_EEG_ROOT",
        "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/ChineseEEG2",
    )
)


def infer_in_channels(eeg_root: Path) -> int:
    train_path = eeg_root / "train.pt"
    data = torch.load(train_path, map_location="cpu")
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list) or not data:
        raise ValueError(f"{train_path} does not contain non-empty list data")
    eeg = data[0]["eeg"]
    if getattr(eeg, "ndim", None) != 2:
        raise ValueError(f"Unexpected EEG shape in {train_path}: {getattr(eeg, 'shape', None)}")
    return int(eeg.shape[0])


def parse_args():
    parser = argparse.ArgumentParser(
        "Build ChineseEEG-2 text-side asset inputs and BrainMosaic configs"
    )
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--text-root", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chineseeeg2_full"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--truncate-dim", type=int, default=256)
    parser.add_argument("--cluster-sim-threshold", type=float, default=0.78)
    parser.add_argument(
        "--use-expanded-tokens",
        action="store_true",
        help="Point text_embedding.chineseeeg2.json at token_explanations.expanded.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eeg_root = args.eeg_root.resolve()
    text_root = (args.text_root or (eeg_root / "text_assets")).resolve()
    in_channels = infer_in_channels(eeg_root)
    sentence_counts, token_counts, split_sizes, tag = build_dataset_assets(
        eeg_root=eeg_root,
        text_root=text_root,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        dataset_name="ChineseEEG2",
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
        cluster_sim_threshold=args.cluster_sim_threshold,
        in_channels=in_channels,
        raw_token_config=not args.use_expanded_tokens,
    )

    print("[OK] ChineseEEG-2 text-side input files")
    print(f"  train records: {split_sizes.get('train', 0)}")
    print(f"  val records: {split_sizes.get('val', 0)}")
    print(f"  unique sentences: {len(sentence_counts)}")
    print(f"  unique tokens: {len(token_counts)}")
    print(f"  text root: {text_root}")
    print(f"  config dir: {args.config_dir.resolve()}")
    print(f"  config tag: {tag}")
    print(f"  in_channels: {in_channels}")
    print(
        "  token embedding source: "
        + ("token_explanations.expanded.json" if args.use_expanded_tokens else "token_explanations.json")
    )


if __name__ == "__main__":
    main()
