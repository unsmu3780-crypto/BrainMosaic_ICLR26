import argparse
import os
from pathlib import Path

from text_assets_common import DEFAULT_MODEL_PATH, build_dataset_assets

CHISCO_MODEL_PATH = os.environ.get("CHISCO_EMBEDDING_MODEL", DEFAULT_MODEL_PATH)
DEFAULT_EEG_ROOT = Path(
    os.environ.get(
        "CHISCO_EEG_ROOT",
        "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data/Chisco",
    )
)

def parse_args():
    parser = argparse.ArgumentParser(
        "Build Chisco text-side asset inputs and BrainMosaic configs"
    )
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--text-root", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chisco"))
    parser.add_argument("--model-name", default=CHISCO_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--truncate-dim", type=int, default=256)
    parser.add_argument("--cluster-sim-threshold", type=float, default=0.78)
    parser.add_argument(
        "--raw-token-config",
        action="store_true",
        help="Point text_embedding.chisco.json at token_explanations.json instead of token_explanations.expanded.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eeg_root = args.eeg_root.resolve()
    text_root = (args.text_root or (eeg_root / "text_assets")).resolve()
    sentence_counts, token_counts, split_sizes, _ = build_dataset_assets(
        eeg_root=eeg_root,
        text_root=text_root,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        dataset_name="Chisco",
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
        cluster_sim_threshold=args.cluster_sim_threshold,
        in_channels=132,
        raw_token_config=args.raw_token_config,
    )

    print("[OK] Chisco text-side input files")
    print(f"  train records: {split_sizes.get('train', 0)}")
    print(f"  val records: {split_sizes.get('val', 0)}")
    print(f"  unique sentences: {len(sentence_counts)}")
    print(f"  unique tokens: {len(token_counts)}")
    print(f"  text root: {text_root}")
    print(f"  config dir: {args.config_dir.resolve()}")
    print(f"  embedding model: {args.model_name}")
    print(
        "  token embedding source: "
        + ("token_explanations.expanded.json" if not args.raw_token_config else "token_explanations.json")
    )
    print("\nNext commands:")
    if not args.raw_token_config:
        print("  python scripts/expand_chisco_tokens.py --backend template --resume")
    print("  python labels/gen_embedding.py --config configs/text_embedding.chisco.json")
    print("  python labels/emb_preprocessing.py --config configs/token_bank.chisco.json")
    print("  python main.py --config configs/train.chisco.json")


if __name__ == "__main__":
    main()
