import argparse
import os
from pathlib import Path

from text_assets_common import DEFAULT_MODEL_PATH, build_dataset_assets


DEFAULT_REAL_DATA = Path(
    os.environ.get(
        "ZUCO_REAL_DATA_ROOT",
        "/home/share/huadjyin/home/tangwangyang/workspace/sunmengmeng/data/DIGnet/real_data",
    )
)


def parse_args():
    parser = argparse.ArgumentParser("Build ZuCo task text-side asset inputs and BrainMosaic configs")
    parser.add_argument("--task", required=True, choices=["ZuCoSR", "ZuCoNR", "ZuCoTSR"])
    parser.add_argument("--eeg-root", type=Path, default=None)
    parser.add_argument("--text-root", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--truncate-dim", type=int, default=256)
    parser.add_argument("--cluster-sim-threshold", type=float, default=0.78)
    parser.add_argument("--raw-token-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    eeg_root = (args.eeg_root or (DEFAULT_REAL_DATA / args.task)).resolve()
    text_root = (args.text_root or (eeg_root / "text_assets")).resolve()
    output_dir = args.output_dir or Path(f"outputs/{args.task.lower()}_full")

    sentence_counts, token_counts, split_sizes, tag = build_dataset_assets(
        eeg_root=eeg_root,
        text_root=text_root,
        config_dir=args.config_dir,
        output_dir=output_dir,
        dataset_name=args.task,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
        cluster_sim_threshold=args.cluster_sim_threshold,
        in_channels=105,
        raw_token_config=args.raw_token_config,
    )

    print(f"[OK] {args.task} text-side input files")
    print(f"  train records: {split_sizes.get('train', 0)}")
    print(f"  val records: {split_sizes.get('val', 0)}")
    print(f"  unique sentences: {len(sentence_counts)}")
    print(f"  unique tokens: {len(token_counts)}")
    print(f"  text root: {text_root}")
    print(f"  config dir: {args.config_dir.resolve()}")
    print(f"  config tag: {tag}")


if __name__ == "__main__":
    main()
