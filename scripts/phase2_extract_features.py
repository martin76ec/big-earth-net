from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _project_paths import project_paths, find_processed_dataset_dir, find_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2/3: extract embeddings for downstream classifier.")
    parser.add_argument("--model", required=True, choices=["ijepa", "resnet50"])
    parser.add_argument("--smoke", action="store_true", help="Write smoke embeddings (separate filenames)")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Processed dataset directory. If omitted, searches under data/processed/.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="I-JEPA Lightning checkpoint. If omitted, searches under checkpoints/ijepa/.",
    )
    parser.add_argument("--config", default="configs/ijepa_small.yaml")
    args = parser.parse_args()

    paths = project_paths()
    data_dir = Path(args.data_dir) if args.data_dir else find_processed_dataset_dir(paths.processed_dir)

    out_name = {
        "ijepa": "ijepa_embeddings_smoke.npz" if args.smoke else "ijepa_embeddings.npz",
        "resnet50": "resnet50_embeddings_smoke.npz" if args.smoke else "resnet50_embeddings.npz",
    }[args.model]
    out_path = paths.processed_dir / out_name

    cmd = [
        sys.executable,
        "-m",
        "src.train.extract_features",
        "--model_type",
        args.model,
        "--data_dir",
        str(data_dir),
        "--output",
        str(out_path),
        "--config",
        str(paths.root / args.config),
    ]

    if args.model == "ijepa":
        ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint(paths.root)
        cmd += ["--checkpoint", str(ckpt)]

    print(f"[phase2] model={args.model}", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
