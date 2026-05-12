from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _project_paths import project_paths, find_processed_dataset_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: I-JEPA pretraining.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny training job")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Processed dataset directory. If omitted, searches under data/processed/.",
    )
    parser.add_argument("--config", default="configs/ijepa_small.yaml")
    args = parser.parse_args()

    paths = project_paths()
    data_dir = Path(args.data_dir) if args.data_dir else find_processed_dataset_dir(paths.processed_dir)
    ckpt_dir = paths.root / "checkpoints" / "ijepa"

    cmd = [
        sys.executable,
        "-m",
        "src.train.ijepa_train",
        "--config",
        str(paths.root / args.config),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(ckpt_dir),
    ]

    if args.smoke:
        cmd += ["--batch_size", "16", "--epochs", "1"]

    print("[phase1]", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
