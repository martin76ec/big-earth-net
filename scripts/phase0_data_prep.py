from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _project_paths import project_paths, find_best_tar_zst


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: extract + downsample BigEarthNet into data/processed.")
    parser.add_argument("--smoke", action="store_true", help="Create a small processed dataset for quick validation")
    parser.add_argument(
        "--tar_path",
        default=None,
        help="Path to BigEarthNet-S2.tar.zst. If omitted, searches under data/.",
    )
    args = parser.parse_args()

    paths = project_paths()
    tar_path = Path(args.tar_path) if args.tar_path else find_best_tar_zst(paths.data_dir)

    output_dir = paths.processed_dir / ("bigearthnet_smoke" if args.smoke else "bigearthnet_25k_singlelabel")
    max_patches = 200 if args.smoke else 25000

    cmd = [
        sys.executable,
        "-m",
        "src.data.extract_and_downsample",
        "--tar_path",
        str(tar_path),
        "--output_dir",
        str(output_dir),
        "--max_patches",
        str(max_patches),
        "--single_label_strategy",
        "first",
    ]
    print("[phase0]", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
