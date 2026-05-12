"""
Smoke test: verify BigEarthNet metadata and dataloader shapes.

AI-assist prompt:
"Write a quick smoke test script that loads BigEarthNet metadata, reads 5 patches,
prints tensor shapes, and verifies single-label counts."
"""

import argparse
import json
from pathlib import Path

from _project_paths import project_paths, find_processed_dataset_dir

import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to a processed dataset directory containing metadata.json. If omitted, searches under data/processed/.",
    )
    args = parser.parse_args()

    paths = project_paths()
    data_dir = Path(args.data_dir) if args.data_dir else find_processed_dataset_dir(paths.processed_dir)

    meta_path = data_dir / "metadata.json"
    assert meta_path.exists(), f"Missing {meta_path}"

    with open(meta_path, "r") as f:
        data = json.load(f)

    patches = data["patches"]
    print(f"Total patches: {len(patches)}")
    print(f"Unique labels: {len(set(p['single_label'] for p in patches))}")

    # Read first 5 patches
    for i, p in enumerate(patches[:5]):
        patch_dir = Path(p["patch_dir"])
        tifs = sorted(patch_dir.glob("*.tif"))
        print(f"Patch {i}: {len(tifs)} TIFs, label={p['single_label']}")
        bands = []
        for tif in tifs:
            with rasterio.open(tif) as src:
                band = src.read(1)
                bands.append(band.shape)
        print(f"  Band shapes: {bands}")

    print("[smoke_data] PASSED")


if __name__ == "__main__":
    main()
