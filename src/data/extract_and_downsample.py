"""
Extract BigEarthNet-S2 from tar.zst, downsample to N patches,
and convert multi-label patches to single-label using the first label.

AI-assist prompt:
"Write a robust Python script to extract a tar.zst archive containing BigEarthNet-S2,
find all patch folders, parse their label JSONs, convert multi-label to single-label
by picking the first label, randomly downsample to a user-specified count,
and save a metadata CSV/JSON with paths and single labels."
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


def extract_tar_zst(tar_path: str, output_dir: str) -> str:
    """Extract tar.zst to output_dir. Return root folder inside."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[extract] Decompressing {tar_path} -> {output_dir} ...")
    # zstd -d --stdout BigEarthNet-S2.tar.zst | tar -x -C output_dir
    # Using tar with --zstd if available, else pipe
    cmd = (
        f"zstd -d --stdout '{tar_path}' | tar -x -C '{output_dir}'"
    )
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Extraction failed with code {ret}")

    # Determine the actual root folder inside output_dir
    entries = [e for e in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, e))]
    if len(entries) == 1:
        root = os.path.join(output_dir, entries[0])
    else:
        root = output_dir
    print(f"[extract] Root folder: {root}")
    return root


def find_patches(root: str):
    """Yield (patch_dir, label_json_path) for each patch."""
    for patch_dir in Path(root).rglob("*"):
        if not patch_dir.is_dir():
            continue
        json_files = list(patch_dir.glob("*.json"))
        if not json_files:
            continue
        # BigEarthNet has one JSON per patch
        yield str(patch_dir), str(json_files[0])


def single_label_from_json(json_path: str, strategy: str = "first") -> str:
    """Return a single label string from a BigEarthNet label JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    labels = data.get("labels", [])
    if not labels:
        return "unknown"
    if strategy == "first":
        return labels[0]
    # Could add 'majority' if we had a mapping table
    return labels[0]


def compute_band_stats(patch_dirs: list, max_patches: int = 500) -> dict:
    """Compute per-band mean and std over a subset of patches."""
    print(f"[stats] Computing band stats on up to {max_patches} patches...")
    sums = np.zeros(10, dtype=np.float64)
    sumsq = np.zeros(10, dtype=np.float64)
    count = 0
    sample = patch_dirs if len(patch_dirs) <= max_patches else random.sample(patch_dirs, max_patches)
    for patch_dir in tqdm(sample, desc="band stats"):
        tifs = sorted(Path(patch_dir).glob("*.tif"))
        if len(tifs) != 10:
            continue
        bands = []
        for tif in tifs:
            with rasterio.open(tif) as src:
                band = src.read(1).astype(np.float32)
                bands.append(band)
        img = np.stack(bands, axis=0)  # (10, H, W)
        sums += img.reshape(10, -1).mean(axis=1)
        sumsq += (img ** 2).reshape(10, -1).mean(axis=1)
        count += 1
    mean = sums / count
    std = np.sqrt(sumsq / count - mean ** 2)
    return {"mean": mean.tolist(), "std": std.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_patches", type=int, default=25000)
    parser.add_argument("--single_label_strategy", default="first", choices=["first"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Step 0: extract
    extracted_root = extract_tar_zst(args.tar_path, args.output_dir + "_raw")

    # Step 1: enumerate patches
    patches = list(find_patches(extracted_root))
    print(f"[extract] Found {len(patches)} total patches.")

    if len(patches) < args.max_patches:
        print(f"[WARN] Requested {args.max_patches} but only {len(patches)} available. Using all.")
        selected = patches
    else:
        selected = random.sample(patches, args.max_patches)

    # Step 2: build metadata
    metadata = []
    for patch_dir, json_path in tqdm(selected, desc="parsing labels"):
        label = single_label_from_json(json_path, strategy=args.single_label_strategy)
        metadata.append({
            "patch_dir": patch_dir,
            "label_json": json_path,
            "single_label": label,
        })

    # Step 3: compute stats
    stats = compute_band_stats([m["patch_dir"] for m in metadata], max_patches=500)

    # Step 4: save
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "patches": metadata,
            "band_stats": stats,
            "num_patches": len(metadata),
            "strategy": args.single_label_strategy,
        }, f, indent=2)
    print(f"[save] Metadata written to {meta_path}")
    print(f"[save] Total patches: {len(metadata)}")
    print(f"[save] Unique labels: {len(set(m['single_label'] for m in metadata))}")


if __name__ == "__main__":
    main()
