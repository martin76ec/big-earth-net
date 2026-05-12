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
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


EXPECTED_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def extract_tar_zst(tar_path: str, output_dir: str) -> str:
    """Extract tar.zst to output_dir. Return root folder inside."""
    os.makedirs(output_dir, exist_ok=True)
    existing_entries = [e for e in os.listdir(output_dir) if not e.startswith(".")]
    if existing_entries:
        print(f"[extract] Reusing existing extracted data in {output_dir}")
    else:
        print(f"[extract] Decompressing {tar_path} -> {output_dir} ...")
        cmd = f"zstd -d --stdout '{tar_path}' | tar -x -C '{output_dir}'"
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


def select_band_files(patch_dir: str):
    tif_paths = list(Path(patch_dir).glob("*.tif"))
    if not tif_paths:
        return []

    by_name = {tif.stem.upper(): tif for tif in tif_paths}
    selected = []
    for band in EXPECTED_BANDS:
        match = next((path for stem, path in by_name.items() if stem.endswith(f"_{band}") or stem == band), None)
        if match is None:
            selected = []
            break
        selected.append(match)

    if selected:
        return selected

    tif_paths = sorted(tif_paths)
    if len(tif_paths) == len(EXPECTED_BANDS):
        return tif_paths
    return []


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
        tifs = select_band_files(patch_dir)
        if len(tifs) != len(EXPECTED_BANDS):
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
    print(f"[extract] Found {len(patches)} total patch directories with labels.")

    metadata_all = []
    skipped_missing_bands = 0
    for patch_dir, json_path in tqdm(patches, desc="validating patches"):
        band_files = select_band_files(patch_dir)
        if len(band_files) != len(EXPECTED_BANDS):
            skipped_missing_bands += 1
            continue
        with open(json_path, "r") as f:
            raw = json.load(f)
        all_labels = raw.get("labels", [])
        metadata_all.append({
            "patch_dir": patch_dir,
            "label_json": json_path,
            "single_label": single_label_from_json(json_path, strategy=args.single_label_strategy),
            "all_labels": all_labels,
            "band_files": [str(path) for path in band_files],
        })

    print(f"[extract] Valid patches with expected bands: {len(metadata_all)}")
    if skipped_missing_bands:
        print(f"[extract] Skipped {skipped_missing_bands} patches with unexpected band files")

    if len(metadata_all) < args.max_patches:
        print(f"[WARN] Requested {args.max_patches} but only {len(metadata_all)} valid patches available. Using all.")
        metadata = metadata_all
    else:
        by_label = {}
        for item in metadata_all:
            by_label.setdefault(item["single_label"], []).append(item)

        target_total = args.max_patches
        selected = []
        label_names = sorted(by_label)
        base_per_label = max(1, target_total // max(1, len(label_names)))
        leftovers = []
        for label in label_names:
            items = by_label[label]
            take = min(len(items), base_per_label)
            selected.extend(random.sample(items, take))
            if len(items) > take:
                leftovers.extend(item for item in items if item not in selected)

        if len(selected) < target_total:
            remaining = target_total - len(selected)
            selected.extend(random.sample(leftovers, min(remaining, len(leftovers))))
        metadata = selected[:target_total]

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
            "expected_bands": EXPECTED_BANDS,
        }, f, indent=2)
    print(f"[save] Metadata written to {meta_path}")
    print(f"[save] Total patches: {len(metadata)}")
    print(f"[save] Unique labels: {len(set(m['single_label'] for m in metadata))}")


if __name__ == "__main__":
    main()
