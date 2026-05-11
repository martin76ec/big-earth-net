"""
PyTorch Lightning DataModule for BigEarthNet-S2 single-label subset.

AI-assist prompt:
"Write a PyTorch Lightning DataModule that reads BigEarthNet 10-band TIF patches
from a JSON metadata file, applies per-band normalization, resizes all bands
to a common size, and supports train/val/test splits with stratification."
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset


def _read_patch(patch_dir: str, stats: dict, target_size: int = 120):
    """Read 10 bands, resize to target_size, normalize with precomputed mean/std."""
    tifs = sorted(Path(patch_dir).glob("*.tif"))
    if len(tifs) != 10:
        # Fallback: pad with zeros or skip
        raise ValueError(f"Expected 10 TIFs in {patch_dir}, found {len(tifs)}")
    bands = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            band = src.read(1).astype(np.float32)
            # Resize if necessary (simple nearest-neighbor for speed)
            if band.shape != (target_size, target_size):
                # Very simple resize via numpy repeat/slice; in production use cv2 or PIL
                h, w = band.shape
                if h != target_size or w != target_size:
                    # Use skimage or torch interpolate
                    band = torch.tensor(band).unsqueeze(0).unsqueeze(0)
                    band = torch.nn.functional.interpolate(
                        band, size=(target_size, target_size), mode="bilinear", align_corners=False
                    )
                    band = band.squeeze().numpy()
            bands.append(band)
    img = np.stack(bands, axis=0)  # (10, H, W)
    mean = np.array(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.where(std == 0, 1.0, std)
    img = (img - mean) / std
    return torch.from_numpy(img).float()


class BigEarthNetDataset(Dataset):
    def __init__(self, metadata: list, stats: dict, label_to_idx: dict, target_size: int = 120):
        self.metadata = metadata
        self.stats = stats
        self.label_to_idx = label_to_idx
        self.target_size = target_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img = _read_patch(item["patch_dir"], self.stats, self.target_size)
        label = self.label_to_idx[item["single_label"]]
        return img, label


class BigEarthNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        target_size: int = 120,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.stats = {}

    def setup(self, stage=None):
        with open(self.metadata_path, "r") as f:
            data = json.load(f)
        patches = data["patches"]
        self.stats = data["band_stats"]
        labels = [p["single_label"] for p in patches]
        unique_labels = sorted(set(labels))
        self.label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
        y = np.array([self.label_to_idx[lab] for lab in labels])

        # Stratified split: train / val / test
        indices = np.arange(len(patches))
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.seed)
        trainval_idx, test_idx = next(sss1.split(indices, y))
        y_trainval = y[trainval_idx]
        val_ratio_of_trainval = self.val_ratio / (1 - self.test_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_of_trainval, random_state=self.seed)
        train_idx, val_idx = next(sss2.split(trainval_idx, y_trainval))
        train_idx = trainval_idx[train_idx]
        val_idx = trainval_idx[val_idx]

        self.train_ds = BigEarthNetDataset(
            [patches[i] for i in train_idx], self.stats, self.label_to_idx, self.target_size
        )
        self.val_ds = BigEarthNetDataset(
            [patches[i] for i in val_idx], self.stats, self.label_to_idx, self.target_size
        )
        self.test_ds = BigEarthNetDataset(
            [patches[i] for i in test_idx], self.stats, self.label_to_idx, self.target_size
        )
        self.num_classes = len(unique_labels)
        print(f"[DataModule] Classes: {self.num_classes} | Train: {len(self.train_ds)} | Val: {len(self.val_ds)} | Test: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
