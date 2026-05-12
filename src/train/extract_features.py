"""
Extract fixed-size embeddings from trained I-JEPA or ResNet-50.

AI-assist prompt:
"Write a script that loads a trained model checkpoint, runs inference on a dataset,
and saves embeddings and labels as numpy arrays."
"""

import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.bigearth_datamodule import BigEarthNetDataset
from src.models.ijepa_adapter import IJEPA, IJEPAFeatureExtractor, VisionTransformer
from src.models.resnet_baseline import ResNet50FeatureExtractor


@torch.no_grad()
def extract(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings = []
    labels = []
    for x, y in dataloader:
        x = x.to(device)
        z = model(x)
        embeddings.append(z.cpu().numpy())
        labels.append(y.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["ijepa", "resnet50"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--config", default="configs/ijepa_small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    meta_path = f"{args.data_dir}/metadata.json"
    with open(meta_path, "r") as f:
        data = json.load(f)
    patches = data["patches"]
    stats = data["band_stats"]
    unique_labels = sorted(set(p["single_label"] for p in patches))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    dataset = BigEarthNetDataset(patches, stats, label_to_idx, target_size=120)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.model_type == "ijepa":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for I-JEPA")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg["model"]
        context_encoder = VisionTransformer(**{k: v for k, v in model_cfg.items() if k in {
            "img_size","patch_size","in_chans","embed_dim","depth","num_heads","mlp_ratio","drop_path_rate"
        }})
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Load context encoder weights from Lightning checkpoint
        state_dict = ckpt.get("state_dict", ckpt)
        encoder_state = {k.replace("model.context_encoder.", ""): v for k, v in state_dict.items() if k.startswith("model.context_encoder.")}
        context_encoder.load_state_dict(encoder_state, strict=False)
        model = IJEPAFeatureExtractor(context_encoder)
    elif args.model_type == "resnet50":
        model = ResNet50FeatureExtractor(in_chans=10, pretrained=True)
    else:
        raise ValueError(args.model_type)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    embeddings, labels = extract(model, dataloader, device)
    np.savez(args.output, embeddings=embeddings, labels=labels, label_names=unique_labels)
    print(f"[extract] Saved {embeddings.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
