"""
Smoke test: verify I-JEPA forward pass and loss computation on a dummy batch.

AI-assist prompt:
"Write a smoke test that instantiates the I-JEPA model, runs one forward pass
on synthetic 10-channel data, checks that loss is a scalar and backward works."
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Ensure project root is on sys.path when running `python scripts/...`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ijepa_adapter import IJEPA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ijepa_small.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    for k, v in cfg.get("ijepa", {}).items():
        model_cfg[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IJEPA(model_cfg).to(device)
    B = 4
    x = torch.randn(B, 10, 120, 120, device=device)
    num_patches = model.context_encoder.patch_embed.num_patches
    num_target_blocks = model_cfg.get("target_blocks", 4)
    block_size = max(1, int(num_patches * model_cfg.get("target_block_scale", 0.15)))
    context_size = max(1, int(num_patches * model_cfg.get("context_scale", 0.5)))
    target_masks = [
        torch.stack([torch.randperm(num_patches, device=device)[:block_size].sort().values for _ in range(B)])
        for _ in range(num_target_blocks)
    ]
    context_mask = torch.stack([
        torch.randperm(num_patches, device=device)[:context_size].sort().values
        for _ in range(B)
    ])

    loss = model(x, target_masks, context_mask)
    assert loss.dim() == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()
    print(f"[smoke_ijepa] Loss: {loss.item():.4f}")
    print("[smoke_ijepa] PASSED")


if __name__ == "__main__":
    main()
