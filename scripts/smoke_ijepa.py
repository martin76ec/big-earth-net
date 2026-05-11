"""
Smoke test: verify I-JEPA forward pass and loss computation on a dummy batch.

AI-assist prompt:
"Write a smoke test that instantiates the I-JEPA model, runs one forward pass
on synthetic 10-channel data, checks that loss is a scalar and backward works."
"""

import argparse

import torch
import yaml

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

    model = IJEPA(model_cfg).cuda()
    B = 4
    x = torch.randn(B, 10, 120, 120).cuda()
    num_patches = model.context_encoder.patch_embed.num_patches
    target_masks = [torch.randint(0, 2, (B, num_patches), dtype=torch.bool).cuda() for _ in range(4)]
    context_mask = torch.randint(0, 2, (B, num_patches), dtype=torch.bool).cuda()

    loss = model(x, target_masks, context_mask)
    assert loss.dim() == 0, "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()
    print(f"[smoke_ijepa] Loss: {loss.item():.4f}")
    print("[smoke_ijepa] PASSED")


if __name__ == "__main__":
    main()
