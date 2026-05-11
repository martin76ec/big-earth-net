"""
PyTorch Lightning training module for I-JEPA self-supervised pretraining.

AI-assist prompt:
"Write a PyTorch Lightning module that trains an I-JEPA model with EMA updates,
learning rate warmup and cosine decay, mixed precision, and checkpointing."
"""

import math
import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.ijepa_adapter import IJEPA


def sample_target_block(num_patches, target_scale):
    """Sample a contiguous square block of patches."""
    H = W = int(math.sqrt(num_patches))
    block_h = max(1, int(H * math.sqrt(target_scale)))
    block_w = max(1, int(W * math.sqrt(target_scale)))
    top = random.randint(0, H - block_h)
    left = random.randint(0, W - block_w)
    mask = torch.zeros(H * W, dtype=torch.bool)
    for i in range(top, top + block_h):
        for j in range(left, left + block_w):
            idx = i * W + j
            if idx < num_patches:
                mask[idx] = True
    return mask


def sample_context_mask(num_patches, context_scale, target_masks):
    """Sample context mask ensuring minimal overlap with target blocks."""
    # Start with random sampling
    n_keep = int(num_patches * context_scale)
    indices = torch.randperm(num_patches)[:n_keep]
    mask = torch.zeros(num_patches, dtype=torch.bool)
    mask[indices] = True
    # Remove target patches from context
    for tm in target_masks:
        mask = mask & (~tm)
    return mask


class IJEPAModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.model = IJEPA(config)
        self.ema_decay = config.get("ema_decay", 0.996)
        self.ema_end_decay = config.get("ema_end_decay", 1.0)
        self.num_target_blocks = config.get("target_blocks", 4)
        self.target_block_scale = config.get("target_block_scale", 0.15)
        self.context_scale = config.get("context_scale", 0.85)
        self.num_patches = self.model.context_encoder.patch_embed.num_patches
        self.save_hyperparameters(config)

    def forward(self, x, target_masks, context_mask):
        return self.model(x, target_masks, context_mask)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # labels unused in SSL
        B = x.shape[0]
        device = x.device

        # Sample target blocks and context mask per image
        target_masks = []
        for _ in range(self.num_target_blocks):
            masks = torch.stack([sample_target_block(self.num_patches, self.target_block_scale) for _ in range(B)])
            target_masks.append(masks.to(device))

        context_masks = torch.stack([sample_context_mask(self.num_patches, self.context_scale, [tm[i] for tm in target_masks]) for i in range(B)]).to(device)

        loss = self.model(x, target_masks, context_masks)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        # EMA update
        self.model.update_target_encoder(self.ema_decay)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.context_encoder.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        # Also add predictor params
        optimizer.add_param_group({"params": self.model.predictor.parameters()})

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.get("warmup_epochs", 10) * self.trainer.num_training_batches

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def main():
    import argparse
    from src.data.bigearth_datamodule import BigEarthNetDataModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/ijepa")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    # Merge I-JEPA hparams into model_cfg
    for k, v in cfg.get("ijepa", {}).items():
        model_cfg[k] = v

    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.epochs:
        train_cfg["epochs"] = args.epochs

    pl.seed_everything(args.seed)

    dm = BigEarthNetDataModule(
        metadata_path=f"{args.data_dir}/metadata.json",
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 8),
        target_size=model_cfg["img_size"],
        seed=args.seed,
    )
    dm.setup()

    module = IJEPAModule(model_cfg)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="ijepa_{epoch:02d}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=True,
        every_n_epochs=train_cfg.get("save_every_n_epochs", 10),
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="gpu",
        devices=1,
        precision="16-mixed" if train_cfg.get("mixed_precision", True) else 32,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=1.0,
    )

    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
