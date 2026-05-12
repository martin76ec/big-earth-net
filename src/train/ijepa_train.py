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
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.models.ijepa_adapter import IJEPA


def sample_target_block_indices(num_patches, target_scale):
    """Sample a contiguous square target block and return patch indices."""
    grid_size = int(math.sqrt(num_patches))
    block_h = max(1, int(grid_size * math.sqrt(target_scale)))
    block_w = max(1, int(grid_size * math.sqrt(target_scale)))
    block_h = min(block_h, grid_size)
    block_w = min(block_w, grid_size)
    top = random.randint(0, grid_size - block_h)
    left = random.randint(0, grid_size - block_w)
    indices = []
    for row in range(top, top + block_h):
        for col in range(left, left + block_w):
            indices.append(row * grid_size + col)
    return torch.tensor(indices, dtype=torch.long)


def sample_context_indices(num_patches, context_scale, target_masks):
    """Sample context patch indices from the complement of the target blocks."""
    available = torch.ones(num_patches, dtype=torch.bool)
    for target_mask in target_masks:
        available[target_mask] = False
    available_indices = torch.nonzero(available, as_tuple=False).squeeze(1)
    if available_indices.numel() == 0:
        return torch.arange(num_patches, dtype=torch.long)

    desired = max(1, int(num_patches * context_scale))
    count = min(desired, available_indices.numel())
    chosen = available_indices[torch.randperm(available_indices.numel())[:count]]
    return chosen.sort().values


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

        # Sample target/context indices using the reference I-JEPA masking pattern.
        target_masks_per_image = []
        context_masks_per_image = []
        for _ in range(B):
            targets_for_image = [sample_target_block_indices(self.num_patches, self.target_block_scale) for _ in range(self.num_target_blocks)]
            context_for_image = sample_context_indices(self.num_patches, self.context_scale, targets_for_image)
            target_masks_per_image.append(targets_for_image)
            context_masks_per_image.append(context_for_image)

        context_width = min(mask.numel() for mask in context_masks_per_image)
        context_masks = torch.stack([mask[:context_width] for mask in context_masks_per_image]).to(device)

        target_masks = []
        for target_idx in range(self.num_target_blocks):
            block_width = min(target_masks_per_image[sample_idx][target_idx].numel() for sample_idx in range(B))
            stacked = torch.stack([
                target_masks_per_image[sample_idx][target_idx][:block_width]
                for sample_idx in range(B)
            ])
            target_masks.append(stacked.to(device))

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

        warmup_epochs = self.cfg.get("warmup_epochs", 10)
        total_epochs = self.cfg["epochs"]
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


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
    model_cfg["epochs"] = train_cfg["epochs"]
    model_cfg["lr"] = train_cfg["lr"]
    model_cfg["weight_decay"] = train_cfg["weight_decay"]
    model_cfg["warmup_epochs"] = train_cfg.get("warmup_epochs", 10)
    # Merge I-JEPA hparams into model_cfg
    for k, v in cfg.get("ijepa", {}).items():
        model_cfg[k] = v

    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.epochs:
        train_cfg["epochs"] = args.epochs
        model_cfg["epochs"] = args.epochs

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
        filename="ijepa-best",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=True,
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
