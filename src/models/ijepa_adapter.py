"""
Simplified I-JEPA adapter for 10-band BigEarthNet-S2.

Architecture:
- Context encoder: ViT (vision transformer)
- Target encoder: EMA of context encoder (stop-gradient)
- Predictor: lightweight ViT that predicts target representations from context

Based on the I-JEPA paper (Assran et al., 2023) and adapted for 120x120x10 input.

AI-assist prompt:
"Implement a simplified I-JEPA model in PyTorch for 10-channel 120x120 images.
Include a Vision Transformer encoder, an EMA target encoder, a predictor network,
and the JEPA masking/target-block loss. Keep it modular for PyTorch Lightning."
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Patch embedding layer adapted for arbitrary input channels."""
    def __init__(self, img_size=120, patch_size=8, in_chans=10, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT encoder for I-JEPA."""
    def __init__(
        self,
        img_size=120,
        patch_size=8,
        in_chans=10,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
    ):
        super().__init__()
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_all_tokens=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if return_all_tokens:
            return x
        return x[:, 0]  # cls token


class Predictor(nn.Module):
    """Lightweight ViT predictor."""
    def __init__(self, embed_dim=384, predictor_embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        # x: (B, N, embed_dim) context representations
        x = self.predictor_embed(x)
        if mask is not None:
            B, N, C = x.shape
            mask_tokens = self.mask_token.expand(B, N, -1)
            w = mask.float().unsqueeze(-1)
            x = x * (1 - w) + mask_tokens * w
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.predictor_proj(x)
        return x


class IJEPA(nn.Module):
    """
    Full I-JEPA model: context encoder, target encoder (EMA), predictor, masking.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.context_encoder = VisionTransformer(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            drop_path_rate=config.get("drop_path_rate", 0.1),
        )
        self.target_encoder = VisionTransformer(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            drop_path_rate=config.get("drop_path_rate", 0.1),
        )
        # Initialize target encoder with context encoder weights
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        # Disable gradients for target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            embed_dim=config["embed_dim"],
            predictor_embed_dim=config["predictor"]["embed_dim"],
            depth=config["predictor"]["depth"],
            num_heads=config["predictor"]["num_heads"],
            mlp_ratio=config["predictor"].get("mlp_ratio", 4.0),
        )

    def forward_target(self, x, target_masks):
        """Extract target representations (no grad)."""
        with torch.no_grad():
            h = self.target_encoder(x, return_all_tokens=True)
            # h: (B, N+1, D); we ignore cls token at index 0
            h = h[:, 1:, :]  # (B, N, D)
            B = h.shape[0]
            # target_masks: list of (B, N) bool tensors, one per target block
            targets = []
            for mask in target_masks:
                # Average pooled representation of masked patches
                mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                target_rep = (h * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
                targets.append(target_rep)
            targets = torch.stack(targets, dim=1)  # (B, num_targets, D)
            if self.cfg.get("normalize_target", True):
                targets = F.layer_norm(targets, targets.shape[-1:])
        return targets

    def forward_context(self, x, context_mask):
        """Extract context representations and predict targets."""
        h = self.context_encoder(x, return_all_tokens=True)
        h = h[:, 1:, :]  # remove cls token, (B, N, D)
        B, N, D = h.shape
        # Apply context mask: set non-context patches to zero
        context_mask_expanded = context_mask.unsqueeze(-1).float()
        h = h * context_mask_expanded

        # Predict target representations for all positions
        pred = self.predictor(h)  # (B, N, D)
        return pred

    def forward(self, x, target_masks, context_mask):
        """
        x: (B, C, H, W)
        target_masks: list of (B, N) bool tensors
        context_mask: (B, N) bool tensor
        Returns loss scalar.
        """
        targets = self.forward_target(x, target_masks)
        pred = self.forward_context(x, context_mask)

        # Gather predicted representations for target patches
        B = x.shape[0]
        num_targets = len(target_masks)
        loss = 0
        for i, mask in enumerate(target_masks):
            # mask: (B, N)
            mask_expanded = mask.unsqueeze(-1).float()
            pred_target = (pred * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
            # L2 loss in predictor space
            loss += F.mse_loss(pred_target, targets[:, i, :])
        loss = loss / num_targets
        return loss

    def update_target_encoder(self, ema_decay):
        """Exponential moving average update of target encoder."""
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(ema_decay).add_((1. - ema_decay) * param_q.data)


class IJEPAFeatureExtractor(nn.Module):
    """Wrapper to extract fixed-size embeddings from the trained context encoder."""
    def __init__(self, context_encoder: VisionTransformer):
        super().__init__()
        self.encoder = context_encoder
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.encoder(x, return_all_tokens=False)
