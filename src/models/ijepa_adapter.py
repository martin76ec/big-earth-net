"""
Reference-style I-JEPA adaptation for 10-band BigEarthNet-S2.

This module adapts the public facebookresearch/ijepa architecture patterns:
- fixed 2D sin-cos positional embeddings
- masked token gathering for context/target blocks
- separate predictor operating on context tokens plus mask tokens
- EMA target encoder updated outside gradient flow

The implementation is kept self-contained so it can run inside the course repo
without pulling the upstream package at runtime.

AI-assist prompt:
"Adapt the public I-JEPA reference architecture into a self-contained PyTorch
module for 10-channel 120x120 BigEarthNet patches, preserving the reference
encoder/predictor/masking flow while avoiding an external runtime dependency."
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0.0, std=1.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_masks(x, masks):
    all_x = []
    for mask in masks:
        index = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=index))
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, batch_size, repeat):
    chunks = torch.chunk(x, len(x) // batch_size, dim=0)
    repeated = []
    for chunk in chunks:
        repeated.extend([chunk] * repeat)
    return torch.cat(repeated, dim=0)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=120, patch_size=8, in_chans=10, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=120,
        patch_size=8,
        in_chans=10,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i], norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, masks=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]
            x = apply_masks(x, masks)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        embed_dim=384,
        predictor_embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, int(num_patches**0.5))
        self.predictor_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dpr[i], norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, masks_x, masks):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]

        batch_size = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        pos_x = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        x = x + apply_masks(pos_x, masks_x)

        _, num_context, _ = x.shape
        pos_targets = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_targets = apply_masks(pos_targets, masks)
        pos_targets = repeat_interleave_batch(pos_targets, batch_size, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_targets.size(0), pos_targets.size(1), 1)
        pred_tokens = pred_tokens + pos_targets

        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, num_context:]
        return self.predictor_proj(x)


class IJEPA(nn.Module):
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
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = VisionTransformerPredictor(
            num_patches=self.context_encoder.patch_embed.num_patches,
            embed_dim=config["embed_dim"],
            predictor_embed_dim=config["predictor"]["embed_dim"],
            depth=config["predictor"]["depth"],
            num_heads=config["predictor"]["num_heads"],
            mlp_ratio=config["predictor"].get("mlp_ratio", 4.0),
        )

    def forward_target(self, x, target_masks):
        with torch.no_grad():
            target_features = self.target_encoder(x)
            target_features = apply_masks(target_features, target_masks)
            if self.cfg.get("normalize_target", True):
                target_features = F.layer_norm(target_features, target_features.shape[-1:])
        return target_features

    def forward_context(self, x, context_masks, target_masks):
        context_features = self.context_encoder(x, masks=context_masks)
        return self.predictor(context_features, context_masks, target_masks)

    def forward(self, x, target_masks, context_masks):
        targets = self.forward_target(x, target_masks)
        predictions = self.forward_context(x, context_masks, target_masks)
        return F.smooth_l1_loss(predictions, targets)

    def update_target_encoder(self, ema_decay):
        with torch.no_grad():
            for source, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                target.data.mul_(ema_decay).add_((1.0 - ema_decay) * source.data)


class IJEPAFeatureExtractor(nn.Module):
    def __init__(self, context_encoder: VisionTransformer):
        super().__init__()
        self.encoder = context_encoder
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        tokens = self.encoder(x)
        return tokens.mean(dim=1)
