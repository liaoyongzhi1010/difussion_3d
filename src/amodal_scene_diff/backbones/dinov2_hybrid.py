from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from amodal_scene_diff.structures import SingleViewSceneBatch

from .dinov2 import _Dinov2Core


def _normalize_depth(depth_obs: torch.Tensor) -> torch.Tensor:
    depth = depth_obs.float().clone()
    valid = depth > 0
    if bool(valid.any()):
        values = depth[valid]
        dmin = values.min()
        dmax = values.max()
        if float((dmax - dmin).abs().item()) > 1.0e-6:
            depth[valid] = (values - dmin) / (dmax - dmin)
        else:
            depth[valid] = 0.0
    depth[~valid] = 0.0
    return depth.clamp(0.0, 1.0)


def _depth_gradient(depth: torch.Tensor) -> torch.Tensor:
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[..., :, 1:] = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    grad_y[..., 1:, :] = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    return (grad_x + grad_y).clamp(0.0, 1.0)


class _AuxiliaryGeometryTokenizer(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        aux_channels: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        num_tokens = self.grid_size * self.grid_size
        self.patch_embed = nn.Conv2d(aux_channels, d_model, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def _build_aux_map(self, batch: SingleViewSceneBatch) -> torch.Tensor:
        depth = _normalize_depth(batch.cond.depth_obs)
        mask = batch.cond.visible_union_mask.float().clamp(0.0, 1.0)
        masked_depth = depth * mask
        gradient = _depth_gradient(depth)
        aux_map = torch.cat([depth, mask, masked_depth, gradient], dim=1)
        if aux_map.shape[-2:] != (self.image_size, self.image_size):
            aux_map = F.interpolate(aux_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return aux_map

    def forward(self, batch: SingleViewSceneBatch) -> torch.Tensor:
        aux_map = self._build_aux_map(batch)
        tokens = self.patch_embed(aux_map).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(aux_map.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.transformer(tokens)
        return self.norm(tokens)


class _CrossTokenFusionBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query_tokens),
            self.kv_norm(aux_tokens),
            self.kv_norm(aux_tokens),
            need_weights=False,
        )
        fused = query_tokens + attn_out
        return fused + self.ffn(self.ffn_norm(fused))


class Dinov2HybridBackbone(_Dinov2Core):
    """DINOv2 RGB stream fused with a depth-aware auxiliary stream via cross-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        model_name: str,
        image_size: int,
        freeze_backbone: bool,
        train_last_n_blocks: int,
        allow_pseudo_rgb: bool,
        aux_patch_size: int,
        aux_channels: int,
        aux_encoder_layers: int,
        aux_num_heads: int,
        fusion_layers: int,
        fusion_heads: int,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__(
            d_model=d_model,
            model_name=model_name,
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            train_last_n_blocks=train_last_n_blocks,
            allow_pseudo_rgb=allow_pseudo_rgb,
        )
        self.aux_tokenizer = _AuxiliaryGeometryTokenizer(
            image_size=image_size,
            patch_size=aux_patch_size,
            aux_channels=aux_channels,
            d_model=d_model,
            num_layers=aux_encoder_layers,
            num_heads=aux_num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )
        self.fusion_blocks = nn.ModuleList(
            [
                _CrossTokenFusionBlock(d_model=d_model, num_heads=fusion_heads, ffn_ratio=ffn_ratio, dropout=dropout)
                for _ in range(fusion_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        rgb_tokens = self.encode_rgb_tokens(batch)
        aux_tokens = self.aux_tokenizer(batch)
        fused = rgb_tokens
        for block in self.fusion_blocks:
            fused = block(fused, aux_tokens)
        fused = self.output_norm(fused)
        return {"global_token": fused[:, 0], "patch_tokens": fused[:, 1:]}
