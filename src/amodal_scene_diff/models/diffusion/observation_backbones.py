from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from amodal_scene_diff.structures import SingleViewSceneBatch


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


class PatchObservationEncoder(nn.Module):
    def __init__(
        self,
        *,
        obs_channels: int,
        image_size: int,
        patch_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        num_tokens = self.grid_size * self.grid_size
        self.patch_embed = nn.Conv2d(obs_channels, d_model, kernel_size=patch_size, stride=patch_size)
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

    def forward(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        obs_image = batch.cond.obs_image
        tokens = self.patch_embed(obs_image).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(obs_image.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return {
            "global_token": tokens[:, 0],
            "patch_tokens": tokens[:, 1:],
        }


class _Dinov2Base(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        model_name: str,
        image_size: int,
        freeze_backbone: bool,
        train_last_n_blocks: int,
        allow_pseudo_rgb: bool,
    ) -> None:
        super().__init__()
        from transformers import Dinov2Model

        self.image_size = int(image_size)
        self.allow_pseudo_rgb = bool(allow_pseudo_rgb)
        self.backbone = Dinov2Model.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.hidden_size = hidden_size
        self.d_model = int(d_model)
        self.rgb_proj = nn.Identity() if hidden_size == int(d_model) else nn.Linear(hidden_size, int(d_model))
        self.rgb_norm = nn.LayerNorm(int(d_model))
        self._configure_trainable_layers(freeze_backbone=freeze_backbone, train_last_n_blocks=train_last_n_blocks)

    def _configure_trainable_layers(self, *, freeze_backbone: bool, train_last_n_blocks: int) -> None:
        if freeze_backbone or train_last_n_blocks > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if train_last_n_blocks > 0:
            num_layers = len(self.backbone.encoder.layer)
            keep = min(max(int(train_last_n_blocks), 0), num_layers)
            for block in self.backbone.encoder.layer[num_layers - keep :]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "layernorm"):
                for param in self.backbone.layernorm.parameters():
                    param.requires_grad = True
        elif not freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def _select_rgb_inputs(self, batch: SingleViewSceneBatch) -> torch.Tensor:
        if not self.allow_pseudo_rgb and not bool(batch.cond.rgb_available.all().item()):
            raise RuntimeError(
                "transformers_dinov2 backbone requires real RGB observations. "
                "Current batch marks rgb_available=false; wire RGB into the dataset first or set allow_pseudo_rgb=true."
            )
        if int(batch.cond.obs_image.shape[1]) < 3:
            raise ValueError(f"expected at least 3 observation channels, got {tuple(batch.cond.obs_image.shape)}")
        rgb = batch.cond.obs_image[:, :3]
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return rgb

    def _encode_rgb_tokens(self, batch: SingleViewSceneBatch) -> torch.Tensor:
        rgb = self._select_rgb_inputs(batch)
        outputs = self.backbone(pixel_values=rgb)
        return self.rgb_norm(self.rgb_proj(outputs.last_hidden_state))


class TransformersDinov2ObservationEncoder(_Dinov2Base):
    def __init__(
        self,
        *,
        d_model: int,
        model_name: str,
        image_size: int,
        freeze_backbone: bool,
        train_last_n_blocks: int,
        allow_pseudo_rgb: bool,
    ) -> None:
        super().__init__(
            d_model=d_model,
            model_name=model_name,
            image_size=image_size,
            freeze_backbone=freeze_backbone,
            train_last_n_blocks=train_last_n_blocks,
            allow_pseudo_rgb=allow_pseudo_rgb,
        )

    def forward(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        tokens = self._encode_rgb_tokens(batch)
        return {
            "global_token": tokens[:, 0],
            "patch_tokens": tokens[:, 1:],
        }


class AuxiliaryGeometryTokenizer(nn.Module):
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


class CrossTokenFusionBlock(nn.Module):
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
        attn_out, _ = self.attn(self.query_norm(query_tokens), self.kv_norm(aux_tokens), self.kv_norm(aux_tokens), need_weights=False)
        fused = query_tokens + attn_out
        return fused + self.ffn(self.ffn_norm(fused))


class TransformersDinov2HybridObservationEncoder(_Dinov2Base):
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
        self.aux_tokenizer = AuxiliaryGeometryTokenizer(
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
                CrossTokenFusionBlock(d_model=d_model, num_heads=fusion_heads, ffn_ratio=ffn_ratio, dropout=dropout)
                for _ in range(fusion_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        rgb_tokens = self._encode_rgb_tokens(batch)
        aux_tokens = self.aux_tokenizer(batch)
        fused = rgb_tokens
        for block in self.fusion_blocks:
            fused = block(fused, aux_tokens)
        fused = self.output_norm(fused)
        return {
            "global_token": fused[:, 0],
            "patch_tokens": fused[:, 1:],
        }


def build_single_view_observation_encoder(
    *,
    obs_channels: int,
    image_size: int,
    patch_size: int,
    d_model: int,
    encoder_layers: int,
    num_heads: int,
    ffn_ratio: float,
    dropout: float,
    backbone_cfg: dict[str, Any] | None = None,
) -> nn.Module:
    cfg = dict(backbone_cfg or {})
    backbone_type = str(cfg.get("type", "patch_vit")).lower()

    if backbone_type == "patch_vit":
        return PatchObservationEncoder(
            obs_channels=obs_channels,
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )

    if backbone_type == "transformers_dinov2":
        return TransformersDinov2ObservationEncoder(
            d_model=d_model,
            model_name=str(cfg.get("model_name", "facebook/dinov2-base")),
            image_size=int(cfg.get("image_size", 518)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            train_last_n_blocks=int(cfg.get("train_last_n_blocks", 0)),
            allow_pseudo_rgb=bool(cfg.get("allow_pseudo_rgb", False)),
        )

    if backbone_type == "transformers_dinov2_hybrid":
        return TransformersDinov2HybridObservationEncoder(
            d_model=d_model,
            model_name=str(cfg.get("model_name", "facebook/dinov2-large")),
            image_size=int(cfg.get("image_size", 518)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            train_last_n_blocks=int(cfg.get("train_last_n_blocks", 0)),
            allow_pseudo_rgb=bool(cfg.get("allow_pseudo_rgb", False)),
            aux_patch_size=int(cfg.get("aux_patch_size", 14)),
            aux_channels=int(cfg.get("aux_channels", 4)),
            aux_encoder_layers=int(cfg.get("aux_encoder_layers", 4)),
            aux_num_heads=int(cfg.get("aux_num_heads", max(1, num_heads))),
            fusion_layers=int(cfg.get("fusion_layers", 2)),
            fusion_heads=int(cfg.get("fusion_heads", max(1, num_heads))),
            ffn_ratio=float(cfg.get("ffn_ratio", ffn_ratio)),
            dropout=float(cfg.get("dropout", dropout)),
        )

    raise ValueError(f"unsupported observation backbone type: {backbone_type}")
