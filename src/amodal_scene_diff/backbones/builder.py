from __future__ import annotations

from typing import Any

from torch import nn

from .dinov2 import Dinov2Backbone
from .dinov2_hybrid import Dinov2HybridBackbone
from .patch_vit import PatchViTBackbone


def build_observation_backbone(
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
        return PatchViTBackbone(
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
        return Dinov2Backbone(
            d_model=d_model,
            model_name=str(cfg.get("model_name", "facebook/dinov2-base")),
            image_size=int(cfg.get("image_size", 518)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            train_last_n_blocks=int(cfg.get("train_last_n_blocks", 0)),
            allow_pseudo_rgb=bool(cfg.get("allow_pseudo_rgb", False)),
        )
    if backbone_type == "transformers_dinov2_hybrid":
        return Dinov2HybridBackbone(
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
