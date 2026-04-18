from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from amodal_scene_diff.structures import SingleViewSceneBatch


class _Dinov2Core(nn.Module):
    """Shared DINOv2 adapter with optional last-N-block finetuning."""

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
        self._configure_trainable(freeze_backbone=freeze_backbone, train_last_n_blocks=train_last_n_blocks)

    def _configure_trainable(self, *, freeze_backbone: bool, train_last_n_blocks: int) -> None:
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

    def select_rgb_inputs(self, batch: SingleViewSceneBatch) -> torch.Tensor:
        if not self.allow_pseudo_rgb and not bool(batch.cond.rgb_available.all().item()):
            raise RuntimeError(
                "DINOv2 backbone requires real RGB observations. "
                "Batch marks rgb_available=false; wire RGB into the dataset first "
                "or set allow_pseudo_rgb=true."
            )
        if int(batch.cond.obs_image.shape[1]) < 3:
            raise ValueError(f"expected at least 3 observation channels, got {tuple(batch.cond.obs_image.shape)}")
        rgb = batch.cond.obs_image[:, :3]
        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return rgb

    def encode_rgb_tokens(self, batch: SingleViewSceneBatch) -> torch.Tensor:
        rgb = self.select_rgb_inputs(batch)
        outputs = self.backbone(pixel_values=rgb)
        return self.rgb_norm(self.rgb_proj(outputs.last_hidden_state))


class Dinov2Backbone(_Dinov2Core):
    """DINOv2-only encoder wrapped to emit {global_token, patch_tokens}."""

    def forward(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        tokens = self.encode_rgb_tokens(batch)
        return {"global_token": tokens[:, 0], "patch_tokens": tokens[:, 1:]}
