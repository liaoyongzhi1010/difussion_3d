from __future__ import annotations

import torch
from torch import nn

from amodal_scene_diff.structures import SingleViewSceneBatch


class PatchViTBackbone(nn.Module):
    """Scratch ViT encoder used only as a pseudo-RGB fallback backbone."""

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
        return {"global_token": tokens[:, 0], "patch_tokens": tokens[:, 1:]}
