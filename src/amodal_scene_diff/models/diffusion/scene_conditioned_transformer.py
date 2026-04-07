from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from amodal_scene_diff.structures import C_OBJ, D_MODEL, D_POSE, K_HID, K_VIS, N_OBJ_MAX, Z_DIM, SceneDiffusionBatch


class SceneConditionedTransformer(nn.Module):
    """First non-trivial oracle-scaffold scene backbone.

    This is still a supervised bootstrap model, not the final diffusion model.
    It replaces the earlier stub with a relation-aware transformer over:
    - layout token
    - uncertainty token
    - visible scene tokens
    - hidden object queries
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(3, d_model)
        self.global_proj = nn.Linear(D_MODEL, d_model)
        self.layout0_proj = nn.Linear(D_POSE, d_model)
        self.pose_proj = nn.Linear(D_POSE, d_model)
        self.conf_proj = nn.Linear(3, d_model)
        self.visible_in = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.layout_in = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.uncertainty_in = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.hidden_queries = nn.Parameter(torch.randn(K_HID, d_model) * 0.02)

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
        self.object_norm = nn.LayerNorm(d_model)
        self.visible_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, D_POSE + Z_DIM),
        )
        self.hidden_exist_head = nn.Linear(d_model, 1)
        self.hidden_cls_head = nn.Linear(d_model, C_OBJ)
        self.hidden_pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, D_POSE),
        )
        self.hidden_z_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, Z_DIM),
        )
        self.relation_q = nn.Linear(d_model, d_model)
        self.relation_k = nn.Linear(d_model, d_model)
        self.floor_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.wall_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, batch: SceneDiffusionBatch) -> dict[str, torch.Tensor]:
        cond = batch.cond
        target = batch.target
        source = self.source_embedding(cond.source_id.long())
        global_bias = self.global_proj(cond.f_global)

        layout_token = self.layout_in(cond.layout_token_cond.squeeze(1) + self.layout0_proj(cond.layout0_calib) + source + global_bias)
        uncertainty_token = self.uncertainty_in(cond.uncertainty_token.squeeze(1) + source + global_bias)

        visible_aux = self.pose_proj(cond.pose0_calib) + self.conf_proj(
            torch.cat(
                [
                    cond.slot_confidence,
                    cond.lock_gate,
                    cond.visible_valid_mask.float().unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        visible_tokens = self.visible_in(cond.visible_tokens_cond + visible_aux + source.unsqueeze(1) + global_bias.unsqueeze(1))
        hidden_tokens = self.hidden_queries.unsqueeze(0) + source.unsqueeze(1) + global_bias.unsqueeze(1)

        tokens = torch.cat(
            [
                layout_token.unsqueeze(1),
                uncertainty_token.unsqueeze(1),
                visible_tokens,
                hidden_tokens,
            ],
            dim=1,
        )
        src_key_padding_mask = torch.zeros(
            (tokens.shape[0], tokens.shape[1]),
            dtype=torch.bool,
            device=tokens.device,
        )
        src_key_padding_mask[:, 2 : 2 + K_VIS] = ~cond.visible_valid_mask.bool()
        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)

        visible_ctx = self.object_norm(encoded[:, 2 : 2 + K_VIS])
        hidden_ctx = self.object_norm(encoded[:, 2 + K_VIS :])
        object_tokens = torch.cat([visible_ctx, hidden_ctx], dim=1)

        visible_pred = self.visible_head(visible_ctx)
        hidden_exist_logits = self.hidden_exist_head(hidden_ctx)
        hidden_cls_logits = self.hidden_cls_head(hidden_ctx)
        hidden_pose_pred = self.hidden_pose_head(hidden_ctx)
        hidden_z_pred = self.hidden_z_head(hidden_ctx)

        relation_q = self.relation_q(object_tokens)
        relation_k = self.relation_k(object_tokens)
        support_logits = torch.einsum("bid,bjd->bij", relation_q, relation_k) / math.sqrt(self.d_model)
        floor_logits = self.floor_head(object_tokens).squeeze(-1)
        wall_logits = self.wall_head(object_tokens).squeeze(-1)

        visible_target = torch.cat([target.visible_amodal_res_gt, target.visible_z_gt], dim=-1)
        return {
            "visible_pred": visible_pred,
            "visible_target": visible_target,
            "hidden_exist_logits": hidden_exist_logits,
            "hidden_cls_logits": hidden_cls_logits,
            "hidden_pose_pred": hidden_pose_pred,
            "hidden_z_pred": hidden_z_pred,
            "support_logits": support_logits,
            "floor_logits": floor_logits,
            "wall_logits": wall_logits,
        }

    def compute_losses(self, batch: SceneDiffusionBatch) -> dict[str, torch.Tensor]:
        pred = self.forward(batch)
        target = batch.target
        device = batch.cond.f_global.device

        visible_mask = target.visible_loss_mask.float().unsqueeze(-1)
        visible_denom = visible_mask.sum().clamp_min(1.0)
        visible_loss = ((pred["visible_pred"] - pred["visible_target"]) ** 2 * visible_mask).sum() / visible_denom

        hidden_mask = target.hidden_gt_mask.float()
        hidden_mask_3d = hidden_mask.unsqueeze(-1)
        hidden_exist_loss = F.binary_cross_entropy_with_logits(
            pred["hidden_exist_logits"].squeeze(-1),
            hidden_mask,
        )

        if hidden_mask.bool().any():
            cls_loss = F.cross_entropy(
                pred["hidden_cls_logits"][hidden_mask.bool()],
                target.hidden_cls_gt[hidden_mask.bool()],
            )
            pose_loss = (((pred["hidden_pose_pred"] - target.hidden_pose_gt) ** 2) * hidden_mask_3d).sum() / hidden_mask_3d.sum().clamp_min(1.0)
            z_loss = (((pred["hidden_z_pred"] - target.hidden_z_gt) ** 2) * hidden_mask_3d).sum() / hidden_mask_3d.sum().clamp_min(1.0)
        else:
            cls_loss = torch.zeros((), device=device)
            pose_loss = torch.zeros((), device=device)
            z_loss = torch.zeros((), device=device)

        relation_valid = target.relation_valid_mask.float()
        pair_mask = relation_valid.unsqueeze(1) * relation_valid.unsqueeze(2)
        support_loss = F.binary_cross_entropy_with_logits(
            pred["support_logits"],
            target.support_gt.float(),
            weight=pair_mask,
            reduction="sum",
        ) / pair_mask.sum().clamp_min(1.0)
        floor_loss = F.binary_cross_entropy_with_logits(
            pred["floor_logits"],
            target.floor_gt.float(),
            weight=relation_valid,
            reduction="sum",
        ) / relation_valid.sum().clamp_min(1.0)
        wall_loss = F.binary_cross_entropy_with_logits(
            pred["wall_logits"],
            target.wall_gt.float(),
            weight=relation_valid,
            reduction="sum",
        ) / relation_valid.sum().clamp_min(1.0)

        total = visible_loss + hidden_exist_loss + cls_loss + pose_loss + z_loss + 0.25 * (support_loss + floor_loss + wall_loss)
        return {
            "loss_total": total,
            "loss_visible": visible_loss,
            "loss_hidden_exist": hidden_exist_loss,
            "loss_hidden_cls": cls_loss,
            "loss_hidden_pose": pose_loss,
            "loss_hidden_z": z_loss,
            "loss_support": support_loss,
            "loss_floor": floor_loss,
            "loss_wall": wall_loss,
        }

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
