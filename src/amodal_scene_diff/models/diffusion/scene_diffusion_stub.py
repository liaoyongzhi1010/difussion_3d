from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from amodal_scene_diff.structures import C_OBJ, D_MODEL, D_POSE, K_HID, K_VIS, N_OBJ_MAX, Z_DIM, SceneDiffusionBatch


class SceneDiffusionStub(nn.Module):
    """Minimal trainable stub that respects the current batch schema.

    This is not the final research model. It only exists to validate that the
    data pipeline, batching, and optimization loop can run end-to-end.
    """

    def __init__(self) -> None:
        super().__init__()
        self.visible_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_POSE + Z_DIM),
        )
        self.hidden_queries = nn.Parameter(torch.randn(K_HID, D_MODEL) * 0.02)
        self.hidden_backbone = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.hidden_exist_head = nn.Linear(D_MODEL, 1)
        self.hidden_cls_head = nn.Linear(D_MODEL, C_OBJ)
        self.hidden_pose_head = nn.Linear(D_MODEL, D_POSE)
        self.hidden_z_head = nn.Linear(D_MODEL, Z_DIM)
        self.floor_head = nn.Linear(D_MODEL, 1)
        self.wall_head = nn.Linear(D_MODEL, 1)

    def forward(self, batch: SceneDiffusionBatch) -> dict[str, torch.Tensor]:
        cond = batch.cond
        target = batch.target
        device = cond.f_global.device

        global_token = cond.f_global.unsqueeze(1)
        visible_base = cond.visible_tokens_cond + global_token
        visible_pred = self.visible_head(visible_base)

        hidden_tokens = self.hidden_queries.unsqueeze(0).to(device) + global_token
        hidden_tokens = self.hidden_backbone(hidden_tokens)

        hidden_exist_logits = self.hidden_exist_head(hidden_tokens)
        hidden_cls_logits = self.hidden_cls_head(hidden_tokens)
        hidden_pose_pred = self.hidden_pose_head(hidden_tokens)
        hidden_z_pred = self.hidden_z_head(hidden_tokens)

        object_tokens = torch.cat([visible_base, hidden_tokens], dim=1)
        support_logits = torch.einsum("bid,bjd->bij", object_tokens, object_tokens) / math.sqrt(D_MODEL)
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
