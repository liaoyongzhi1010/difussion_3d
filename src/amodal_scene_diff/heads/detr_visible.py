"""DETR-style variable-slot visible head with Hungarian matcher.

Replaces the fixed-K_VIS=12 transformer decoder on the visible branch. Follows
the DETR (Carion et al., 2020) set-prediction recipe:

    - N learnable object queries (default 50)
    - Multi-layer transformer decoder with cross-attention to the observation memory
    - Per-query outputs: existence logit, class logits, box pose, latent code
    - Hungarian matcher with weighted cost (class + L1 box + GIoU(xz) + latent cosine)
    - Set loss combining the four terms on matched pairs + no-object BCE on unmatched queries

Unlike the baseline head, this module expresses a true *set* so the decoder
does not need to know how many visible objects a scene has.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _giou_xz(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Generalized IoU on the x-z footprint of axis-aligned boxes.

    Boxes given as (cx, cz, sx, sz). Shapes: [Na, 4] and [Nb, 4] → [Na, Nb].
    Yaw is ignored here because this term is only part of a matching cost;
    oriented IoU is reserved for final evaluation (metrics.box_iou).
    """
    ax1 = box_a[:, 0] - 0.5 * box_a[:, 2]
    az1 = box_a[:, 1] - 0.5 * box_a[:, 3]
    ax2 = box_a[:, 0] + 0.5 * box_a[:, 2]
    az2 = box_a[:, 1] + 0.5 * box_a[:, 3]
    bx1 = box_b[:, 0] - 0.5 * box_b[:, 2]
    bz1 = box_b[:, 1] - 0.5 * box_b[:, 3]
    bx2 = box_b[:, 0] + 0.5 * box_b[:, 2]
    bz2 = box_b[:, 1] + 0.5 * box_b[:, 3]

    inter_x1 = torch.maximum(ax1.unsqueeze(1), bx1.unsqueeze(0))
    inter_z1 = torch.maximum(az1.unsqueeze(1), bz1.unsqueeze(0))
    inter_x2 = torch.minimum(ax2.unsqueeze(1), bx2.unsqueeze(0))
    inter_z2 = torch.minimum(az2.unsqueeze(1), bz2.unsqueeze(0))
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_z2 - inter_z1, min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (az2 - az1)
    area_b = (bx2 - bx1) * (bz2 - bz1)
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter
    iou = inter / union.clamp_min(1e-9)

    enc_x1 = torch.minimum(ax1.unsqueeze(1), bx1.unsqueeze(0))
    enc_z1 = torch.minimum(az1.unsqueeze(1), bz1.unsqueeze(0))
    enc_x2 = torch.maximum(ax2.unsqueeze(1), bx2.unsqueeze(0))
    enc_z2 = torch.maximum(az2.unsqueeze(1), bz2.unsqueeze(0))
    enc_area = (enc_x2 - enc_x1) * (enc_z2 - enc_z1)
    return iou - (enc_area - union) / enc_area.clamp_min(1e-9)


@dataclass
class HungarianMatchWeights:
    cls: float = 1.0
    l1: float = 2.0
    giou: float = 2.0
    latent: float = 1.0


def hungarian_match(
    pred_cls_logits: torch.Tensor,   # [Nq, C]
    pred_box: torch.Tensor,          # [Nq, 4] as (cx, cz, sx, sz)
    pred_latent: torch.Tensor,       # [Nq, Z]
    gt_cls: torch.Tensor,            # [Ng]
    gt_box: torch.Tensor,            # [Ng, 4]
    gt_latent: torch.Tensor,         # [Ng, Z]
    weights: HungarianMatchWeights,
) -> list[tuple[int, int]]:
    """Return list of (pred_idx, gt_idx) pairs minimizing the weighted cost."""
    if gt_cls.numel() == 0:
        return []
    prob = pred_cls_logits.softmax(dim=-1)                 # [Nq, C]
    cost_cls = -prob[:, gt_cls]                             # [Nq, Ng]
    cost_l1 = torch.cdist(pred_box, gt_box, p=1)            # [Nq, Ng]
    cost_giou = -_giou_xz(pred_box, gt_box)                 # [Nq, Ng]
    p_norm = F.normalize(pred_latent, dim=-1)
    g_norm = F.normalize(gt_latent, dim=-1)
    cost_latent = 1.0 - p_norm @ g_norm.t()                 # [Nq, Ng]
    total = (
        weights.cls * cost_cls
        + weights.l1 * cost_l1
        + weights.giou * cost_giou
        + weights.latent * cost_latent
    )

    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(total.detach().cpu().numpy())
        return list(zip(row.tolist(), col.tolist()))
    except ImportError:
        pass
    # greedy fallback
    assignments: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    flat = total.flatten()
    order = torch.argsort(flat, descending=False).tolist()
    nb = total.shape[1]
    for idx in order:
        r, c = divmod(idx, nb)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        assignments.append((r, c))
        if len(assignments) == min(total.shape[0], total.shape[1]):
            break
    return assignments


class DetrVisibleHead(nn.Module):
    """Variable-slot visible head with DETR decoder and Hungarian set loss.

    Forward returns a dict with per-query predictions; use `set_loss()` during
    training to match and compute the four loss terms.
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int = 50,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 10,
        pose_dim: int = 8,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim

        self.query_embedding = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)

        self.cls_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.exist_head = nn.Linear(d_model, 1)
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pose_dim),
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = memory.shape[0]
        queries = self.query_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        ctx = self.decoder(queries, memory)
        return {
            "tokens": ctx,
            "cls_logits": self.cls_head(ctx),
            "exist_logits": self.exist_head(ctx).squeeze(-1),
            "pose": self.pose_head(ctx),
            "latent": self.latent_head(ctx),
        }

    def set_loss(
        self,
        predictions: dict[str, torch.Tensor],
        gt_cls: list[torch.Tensor],
        gt_box_xz: list[torch.Tensor],       # axis-aligned (cx, cz, sx, sz) in room frame
        gt_pose_full: list[torch.Tensor],     # [Ng, pose_dim]
        gt_latent: list[torch.Tensor],
        weights: HungarianMatchWeights | None = None,
        no_object_class_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = weights or HungarianMatchWeights()
        no_obj = self.num_classes if no_object_class_index is None else no_object_class_index

        device = predictions["cls_logits"].device
        batch_size = predictions["cls_logits"].shape[0]
        cls_losses: list[torch.Tensor] = []
        exist_losses: list[torch.Tensor] = []
        l1_losses: list[torch.Tensor] = []
        giou_losses: list[torch.Tensor] = []
        latent_losses: list[torch.Tensor] = []
        pose_losses: list[torch.Tensor] = []

        num_matched_total = 0

        for b in range(batch_size):
            p_cls = predictions["cls_logits"][b]
            p_exist = predictions["exist_logits"][b]
            p_pose = predictions["pose"][b]
            p_lat = predictions["latent"][b]

            # derive a (cx, cz, sx, sz) box from the first 4 pose dims. The
            # caller owns the exact mapping; we just use the L1-friendly
            # projection for matching.
            p_box_xz = torch.stack(
                [p_pose[:, 0], p_pose[:, 2], p_pose[:, 3].abs(), p_pose[:, 5].abs()], dim=-1
            ) if p_pose.shape[-1] >= 6 else p_pose[:, :4]

            g_cls = gt_cls[b].to(device)
            g_box = gt_box_xz[b].to(device)
            g_pose = gt_pose_full[b].to(device)
            g_lat = gt_latent[b].to(device)

            matches = hungarian_match(
                p_cls[:, : self.num_classes], p_box_xz, p_lat, g_cls, g_box, g_lat, weights
            )

            target_cls = torch.full(
                (self.num_queries,), no_obj, dtype=torch.long, device=device
            )
            target_exist = torch.zeros(self.num_queries, device=device)
            matched_pred = torch.zeros(self.num_queries, dtype=torch.bool, device=device)
            for pi, gi in matches:
                target_cls[pi] = g_cls[gi]
                target_exist[pi] = 1.0
                matched_pred[pi] = True

            cls_losses.append(F.cross_entropy(p_cls, target_cls))
            exist_losses.append(F.binary_cross_entropy_with_logits(p_exist, target_exist))

            if matches:
                num_matched_total += len(matches)
                pi_idx = torch.tensor([pi for pi, _ in matches], device=device)
                gi_idx = torch.tensor([gi for _, gi in matches], device=device)
                l1_losses.append(F.l1_loss(p_box_xz[pi_idx], g_box[gi_idx]))
                giou_losses.append(1.0 - _giou_xz(p_box_xz[pi_idx], g_box[gi_idx]).diagonal().mean())
                lat_sim = F.cosine_similarity(p_lat[pi_idx], g_lat[gi_idx], dim=-1)
                latent_losses.append((1.0 - lat_sim).mean())
                pose_losses.append(F.l1_loss(p_pose[pi_idx], g_pose[gi_idx]))

        def _agg(terms: list[torch.Tensor]) -> torch.Tensor:
            if not terms:
                return torch.zeros((), device=device)
            return torch.stack(terms).mean()

        total = (
            weights.cls * _agg(cls_losses)
            + 1.0 * _agg(exist_losses)
            + weights.l1 * _agg(l1_losses)
            + weights.giou * _agg(giou_losses)
            + weights.latent * _agg(latent_losses)
            + 1.0 * _agg(pose_losses)
        )
        return {
            "loss_total": total,
            "loss_cls": _agg(cls_losses),
            "loss_exist": _agg(exist_losses),
            "loss_l1_box": _agg(l1_losses),
            "loss_giou": _agg(giou_losses),
            "loss_latent_cos": _agg(latent_losses),
            "loss_pose_l1": _agg(pose_losses),
            "num_matched": torch.tensor(num_matched_total, device=device, dtype=torch.float32),
        }


__all__ = ["DetrVisibleHead", "HungarianMatchWeights", "hungarian_match"]
