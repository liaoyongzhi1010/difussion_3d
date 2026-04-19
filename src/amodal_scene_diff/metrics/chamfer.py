"""Chamfer distance for per-object and scene-level point clouds.

All distances are Euclidean in the room coordinate frame (meters). Operates on
torch tensors and is differentiable where `reduce="mean"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


def _nn_sq_distance(query: torch.Tensor, target: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Squared distance from each query point to its nearest target point.

    Shapes: query [Q, 3], target [T, 3] → [Q]. Chunked to bound memory on dense
    clouds (a 10k × 10k cdist matrix is 400 MB at fp32).
    """
    if query.ndim != 2 or target.ndim != 2 or query.shape[-1] != target.shape[-1]:
        raise ValueError(f"expected [Q,D] and [T,D], got {tuple(query.shape)} vs {tuple(target.shape)}")
    if query.numel() == 0 or target.numel() == 0:
        return torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
    out = torch.empty(query.shape[0], device=query.device, dtype=query.dtype)
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        dists = torch.cdist(query[start:end], target, p=2.0)
        out[start:end] = dists.min(dim=1).values.pow(2)
    return out


def chamfer_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mode: Literal["symmetric", "pred_to_gt", "gt_to_pred"] = "symmetric",
    reduce: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Chamfer distance between two point clouds.

    - pred, gt: [N, 3] and [M, 3]
    - symmetric: 0.5 * (E_pred[d(pred→gt)^2] + E_gt[d(gt→pred)^2])
    """
    d_pg = _nn_sq_distance(pred, gt)
    d_gp = _nn_sq_distance(gt, pred)
    if mode == "pred_to_gt":
        per_point = d_pg
    elif mode == "gt_to_pred":
        per_point = d_gp
    elif mode == "symmetric":
        if reduce == "mean":
            return 0.5 * (d_pg.mean() + d_gp.mean())
        if reduce == "sum":
            return d_pg.sum() + d_gp.sum()
        return torch.cat([d_pg, d_gp], dim=0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if reduce == "mean":
        return per_point.mean()
    if reduce == "sum":
        return per_point.sum()
    return per_point


@dataclass(frozen=True)
class ObjectCloud:
    points: torch.Tensor       # [N, 3] in room frame
    visible: bool              # True if this object was in the visible set


@dataclass(frozen=True)
class SceneChamferReport:
    scene: float
    visible_only: float
    hidden_only: float
    n_scene: int
    n_visible: int
    n_hidden: int


def scene_chamfer(
    pred_objects: list[ObjectCloud],
    gt_objects: list[ObjectCloud],
    match: list[tuple[int, int]],
) -> SceneChamferReport:
    """Aggregate chamfer over matched (pred, gt) object pairs.

    `match` is a list of (pred_idx, gt_idx) from Hungarian matching in the
    caller. Objects not in `match` are ignored (handled by recall/precision
    metrics in hidden_recall.py).
    """
    if not match:
        nan = float("nan")
        return SceneChamferReport(nan, nan, nan, 0, 0, 0)
    scene_d: list[torch.Tensor] = []
    visible_d: list[torch.Tensor] = []
    hidden_d: list[torch.Tensor] = []
    for pi, gi in match:
        d = chamfer_distance(pred_objects[pi].points, gt_objects[gi].points, mode="symmetric", reduce="mean")
        scene_d.append(d)
        if gt_objects[gi].visible:
            visible_d.append(d)
        else:
            hidden_d.append(d)

    def _mean(xs: list[torch.Tensor]) -> float:
        if not xs:
            return float("nan")
        return float(torch.stack(xs).mean().item())

    return SceneChamferReport(
        scene=_mean(scene_d),
        visible_only=_mean(visible_d),
        hidden_only=_mean(hidden_d),
        n_scene=len(scene_d),
        n_visible=len(visible_d),
        n_hidden=len(hidden_d),
    )
