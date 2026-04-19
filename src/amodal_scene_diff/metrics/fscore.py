"""F-score@τ for point clouds, following the InstPIFu / Total3D protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .chamfer import _nn_sq_distance


@dataclass(frozen=True)
class FScoreReport:
    precision: Mapping[float, float]
    recall: Mapping[float, float]
    fscore: Mapping[float, float]


def fscore_at_thresholds(
    pred: torch.Tensor,
    gt: torch.Tensor,
    thresholds_m: Sequence[float] = (0.01, 0.02, 0.05),
) -> FScoreReport:
    """F-score at thresholds τ (meters).

    precision(τ) = fraction of pred points within τ of any gt point
    recall(τ)    = fraction of gt points within τ of any pred point
    fscore(τ)    = 2PR / (P + R)
    """
    if pred.numel() == 0 or gt.numel() == 0:
        zeros = {float(t): 0.0 for t in thresholds_m}
        return FScoreReport(precision=zeros, recall=zeros, fscore=zeros)

    d_pg = _nn_sq_distance(pred, gt).sqrt()
    d_gp = _nn_sq_distance(gt, pred).sqrt()

    precision: dict[float, float] = {}
    recall: dict[float, float] = {}
    fscore: dict[float, float] = {}
    for tau in thresholds_m:
        p = float((d_pg <= tau).float().mean().item())
        r = float((d_gp <= tau).float().mean().item())
        f = 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)
        precision[float(tau)] = p
        recall[float(tau)] = r
        fscore[float(tau)] = f
    return FScoreReport(precision=precision, recall=recall, fscore=fscore)


def scene_fscore(
    pred_clouds: Sequence[torch.Tensor],
    gt_clouds: Sequence[torch.Tensor],
    match: Sequence[tuple[int, int]],
    thresholds_m: Sequence[float] = (0.01, 0.02, 0.05),
) -> FScoreReport:
    """Scene-level F-score: concatenate matched object clouds, then evaluate."""
    if not match:
        zeros = {float(t): 0.0 for t in thresholds_m}
        return FScoreReport(precision=zeros, recall=zeros, fscore=zeros)
    pred_cat = torch.cat([pred_clouds[pi] for pi, _ in match], dim=0)
    gt_cat = torch.cat([gt_clouds[gi] for _, gi in match], dim=0)
    return fscore_at_thresholds(pred_cat, gt_cat, thresholds_m=thresholds_m)
