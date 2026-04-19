"""Physical-plausibility proxies: inter-object collision and support violation.

These are standard supplementary metrics in 3D scene generation papers
(BlockFusion, MIDI, etc.). Both are computed in room coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .box_iou import pairwise_box_iou_3d


@dataclass(frozen=True)
class CollisionReport:
    rate: float            # fraction of object pairs with IoU > 0
    mean_overlap: float    # average IoU over colliding pairs (0 if none)
    num_colliding: int
    num_pairs: int


@dataclass(frozen=True)
class SupportReport:
    rate: float            # fraction of objects whose bottom floats or clips
    mean_gap_m: float      # mean signed gap (positive = float, negative = clip)
    num_violations: int
    num_objects: int


def collision_rate(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    iou_threshold: float = 1e-4,
) -> CollisionReport:
    """Fraction of unordered pairs with 3D IoU above `iou_threshold`."""
    n = centers.shape[0]
    if n < 2:
        return CollisionReport(rate=0.0, mean_overlap=0.0, num_colliding=0, num_pairs=0)
    iou = pairwise_box_iou_3d(centers, sizes, yaws, centers, sizes, yaws)
    mask = torch.triu(torch.ones_like(iou, dtype=torch.bool), diagonal=1)
    pair_iou = iou[mask]
    num_pairs = int(pair_iou.numel())
    colliding = pair_iou > iou_threshold
    num_colliding = int(colliding.sum().item())
    rate = 0.0 if num_pairs == 0 else num_colliding / num_pairs
    mean_overlap = 0.0 if num_colliding == 0 else float(pair_iou[colliding].mean().item())
    return CollisionReport(
        rate=rate,
        mean_overlap=mean_overlap,
        num_colliding=num_colliding,
        num_pairs=num_pairs,
    )


def support_violation(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    support_index: torch.Tensor,
    floor_height: float = 0.0,
    tolerance_m: float = 0.05,
) -> SupportReport:
    """Per-object support-surface violation.

    - `support_index[i]` = j means object i should rest on object j's top;
      negative values mean "rests on the floor".
    - An object violates support if |object_bottom_y − supporter_top_y| > tolerance.
    """
    n = centers.shape[0]
    if n == 0:
        return SupportReport(rate=0.0, mean_gap_m=0.0, num_violations=0, num_objects=0)
    bottoms = centers[:, 1] - 0.5 * sizes[:, 1]
    tops = centers[:, 1] + 0.5 * sizes[:, 1]

    gaps = torch.zeros(n, device=centers.device, dtype=centers.dtype)
    for i in range(n):
        supporter = int(support_index[i].item())
        if supporter < 0 or supporter >= n or supporter == i:
            gaps[i] = bottoms[i] - floor_height
        else:
            gaps[i] = bottoms[i] - tops[supporter]

    violations = gaps.abs() > tolerance_m
    num_violations = int(violations.sum().item())
    return SupportReport(
        rate=num_violations / n,
        mean_gap_m=float(gaps.abs().mean().item()),
        num_violations=num_violations,
        num_objects=n,
    )
