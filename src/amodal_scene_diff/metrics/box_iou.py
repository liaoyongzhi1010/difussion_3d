"""Oriented 3D box IoU with yaw rotation around the vertical (y) axis.

Box convention matches the project's pose vector:
    center   = (cx, cy, cz)   — geometric center in room frame (meters)
    size     = (sx, sy, sz)   — full extents along local (x, y, z)
    yaw      = rotation around y-axis in radians (right-hand rule)

The 3D IoU factors as (2D rotated IoU in x-z plane) × (1D IoU along y), which
is exact when the only rotation is yaw.
"""

from __future__ import annotations

import math

import torch


def _corners_xz(center_xz: torch.Tensor, size_xz: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Four footprint corners (x, z) in world order, counter-clockwise.

    Shapes: center_xz [B, 2], size_xz [B, 2], yaw [B] → [B, 4, 2].
    """
    half = size_xz * 0.5
    # local corner signs, CCW when viewed from +y
    signs = torch.tensor(
        [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]],
        device=center_xz.device,
        dtype=center_xz.dtype,
    )
    local = signs.unsqueeze(0) * half.unsqueeze(1)  # [B, 4, 2]
    cos = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
    sin = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)
    rot = torch.cat([
        torch.cat([cos, -sin], dim=-1),
        torch.cat([sin, cos], dim=-1),
    ], dim=-2)  # [B, 2, 2]
    rotated = torch.einsum("bij,bkj->bki", rot, local)
    return rotated + center_xz.unsqueeze(1)


def _poly_area(poly: torch.Tensor) -> torch.Tensor:
    """Signed Shoelace area of a polygon [K, 2]."""
    if poly.shape[0] < 3:
        return poly.new_zeros(())
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y))


def _clip_polygon(subject: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Sutherland–Hodgman: clip convex `subject` polygon by convex `clip` polygon.

    Both expected to be listed counter-clockwise.
    """
    output = subject
    n = clip.shape[0]
    for i in range(n):
        if output.shape[0] == 0:
            return output
        a = clip[i]
        b = clip[(i + 1) % n]
        edge = b - a
        # inside test: cross((b-a), (p-a)) >= 0 for CCW clip
        def _inside(p: torch.Tensor) -> torch.Tensor:
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])

        new_output: list[torch.Tensor] = []
        m = output.shape[0]
        for j in range(m):
            cur = output[j]
            prev = output[(j - 1) % m]
            cur_in = _inside(cur) >= 0
            prev_in = _inside(prev) >= 0
            if cur_in:
                if not prev_in:
                    new_output.append(_intersect(prev, cur, a, b))
                new_output.append(cur)
            elif prev_in:
                new_output.append(_intersect(prev, cur, a, b))
        if not new_output:
            return output.new_zeros((0, 2))
        output = torch.stack(new_output, dim=0)
    return output


def _intersect(p1: torch.Tensor, p2: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Intersection point of segment (p1,p2) with line through (a,b)."""
    r = p2 - p1
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    eps = p1.new_tensor(1e-9)
    t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / torch.where(denom.abs() < eps, denom + eps, denom)
    return p1 + t * r


def _rotated_rect_iou_xz(corners_a: torch.Tensor, corners_b: torch.Tensor) -> torch.Tensor:
    """2D IoU of two rotated rectangles given their 4-corner (CCW) polygons."""
    inter_poly = _clip_polygon(corners_a, corners_b)
    if inter_poly.shape[0] < 3:
        return corners_a.new_zeros(())
    inter = _poly_area(inter_poly)
    area_a = _poly_area(corners_a)
    area_b = _poly_area(corners_b)
    union = area_a + area_b - inter
    return inter / torch.clamp(union, min=1e-9)


def box_iou_3d(
    center_a: torch.Tensor,
    size_a: torch.Tensor,
    yaw_a: torch.Tensor,
    center_b: torch.Tensor,
    size_b: torch.Tensor,
    yaw_b: torch.Tensor,
) -> torch.Tensor:
    """Oriented 3D IoU for N pairs of boxes.

    Each input is batched along dim 0 with matching N. Returns tensor [N].
    """
    if center_a.shape != center_b.shape or size_a.shape != size_b.shape:
        raise ValueError("pairwise inputs must have matching shapes")
    if center_a.ndim != 2 or center_a.shape[-1] != 3:
        raise ValueError(f"expected [N, 3] centers, got {tuple(center_a.shape)}")
    n = center_a.shape[0]
    out = center_a.new_zeros(n)
    corners_a = _corners_xz(center_a[:, [0, 2]], size_a[:, [0, 2]], yaw_a)
    corners_b = _corners_xz(center_b[:, [0, 2]], size_b[:, [0, 2]], yaw_b)
    for i in range(n):
        inter_area_xz = _poly_area(_clip_polygon(corners_a[i], corners_b[i]))
        area_a = size_a[i, 0] * size_a[i, 2]
        area_b = size_b[i, 0] * size_b[i, 2]

        # y-axis (vertical) 1D overlap
        y_a_lo = center_a[i, 1] - 0.5 * size_a[i, 1]
        y_a_hi = center_a[i, 1] + 0.5 * size_a[i, 1]
        y_b_lo = center_b[i, 1] - 0.5 * size_b[i, 1]
        y_b_hi = center_b[i, 1] + 0.5 * size_b[i, 1]
        y_overlap = torch.clamp(torch.minimum(y_a_hi, y_b_hi) - torch.maximum(y_a_lo, y_b_lo), min=0.0)

        inter_vol = inter_area_xz * y_overlap
        vol_a = area_a * size_a[i, 1]
        vol_b = area_b * size_b[i, 1]
        union = vol_a + vol_b - inter_vol
        out[i] = inter_vol / torch.clamp(union, min=1e-9)
    return out


def pairwise_box_iou_3d(
    centers_a: torch.Tensor,
    sizes_a: torch.Tensor,
    yaws_a: torch.Tensor,
    centers_b: torch.Tensor,
    sizes_b: torch.Tensor,
    yaws_b: torch.Tensor,
) -> torch.Tensor:
    """Full pairwise IoU [Na, Nb] (for Hungarian matching)."""
    na = centers_a.shape[0]
    nb = centers_b.shape[0]
    out = centers_a.new_zeros(na, nb)
    if na == 0 or nb == 0:
        return out
    # avoid O(NaNb) Python loop by flattening pairs
    ia, ib = torch.meshgrid(torch.arange(na), torch.arange(nb), indexing="ij")
    ia = ia.reshape(-1)
    ib = ib.reshape(-1)
    iou_flat = box_iou_3d(
        centers_a[ia], sizes_a[ia], yaws_a[ia],
        centers_b[ib], sizes_b[ib], yaws_b[ib],
    )
    return iou_flat.view(na, nb)


__all__ = ["box_iou_3d", "pairwise_box_iou_3d"]


if False:  # pragma: no cover — sanity reference for reviewers
    # Self-IoU of an axis-aligned unit cube == 1.0
    c = torch.zeros(1, 3)
    s = torch.ones(1, 3)
    y = torch.zeros(1)
    assert math.isclose(box_iou_3d(c, s, y, c, s, y).item(), 1.0, abs_tol=1e-5)
