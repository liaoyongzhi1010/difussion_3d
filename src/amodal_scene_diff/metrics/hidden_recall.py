"""Hidden-region recall/precision/F1 via IoU-threshold matching.

Protocol:
    1. Compute pairwise 3D IoU between predicted boxes and GT boxes.
    2. Solve a one-to-one assignment maximizing IoU (Hungarian).
    3. A pair counts as a true positive only if IoU ≥ iou_threshold (default 0.25).
    4. Split predictions/GT by visible vs hidden flags and report each slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .box_iou import pairwise_box_iou_3d


@dataclass(frozen=True)
class DetectionReport:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class AmodalRecallReport:
    visible: DetectionReport
    hidden: DetectionReport
    all: DetectionReport
    matches: list[tuple[int, int, float]]  # (pred_idx, gt_idx, iou)


def _hungarian_max_iou(iou: torch.Tensor) -> list[tuple[int, int]]:
    """One-to-one assignment that maximizes total IoU. O(N^3) SciPy path when
    available; falls back to a greedy-pick routine otherwise.
    """
    if iou.numel() == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        cost = -iou.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except ImportError:
        pass
    # greedy fallback
    matches: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    flat = iou.flatten()
    order = torch.argsort(flat, descending=True).tolist()
    nb = iou.shape[1]
    for idx in order:
        r, c = divmod(idx, nb)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matches.append((r, c))
        if len(matches) == min(iou.shape[0], iou.shape[1]):
            break
    return matches


def _detection_report(tp: int, fp: int, fn: int) -> DetectionReport:
    p = 0.0 if tp + fp == 0 else tp / (tp + fp)
    r = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f = 0.0 if p + r == 0.0 else 2 * p * r / (p + r)
    return DetectionReport(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)


def amodal_detection_report(
    pred_centers: torch.Tensor,
    pred_sizes: torch.Tensor,
    pred_yaws: torch.Tensor,
    pred_visible_mask: torch.Tensor,
    gt_centers: torch.Tensor,
    gt_sizes: torch.Tensor,
    gt_yaws: torch.Tensor,
    gt_visible_mask: torch.Tensor,
    iou_threshold: float = 0.25,
) -> AmodalRecallReport:
    """Compute visible / hidden / combined detection metrics for one scene.

    - `*_visible_mask`: bool tensor; True for the visible subset, False for hidden.
    """
    if pred_centers.shape[0] != pred_visible_mask.shape[0]:
        raise ValueError("prediction mask length must match number of predicted boxes")
    if gt_centers.shape[0] != gt_visible_mask.shape[0]:
        raise ValueError("gt mask length must match number of gt boxes")

    iou = pairwise_box_iou_3d(
        pred_centers, pred_sizes, pred_yaws,
        gt_centers, gt_sizes, gt_yaws,
    )
    assignment = _hungarian_max_iou(iou)
    matches: list[tuple[int, int, float]] = []
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for pi, gi in assignment:
        score = float(iou[pi, gi].item())
        if score >= iou_threshold:
            matches.append((pi, gi, score))
            matched_pred.add(pi)
            matched_gt.add(gi)

    pred_visible = pred_visible_mask.tolist()
    gt_visible = gt_visible_mask.tolist()
    np_pred = pred_centers.shape[0]
    np_gt = gt_centers.shape[0]

    tp_all = len(matches)
    fp_all = np_pred - tp_all
    fn_all = np_gt - tp_all

    tp_v = sum(1 for pi, gi, _ in matches if gt_visible[gi])
    tp_h = sum(1 for pi, gi, _ in matches if not gt_visible[gi])
    fp_v = sum(1 for i, flag in enumerate(pred_visible) if flag and i not in matched_pred)
    fp_h = sum(1 for i, flag in enumerate(pred_visible) if (not flag) and i not in matched_pred)
    fn_v = sum(1 for i, flag in enumerate(gt_visible) if flag and i not in matched_gt)
    fn_h = sum(1 for i, flag in enumerate(gt_visible) if (not flag) and i not in matched_gt)

    return AmodalRecallReport(
        visible=_detection_report(tp_v, fp_v, fn_v),
        hidden=_detection_report(tp_h, fp_h, fn_h),
        all=_detection_report(tp_all, fp_all, fn_all),
        matches=matches,
    )
