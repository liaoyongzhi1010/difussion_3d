"""Top-venue 3D evaluation metrics.

Public entry points:
    - chamfer_distance / scene_chamfer    — point-cloud Chamfer
    - fscore_at_thresholds / scene_fscore — F-score@{0.01, 0.02, 0.05} m
    - box_iou_3d / pairwise_box_iou_3d    — oriented 3D box IoU
    - amodal_detection_report             — visible/hidden TP/FP/FN with IoU matching
    - collision_rate / support_violation  — physical plausibility proxies
"""

from .box_iou import box_iou_3d, pairwise_box_iou_3d
from .chamfer import ObjectCloud, SceneChamferReport, chamfer_distance, scene_chamfer
from .collision import CollisionReport, SupportReport, collision_rate, support_violation
from .fscore import FScoreReport, fscore_at_thresholds, scene_fscore
from .hidden_recall import AmodalRecallReport, DetectionReport, amodal_detection_report

__all__ = [
    "ObjectCloud",
    "SceneChamferReport",
    "chamfer_distance",
    "scene_chamfer",
    "FScoreReport",
    "fscore_at_thresholds",
    "scene_fscore",
    "box_iou_3d",
    "pairwise_box_iou_3d",
    "AmodalRecallReport",
    "DetectionReport",
    "amodal_detection_report",
    "CollisionReport",
    "SupportReport",
    "collision_rate",
    "support_violation",
]
