from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ScenePacketMeta:
    sample_id: str
    scene_id: str
    room_id: str
    camera_id: str
    source_id: int
    camera_intrinsics: Tensor
    camera_extrinsics: Tensor
    image_path: str = ""
    visible_obj_uid: list[str] | None = None
    hidden_obj_uid: list[str] | None = None


@dataclass(frozen=True)
class ScenePacketCondition:
    f_global: Tensor
    layout_token_cond: Tensor
    visible_tokens_cond: Tensor
    uncertainty_token: Tensor
    pose0_calib: Tensor
    layout0_calib: Tensor
    lock_gate: Tensor
    slot_confidence: Tensor
    visible_valid_mask: Tensor
    depth_obs: Tensor
    visible_union_mask: Tensor
    source_id: int | None = None


@dataclass(frozen=True)
class ScenePacketTarget:
    layout_gt: Tensor
    visible_cls_gt: Tensor
    visible_amodal_pose_gt: Tensor
    visible_amodal_res_gt: Tensor
    visible_z_gt: Tensor
    visible_obj_uid: list[str]
    visible_loss_mask: Tensor
    hidden_cls_gt: Tensor
    hidden_pose_gt: Tensor
    hidden_z_gt: Tensor
    hidden_obj_uid: list[str]
    hidden_gt_mask: Tensor
    support_gt: Tensor | None = None
    floor_gt: Tensor | None = None
    wall_gt: Tensor | None = None
    relation_valid_mask: Tensor | None = None


@dataclass(frozen=True)
class ScenePacketV1:
    meta: ScenePacketMeta
    condition: ScenePacketCondition
    target: ScenePacketTarget

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)
