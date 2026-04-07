from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from typing import Any

import torch

Tensor = torch.Tensor

K_VIS = 12
K_HID = 8
N_OBJ_MAX = 20
D_MODEL = 512
Z_DIM = 256
D_POSE = 8
C_OBJ = 10


class SceneSourceId(IntEnum):
    ORACLE = 0
    NOISY_ORACLE = 1
    PREDICTED = 2


def _map_value(value: Any, fn: Any) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, list):
        return [_map_value(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_value(item, fn) for item in value)
    return value


def _dataclass_map(instance: Any, fn: Any) -> Any:
    return replace(
        instance,
        **{field.name: _map_value(getattr(instance, field.name), fn) for field in fields(instance)},
    )


def _expect_rank(name: str, value: Tensor, rank: int) -> None:
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(value.shape)}")


def _expect_dim(name: str, value: Tensor, dim: int, size: int) -> None:
    if value.shape[dim] != size:
        raise ValueError(
            f"{name} must have size {size} at dim {dim}, got shape {tuple(value.shape)}"
        )


def _expect_batch(name: str, value: Tensor, batch_size: int) -> None:
    _expect_dim(name, value, 0, batch_size)


def _expect_list_len(name: str, value: list[Any], size: int) -> None:
    if len(value) != size:
        raise ValueError(f"{name} must have length {size}, got {len(value)}")


@dataclass(frozen=True)
class SceneConditionBatch:
    """Condition tensors consumed by the scene diffusion model."""

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
    source_id: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.f_global.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> "SceneConditionBatch":
        return _dataclass_map(self, lambda tensor: tensor.to(*args, **kwargs))

    def pin_memory(self) -> "SceneConditionBatch":
        return _dataclass_map(self, lambda tensor: tensor.pin_memory())

    def validate(self) -> None:
        batch_size = self.batch_size

        _expect_rank("cond.f_global", self.f_global, 2)
        _expect_dim("cond.f_global", self.f_global, 1, D_MODEL)

        _expect_rank("cond.layout_token_cond", self.layout_token_cond, 3)
        _expect_batch("cond.layout_token_cond", self.layout_token_cond, batch_size)
        _expect_dim("cond.layout_token_cond", self.layout_token_cond, 1, 1)
        _expect_dim("cond.layout_token_cond", self.layout_token_cond, 2, D_MODEL)

        _expect_rank("cond.visible_tokens_cond", self.visible_tokens_cond, 3)
        _expect_batch("cond.visible_tokens_cond", self.visible_tokens_cond, batch_size)
        _expect_dim("cond.visible_tokens_cond", self.visible_tokens_cond, 1, K_VIS)
        _expect_dim("cond.visible_tokens_cond", self.visible_tokens_cond, 2, D_MODEL)

        _expect_rank("cond.uncertainty_token", self.uncertainty_token, 3)
        _expect_batch("cond.uncertainty_token", self.uncertainty_token, batch_size)
        _expect_dim("cond.uncertainty_token", self.uncertainty_token, 1, 1)
        _expect_dim("cond.uncertainty_token", self.uncertainty_token, 2, D_MODEL)

        _expect_rank("cond.pose0_calib", self.pose0_calib, 3)
        _expect_batch("cond.pose0_calib", self.pose0_calib, batch_size)
        _expect_dim("cond.pose0_calib", self.pose0_calib, 1, K_VIS)
        _expect_dim("cond.pose0_calib", self.pose0_calib, 2, D_POSE)

        _expect_rank("cond.layout0_calib", self.layout0_calib, 2)
        _expect_batch("cond.layout0_calib", self.layout0_calib, batch_size)
        _expect_dim("cond.layout0_calib", self.layout0_calib, 1, D_POSE)

        _expect_rank("cond.lock_gate", self.lock_gate, 3)
        _expect_batch("cond.lock_gate", self.lock_gate, batch_size)
        _expect_dim("cond.lock_gate", self.lock_gate, 1, K_VIS)
        _expect_dim("cond.lock_gate", self.lock_gate, 2, 1)

        _expect_rank("cond.slot_confidence", self.slot_confidence, 3)
        _expect_batch("cond.slot_confidence", self.slot_confidence, batch_size)
        _expect_dim("cond.slot_confidence", self.slot_confidence, 1, K_VIS)
        _expect_dim("cond.slot_confidence", self.slot_confidence, 2, 1)

        _expect_rank("cond.visible_valid_mask", self.visible_valid_mask, 2)
        _expect_batch("cond.visible_valid_mask", self.visible_valid_mask, batch_size)
        _expect_dim("cond.visible_valid_mask", self.visible_valid_mask, 1, K_VIS)

        _expect_rank("cond.depth_obs", self.depth_obs, 4)
        _expect_batch("cond.depth_obs", self.depth_obs, batch_size)
        _expect_dim("cond.depth_obs", self.depth_obs, 1, 1)

        _expect_rank("cond.visible_union_mask", self.visible_union_mask, 4)
        _expect_batch("cond.visible_union_mask", self.visible_union_mask, batch_size)
        _expect_dim("cond.visible_union_mask", self.visible_union_mask, 1, 1)

        _expect_rank("cond.source_id", self.source_id, 1)
        _expect_batch("cond.source_id", self.source_id, batch_size)


@dataclass(frozen=True)
class SceneTargetBatch:
    """Supervision tensors and masks used by diffusion losses."""

    layout_gt: Tensor
    visible_cls_gt: Tensor
    visible_amodal_pose_gt: Tensor
    visible_amodal_res_gt: Tensor
    visible_z_gt: Tensor
    visible_loss_mask: Tensor
    hidden_cls_gt: Tensor
    hidden_pose_gt: Tensor
    hidden_z_gt: Tensor
    hidden_gt_mask: Tensor
    support_gt: Tensor
    floor_gt: Tensor
    wall_gt: Tensor
    relation_valid_mask: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.layout_gt.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> "SceneTargetBatch":
        return _dataclass_map(self, lambda tensor: tensor.to(*args, **kwargs))

    def pin_memory(self) -> "SceneTargetBatch":
        return _dataclass_map(self, lambda tensor: tensor.pin_memory())

    def validate(self) -> None:
        batch_size = self.batch_size

        _expect_rank("target.layout_gt", self.layout_gt, 2)
        _expect_batch("target.layout_gt", self.layout_gt, batch_size)
        _expect_dim("target.layout_gt", self.layout_gt, 1, D_POSE)

        _expect_rank("target.visible_cls_gt", self.visible_cls_gt, 2)
        _expect_batch("target.visible_cls_gt", self.visible_cls_gt, batch_size)
        _expect_dim("target.visible_cls_gt", self.visible_cls_gt, 1, K_VIS)

        _expect_rank("target.visible_amodal_pose_gt", self.visible_amodal_pose_gt, 3)
        _expect_batch("target.visible_amodal_pose_gt", self.visible_amodal_pose_gt, batch_size)
        _expect_dim("target.visible_amodal_pose_gt", self.visible_amodal_pose_gt, 1, K_VIS)
        _expect_dim("target.visible_amodal_pose_gt", self.visible_amodal_pose_gt, 2, D_POSE)

        _expect_rank("target.visible_amodal_res_gt", self.visible_amodal_res_gt, 3)
        _expect_batch("target.visible_amodal_res_gt", self.visible_amodal_res_gt, batch_size)
        _expect_dim("target.visible_amodal_res_gt", self.visible_amodal_res_gt, 1, K_VIS)
        _expect_dim("target.visible_amodal_res_gt", self.visible_amodal_res_gt, 2, D_POSE)

        _expect_rank("target.visible_z_gt", self.visible_z_gt, 3)
        _expect_batch("target.visible_z_gt", self.visible_z_gt, batch_size)
        _expect_dim("target.visible_z_gt", self.visible_z_gt, 1, K_VIS)
        _expect_dim("target.visible_z_gt", self.visible_z_gt, 2, Z_DIM)

        _expect_rank("target.visible_loss_mask", self.visible_loss_mask, 2)
        _expect_batch("target.visible_loss_mask", self.visible_loss_mask, batch_size)
        _expect_dim("target.visible_loss_mask", self.visible_loss_mask, 1, K_VIS)

        _expect_rank("target.hidden_cls_gt", self.hidden_cls_gt, 2)
        _expect_batch("target.hidden_cls_gt", self.hidden_cls_gt, batch_size)
        _expect_dim("target.hidden_cls_gt", self.hidden_cls_gt, 1, K_HID)

        _expect_rank("target.hidden_pose_gt", self.hidden_pose_gt, 3)
        _expect_batch("target.hidden_pose_gt", self.hidden_pose_gt, batch_size)
        _expect_dim("target.hidden_pose_gt", self.hidden_pose_gt, 1, K_HID)
        _expect_dim("target.hidden_pose_gt", self.hidden_pose_gt, 2, D_POSE)

        _expect_rank("target.hidden_z_gt", self.hidden_z_gt, 3)
        _expect_batch("target.hidden_z_gt", self.hidden_z_gt, batch_size)
        _expect_dim("target.hidden_z_gt", self.hidden_z_gt, 1, K_HID)
        _expect_dim("target.hidden_z_gt", self.hidden_z_gt, 2, Z_DIM)

        _expect_rank("target.hidden_gt_mask", self.hidden_gt_mask, 2)
        _expect_batch("target.hidden_gt_mask", self.hidden_gt_mask, batch_size)
        _expect_dim("target.hidden_gt_mask", self.hidden_gt_mask, 1, K_HID)

        _expect_rank("target.support_gt", self.support_gt, 3)
        _expect_batch("target.support_gt", self.support_gt, batch_size)
        _expect_dim("target.support_gt", self.support_gt, 1, N_OBJ_MAX)
        _expect_dim("target.support_gt", self.support_gt, 2, N_OBJ_MAX)

        _expect_rank("target.floor_gt", self.floor_gt, 2)
        _expect_batch("target.floor_gt", self.floor_gt, batch_size)
        _expect_dim("target.floor_gt", self.floor_gt, 1, N_OBJ_MAX)

        _expect_rank("target.wall_gt", self.wall_gt, 2)
        _expect_batch("target.wall_gt", self.wall_gt, batch_size)
        _expect_dim("target.wall_gt", self.wall_gt, 1, N_OBJ_MAX)

        _expect_rank("target.relation_valid_mask", self.relation_valid_mask, 2)
        _expect_batch("target.relation_valid_mask", self.relation_valid_mask, batch_size)
        _expect_dim("target.relation_valid_mask", self.relation_valid_mask, 1, N_OBJ_MAX)


@dataclass(frozen=True)
class SceneMetaBatch:
    """Non-learning metadata kept for tracing, eval, and visualization."""

    sample_ids: list[str]
    scene_ids: list[str]
    room_ids: list[str]
    camera_ids: list[str]
    camera_intrinsics: Tensor
    camera_extrinsics: Tensor
    visible_obj_uid: list[list[str]]
    hidden_obj_uid: list[list[str]]

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)

    def to(self, *args: Any, **kwargs: Any) -> "SceneMetaBatch":
        return _dataclass_map(self, lambda tensor: tensor.to(*args, **kwargs))

    def pin_memory(self) -> "SceneMetaBatch":
        return _dataclass_map(self, lambda tensor: tensor.pin_memory())

    def validate(self) -> None:
        batch_size = self.batch_size

        _expect_list_len("meta.scene_ids", self.scene_ids, batch_size)
        _expect_list_len("meta.room_ids", self.room_ids, batch_size)
        _expect_list_len("meta.camera_ids", self.camera_ids, batch_size)
        _expect_list_len("meta.visible_obj_uid", self.visible_obj_uid, batch_size)
        _expect_list_len("meta.hidden_obj_uid", self.hidden_obj_uid, batch_size)

        for index, uids in enumerate(self.visible_obj_uid):
            _expect_list_len(f"meta.visible_obj_uid[{index}]", uids, K_VIS)
        for index, uids in enumerate(self.hidden_obj_uid):
            _expect_list_len(f"meta.hidden_obj_uid[{index}]", uids, K_HID)

        _expect_rank("meta.camera_intrinsics", self.camera_intrinsics, 3)
        _expect_batch("meta.camera_intrinsics", self.camera_intrinsics, batch_size)
        _expect_dim("meta.camera_intrinsics", self.camera_intrinsics, 1, 3)
        _expect_dim("meta.camera_intrinsics", self.camera_intrinsics, 2, 3)

        _expect_rank("meta.camera_extrinsics", self.camera_extrinsics, 3)
        _expect_batch("meta.camera_extrinsics", self.camera_extrinsics, batch_size)
        _expect_dim("meta.camera_extrinsics", self.camera_extrinsics, 1, 4)
        _expect_dim("meta.camera_extrinsics", self.camera_extrinsics, 2, 4)


@dataclass(frozen=True)
class SceneDiffusionBatch:
    """Top-level nested batch object shared by train, eval, and visualization."""

    cond: SceneConditionBatch
    target: SceneTargetBatch
    meta: SceneMetaBatch

    @property
    def batch_size(self) -> int:
        return self.cond.batch_size

    def to(self, *args: Any, **kwargs: Any) -> "SceneDiffusionBatch":
        return SceneDiffusionBatch(
            cond=self.cond.to(*args, **kwargs),
            target=self.target.to(*args, **kwargs),
            meta=self.meta.to(*args, **kwargs),
        )

    def pin_memory(self) -> "SceneDiffusionBatch":
        return SceneDiffusionBatch(
            cond=self.cond.pin_memory(),
            target=self.target.pin_memory(),
            meta=self.meta.pin_memory(),
        )

    def validate(self) -> None:
        self.cond.validate()
        self.target.validate()
        self.meta.validate()

        batch_size = self.cond.batch_size
        if self.target.batch_size != batch_size:
            raise ValueError(
                f"target batch size mismatch: expected {batch_size}, got {self.target.batch_size}"
            )
        if self.meta.batch_size != batch_size:
            raise ValueError(
                f"meta batch size mismatch: expected {batch_size}, got {self.meta.batch_size}"
            )
