from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from amodal_scene_diff.structures import (
    D_MODEL,
    D_POSE,
    K_HID,
    K_VIS,
    N_OBJ_MAX,
    Z_DIM,
    SceneConditionBatch,
    SceneDiffusionBatch,
    SceneMetaBatch,
    SceneSourceId,
    SceneTargetBatch,
)

Tensor = torch.Tensor
ScenePacket = Mapping[str, Any]
PAD_UID = "__pad__"
_MISSING = object()


def collate_scene_packets(samples: Sequence[ScenePacket]) -> SceneDiffusionBatch:
    """Collate logical ScenePacketV1-style samples into a SceneDiffusionBatch."""
    if not samples:
        raise ValueError("samples must be non-empty")

    cond = _collate_condition(samples)
    target = _collate_target(samples)
    meta = _collate_meta(samples, target)

    batch = SceneDiffusionBatch(cond=cond, target=target, meta=meta)
    batch.validate()
    return batch


def _collate_condition(samples: Sequence[ScenePacket]) -> SceneConditionBatch:
    return SceneConditionBatch(
        f_global=torch.stack(
            [_exact_1d(_field(_section(sample, "condition"), "f_global", index), D_MODEL) for index, sample in enumerate(samples)]
        ),
        layout_token_cond=torch.stack(
            [_singleton_token(_field(_section(sample, "condition"), "layout_token_cond", index), D_MODEL) for index, sample in enumerate(samples)]
        ),
        visible_tokens_cond=torch.stack(
            [_pad_2d(_field(_section(sample, "condition"), "visible_tokens_cond", index), K_VIS, D_MODEL) for index, sample in enumerate(samples)]
        ),
        uncertainty_token=torch.stack(
            [_singleton_token(_field(_section(sample, "condition"), "uncertainty_token", index), D_MODEL) for index, sample in enumerate(samples)]
        ),
        pose0_calib=torch.stack(
            [_pad_2d(_field(_section(sample, "condition"), "pose0_calib", index), K_VIS, D_POSE) for index, sample in enumerate(samples)]
        ),
        layout0_calib=torch.stack(
            [_exact_1d(_field(_section(sample, "condition"), "layout0_calib", index), D_POSE) for index, sample in enumerate(samples)]
        ),
        lock_gate=torch.stack(
            [_pad_column(_field(_section(sample, "condition"), "lock_gate", index), K_VIS) for index, sample in enumerate(samples)]
        ),
        slot_confidence=torch.stack(
            [_pad_column(_field(_section(sample, "condition"), "slot_confidence", index), K_VIS) for index, sample in enumerate(samples)]
        ),
        visible_valid_mask=torch.stack(
            [_pad_1d(_field(_section(sample, "condition"), "visible_valid_mask", index), K_VIS, dtype=torch.bool) for index, sample in enumerate(samples)]
        ),
        depth_obs=torch.stack(
            [_single_channel_image(_field(_section(sample, "condition"), "depth_obs", index)) for index, sample in enumerate(samples)]
        ),
        visible_union_mask=torch.stack(
            [_single_channel_image(_field(_section(sample, "condition"), "visible_union_mask", index), dtype=torch.bool) for index, sample in enumerate(samples)]
        ),
        source_id=torch.tensor([_source_id(sample, index) for index, sample in enumerate(samples)], dtype=torch.long),
    )


def _collate_target(samples: Sequence[ScenePacket]) -> SceneTargetBatch:
    visible_loss_masks = [
        _pad_1d(_field(_section(sample, "target"), "visible_loss_mask", index), K_VIS, dtype=torch.bool)
        for index, sample in enumerate(samples)
    ]
    hidden_gt_masks = [
        _pad_1d(_field(_section(sample, "target"), "hidden_gt_mask", index), K_HID, dtype=torch.bool)
        for index, sample in enumerate(samples)
    ]

    return SceneTargetBatch(
        layout_gt=torch.stack(
            [_exact_1d(_field(_section(sample, "target"), "layout_gt", index), D_POSE) for index, sample in enumerate(samples)]
        ),
        visible_cls_gt=torch.stack(
            [_pad_1d(_field(_section(sample, "target"), "visible_cls_gt", index), K_VIS, dtype=torch.long) for index, sample in enumerate(samples)]
        ),
        visible_amodal_pose_gt=torch.stack(
            [_pad_2d(_field(_section(sample, "target"), "visible_amodal_pose_gt", index), K_VIS, D_POSE) for index, sample in enumerate(samples)]
        ),
        visible_amodal_res_gt=torch.stack(
            [_pad_2d(_field(_section(sample, "target"), "visible_amodal_res_gt", index), K_VIS, D_POSE) for index, sample in enumerate(samples)]
        ),
        visible_z_gt=torch.stack(
            [_pad_2d(_field(_section(sample, "target"), "visible_z_gt", index), K_VIS, Z_DIM) for index, sample in enumerate(samples)]
        ),
        visible_loss_mask=torch.stack(visible_loss_masks),
        hidden_cls_gt=torch.stack(
            [_pad_1d(_field(_section(sample, "target"), "hidden_cls_gt", index), K_HID, dtype=torch.long) for index, sample in enumerate(samples)]
        ),
        hidden_pose_gt=torch.stack(
            [_pad_2d(_field(_section(sample, "target"), "hidden_pose_gt", index), K_HID, D_POSE) for index, sample in enumerate(samples)]
        ),
        hidden_z_gt=torch.stack(
            [_pad_2d(_field(_section(sample, "target"), "hidden_z_gt", index), K_HID, Z_DIM) for index, sample in enumerate(samples)]
        ),
        hidden_gt_mask=torch.stack(hidden_gt_masks),
        support_gt=torch.stack(
            [_optional_square(_section(sample, "target"), "support_gt", N_OBJ_MAX) for sample in samples]
        ),
        floor_gt=torch.stack(
            [_optional_vector(_section(sample, "target"), "floor_gt", N_OBJ_MAX) for sample in samples]
        ),
        wall_gt=torch.stack(
            [_optional_vector(_section(sample, "target"), "wall_gt", N_OBJ_MAX) for sample in samples]
        ),
        relation_valid_mask=torch.stack(
            [
                _relation_valid_mask(_section(sample, "target"), visible_loss_masks[index], hidden_gt_masks[index])
                for index, sample in enumerate(samples)
            ]
        ),
    )


def _collate_meta(samples: Sequence[ScenePacket], target: SceneTargetBatch) -> SceneMetaBatch:
    visible_obj_uid = [
        _uid_list(sample, index, key="visible_obj_uid", size=K_VIS) for index, sample in enumerate(samples)
    ]
    hidden_obj_uid = [
        _uid_list(sample, index, key="hidden_obj_uid", size=K_HID) for index, sample in enumerate(samples)
    ]

    return SceneMetaBatch(
        sample_ids=[str(_field(_section(sample, "meta"), "sample_id", index)) for index, sample in enumerate(samples)],
        scene_ids=[str(_field(_section(sample, "meta"), "scene_id", index)) for index, sample in enumerate(samples)],
        room_ids=[str(_field(_section(sample, "meta"), "room_id", index)) for index, sample in enumerate(samples)],
        camera_ids=[str(_field(_section(sample, "meta"), "camera_id", index)) for index, sample in enumerate(samples)],
        camera_intrinsics=torch.stack(
            [_exact_matrix(_field(_section(sample, "meta"), "camera_intrinsics", index), 3, 3) for index, sample in enumerate(samples)]
        ),
        camera_extrinsics=torch.stack(
            [_exact_matrix(_field(_section(sample, "meta"), "camera_extrinsics", index), 4, 4) for index, sample in enumerate(samples)]
        ),
        visible_obj_uid=visible_obj_uid,
        hidden_obj_uid=hidden_obj_uid,
    )


def _section(sample: ScenePacket, key: str) -> Mapping[str, Any]:
    value = sample.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"sample must contain mapping section {key}")
    return value


def _field(section: Mapping[str, Any], key: str, sample_index: int) -> Any:
    if key not in section:
        raise KeyError(f"missing field {key} in sample[{sample_index}]")
    return section[key]


def _maybe_field(section: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    return section.get(key, default)


def _source_id(sample: ScenePacket, sample_index: int) -> int:
    cond = _section(sample, "condition")
    meta = _section(sample, "meta")
    raw = _maybe_field(cond, "source_id", _maybe_field(meta, "source_id", _MISSING))
    if raw is _MISSING:
        raise KeyError(f"missing field source_id in sample[{sample_index}] condition/meta")
    value = int(raw)
    if value not in tuple(item.value for item in SceneSourceId):
        raise ValueError(f"invalid source_id={value} in sample[{sample_index}]")
    return value


def _uid_list(sample: ScenePacket, sample_index: int, key: str, size: int) -> list[str]:
    target = _section(sample, "target")
    meta = _section(sample, "meta")
    raw = _maybe_field(target, key, _maybe_field(meta, key, _MISSING))
    if raw is _MISSING:
        return [PAD_UID] * size
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError(f"{key} in sample[{sample_index}] must be a sequence of uids")
    values = [str(item) for item in raw[:size]]
    values.extend([PAD_UID] * (size - len(values)))
    return values


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _exact_1d(value: Any, size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim != 1 or tensor.shape[0] != size:
        raise ValueError(f"expected shape [{size}], got {tuple(tensor.shape)}")
    return tensor


def _pad_1d(value: Any, size: int, *, dtype: torch.dtype, pad_value: float | bool | int = 0) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim != 1:
        raise ValueError(f"expected rank-1 tensor, got {tuple(tensor.shape)}")
    tensor = tensor[:size]
    if tensor.shape[0] == size:
        return tensor
    pad = torch.full((size - tensor.shape[0],), pad_value, dtype=dtype)
    return torch.cat([tensor, pad], dim=0)


def _exact_matrix(value: Any, rows: int, cols: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim != 2 or tensor.shape != (rows, cols):
        raise ValueError(f"expected shape [{rows}, {cols}], got {tuple(tensor.shape)}")
    return tensor


def _pad_2d(value: Any, rows: int, cols: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"expected rank-2 tensor, got {tuple(tensor.shape)}")
    if tensor.shape[1] != cols:
        raise ValueError(f"expected width {cols}, got {tuple(tensor.shape)}")
    tensor = tensor[:rows]
    if tensor.shape[0] == rows:
        return tensor
    pad = torch.zeros((rows - tensor.shape[0], cols), dtype=dtype)
    return torch.cat([tensor, pad], dim=0)


def _singleton_token(value: Any, width: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape != (1, width):
        raise ValueError(f"expected shape [1, {width}], got {tuple(tensor.shape)}")
    return tensor


def _pad_column(value: Any, rows: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2 or tensor.shape[1] != 1:
        raise ValueError(f"expected shape [N, 1], got {tuple(tensor.shape)}")
    tensor = tensor[:rows]
    if tensor.shape[0] == rows:
        return tensor
    pad = torch.zeros((rows - tensor.shape[0], 1), dtype=dtype)
    return torch.cat([tensor, pad], dim=0)


def _single_channel_image(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected shape [1, H, W], got {tuple(tensor.shape)}")
    return tensor


def _optional_square(section: Mapping[str, Any], key: str, size: int) -> Tensor:
    raw = _maybe_field(section, key, _MISSING)
    if raw is _MISSING:
        return torch.zeros(size, size, dtype=torch.float32)
    tensor = _as_tensor(raw, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"expected square matrix for {key}, got {tuple(tensor.shape)}")
    tensor = tensor[:size, :size]
    if tensor.shape == (size, size):
        return tensor
    output = torch.zeros(size, size, dtype=torch.float32)
    output[: tensor.shape[0], : tensor.shape[1]] = tensor
    return output


def _optional_vector(section: Mapping[str, Any], key: str, size: int) -> Tensor:
    raw = _maybe_field(section, key, _MISSING)
    if raw is _MISSING:
        return torch.zeros(size, dtype=torch.float32)
    return _pad_1d(raw, size, dtype=torch.float32)


def _relation_valid_mask(
    section: Mapping[str, Any],
    visible_loss_mask: Tensor,
    hidden_gt_mask: Tensor,
) -> Tensor:
    raw = _maybe_field(section, "relation_valid_mask", _MISSING)
    if raw is not _MISSING:
        return _pad_1d(raw, N_OBJ_MAX, dtype=torch.bool)
    combined = torch.cat([visible_loss_mask.to(torch.bool), hidden_gt_mask.to(torch.bool)], dim=0)
    return _pad_1d(combined, N_OBJ_MAX, dtype=torch.bool)
