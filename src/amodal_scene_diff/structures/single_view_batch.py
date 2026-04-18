from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from .scene_batch import SceneMetaBatch, SceneTargetBatch

Tensor = torch.Tensor


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


def _expect_batch(name: str, value: Tensor, batch_size: int) -> None:
    if int(value.shape[0]) != int(batch_size):
        raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class SingleViewConditionBatch:
    """Single-view observation tensors consumed by the paper pipeline."""

    obs_image: Tensor
    depth_obs: Tensor
    visible_union_mask: Tensor
    rgb_available: Tensor
    source_id: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.obs_image.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> "SingleViewConditionBatch":
        return _dataclass_map(self, lambda tensor: tensor.to(*args, **kwargs))

    def pin_memory(self) -> "SingleViewConditionBatch":
        return _dataclass_map(self, lambda tensor: tensor.pin_memory())

    def validate(self) -> None:
        batch_size = self.batch_size
        _expect_rank("cond.obs_image", self.obs_image, 4)
        _expect_batch("cond.obs_image", self.obs_image, batch_size)
        if int(self.obs_image.shape[1]) < 3:
            raise ValueError(f"cond.obs_image must have at least 3 channels, got {tuple(self.obs_image.shape)}")

        _expect_rank("cond.depth_obs", self.depth_obs, 4)
        _expect_batch("cond.depth_obs", self.depth_obs, batch_size)
        if int(self.depth_obs.shape[1]) != 1:
            raise ValueError(f"cond.depth_obs must be single-channel, got {tuple(self.depth_obs.shape)}")

        _expect_rank("cond.visible_union_mask", self.visible_union_mask, 4)
        _expect_batch("cond.visible_union_mask", self.visible_union_mask, batch_size)
        if int(self.visible_union_mask.shape[1]) != 1:
            raise ValueError(
                f"cond.visible_union_mask must be single-channel, got {tuple(self.visible_union_mask.shape)}"
            )

        _expect_rank("cond.rgb_available", self.rgb_available, 1)
        _expect_batch("cond.rgb_available", self.rgb_available, batch_size)

        _expect_rank("cond.source_id", self.source_id, 1)
        _expect_batch("cond.source_id", self.source_id, batch_size)


@dataclass(frozen=True)
class SingleViewSceneBatch:
    cond: SingleViewConditionBatch
    target: SceneTargetBatch
    meta: SceneMetaBatch

    @property
    def batch_size(self) -> int:
        return self.cond.batch_size

    def to(self, *args: Any, **kwargs: Any) -> "SingleViewSceneBatch":
        return SingleViewSceneBatch(
            cond=self.cond.to(*args, **kwargs),
            target=self.target.to(*args, **kwargs),
            meta=self.meta.to(*args, **kwargs),
        )

    def pin_memory(self) -> "SingleViewSceneBatch":
        return SingleViewSceneBatch(
            cond=self.cond.pin_memory(),
            target=self.target.pin_memory(),
            meta=self.meta.pin_memory(),
        )

    def validate(self) -> None:
        self.cond.validate()
        self.target.validate()
        self.meta.validate()
        if self.cond.batch_size != self.target.batch_size or self.cond.batch_size != self.meta.batch_size:
            raise ValueError(
                "single-view batch sections must agree on batch size: "
                f"cond={self.cond.batch_size}, target={self.target.batch_size}, meta={self.meta.batch_size}"
            )
