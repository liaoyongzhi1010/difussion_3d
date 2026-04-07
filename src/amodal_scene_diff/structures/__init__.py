"""Shared batch structures for AmodalSceneDiff."""

from .scene_batch import (
    C_OBJ,
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
from .scene_packet import (
    ScenePacketCondition,
    ScenePacketMeta,
    ScenePacketTarget,
    ScenePacketV1,
)

__all__ = [
    "C_OBJ",
    "D_MODEL",
    "D_POSE",
    "K_HID",
    "K_VIS",
    "N_OBJ_MAX",
    "SceneConditionBatch",
    "SceneDiffusionBatch",
    "SceneMetaBatch",
    "ScenePacketCondition",
    "ScenePacketMeta",
    "ScenePacketTarget",
    "ScenePacketV1",
    "SceneSourceId",
    "SceneTargetBatch",
    "Z_DIM",
]
