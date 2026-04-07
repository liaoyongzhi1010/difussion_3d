from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from amodal_scene_diff.structures import (
    D_MODEL,
    D_POSE,
    K_HID,
    K_VIS,
    N_OBJ_MAX,
    Z_DIM,
    ScenePacketCondition,
    ScenePacketMeta,
    ScenePacketTarget,
    ScenePacketV1,
)

Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize ScenePacketV1 `.pt` files from metadata/caches.")
    parser.add_argument("--config", default="configs/preprocess/materialize_packets_oracle.yaml")
    parser.add_argument("--index-jsonl", default=None)
    parser.add_argument("--scaffold-root", default=None)
    parser.add_argument("--geometry-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--source-id", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-missing-geometry", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config at {path} must load as dict")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"jsonl row must be dict: {path}")
            rows.append(row)
    return rows


def as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def exact_1d(value: Any, size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = as_tensor(value, dtype=dtype)
    if tensor.ndim != 1 or tensor.shape[0] != size:
        raise ValueError(f"expected shape [{size}], got {tuple(tensor.shape)}")
    return tensor


def exact_or_singleton_token(value: Any, width: int) -> Tensor:
    tensor = as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape != (1, width):
        raise ValueError(f"expected shape [1, {width}], got {tuple(tensor.shape)}")
    return tensor


def pad_2d(value: Any, rows: int, cols: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = as_tensor(value, dtype=dtype)
    if tensor.ndim != 2 or tensor.shape[1] != cols:
        raise ValueError(f"expected shape [N, {cols}], got {tuple(tensor.shape)}")
    tensor = tensor[:rows]
    if tensor.shape[0] == rows:
        return tensor
    pad = torch.zeros((rows - tensor.shape[0], cols), dtype=dtype)
    return torch.cat([tensor, pad], dim=0)


def pad_1d(value: Any, size: int, *, dtype: torch.dtype, pad_value: int | float | bool = 0) -> Tensor:
    tensor = as_tensor(value, dtype=dtype)
    if tensor.ndim != 1:
        raise ValueError(f"expected rank-1 tensor, got {tuple(tensor.shape)}")
    tensor = tensor[:size]
    if tensor.shape[0] == size:
        return tensor
    pad = torch.full((size - tensor.shape[0],), pad_value, dtype=dtype)
    return torch.cat([tensor, pad], dim=0)


def pad_uid_list(values: list[str], size: int) -> list[str]:
    trimmed = [str(value) for value in values[:size]]
    trimmed.extend(["__pad__"] * (size - len(trimmed)))
    return trimmed


def load_scaffold(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"scaffold cache must be dict: {path}")
    if "condition" in payload and isinstance(payload["condition"], dict):
        return payload["condition"]
    return payload


def load_geometry_latent(path: Path, allow_missing: bool) -> Tensor:
    if not path.exists():
        if allow_missing:
            return torch.zeros(Z_DIM, dtype=torch.float32)
        raise FileNotFoundError(f"missing geometry latent: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ["z_mu", "z", "latent"]:
            if key in payload:
                return exact_1d(payload[key], Z_DIM)
    if isinstance(payload, Tensor):
        return exact_1d(payload, Z_DIM)
    raise TypeError(f"unsupported geometry payload at {path}")


def _single_channel(value: Any, *, dtype: torch.dtype) -> Tensor:
    tensor = as_tensor(value, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected single-channel image [1,H,W], got {tuple(tensor.shape)}")
    return tensor


def build_condition(scaffold: dict[str, Any], source_id: int) -> ScenePacketCondition:
    return ScenePacketCondition(
        f_global=exact_1d(scaffold["f_global"], D_MODEL),
        layout_token_cond=exact_or_singleton_token(scaffold["layout_token_cond"], D_MODEL),
        visible_tokens_cond=pad_2d(scaffold["visible_tokens_cond"], K_VIS, D_MODEL),
        uncertainty_token=exact_or_singleton_token(scaffold["uncertainty_token"], D_MODEL),
        pose0_calib=pad_2d(scaffold["pose0_calib"], K_VIS, D_POSE),
        layout0_calib=exact_1d(scaffold["layout0_calib"], D_POSE),
        lock_gate=pad_2d(scaffold["lock_gate"], K_VIS, 1),
        slot_confidence=pad_2d(scaffold["slot_confidence"], K_VIS, 1),
        visible_valid_mask=pad_1d(scaffold["visible_valid_mask"], K_VIS, dtype=torch.bool),
        depth_obs=_single_channel(scaffold["depth_obs"], dtype=torch.float32),
        visible_union_mask=_single_channel(scaffold["visible_union_mask"], dtype=torch.bool),
        source_id=source_id,
    )


def _optional_square(value: Any, size: int) -> Tensor:
    if value is None:
        return torch.zeros(size, size, dtype=torch.float32)
    tensor = as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"expected rank-2 square tensor, got {tuple(tensor.shape)}")
    tensor = tensor[:size, :size]
    output = torch.zeros(size, size, dtype=torch.float32)
    output[: tensor.shape[0], : tensor.shape[1]] = tensor
    return output


def _optional_vector(value: Any, size: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if value is None:
        return torch.zeros(size, dtype=dtype)
    return pad_1d(value, size, dtype=dtype)


def build_target(metadata: dict[str, Any], geometry_root: Path, allow_missing_geometry: bool) -> ScenePacketTarget:
    visible_objects = list(metadata.get("visible_objects") or [])
    hidden_objects = list(metadata.get("hidden_objects") or [])
    relations = metadata.get("relations") or {}

    visible_uid = [str(obj["uid"]) for obj in visible_objects]
    hidden_uid = [str(obj["uid"]) for obj in hidden_objects]

    visible_cls = torch.tensor([int(obj.get("class_id", 0)) for obj in visible_objects], dtype=torch.long)
    hidden_cls = torch.tensor([int(obj.get("class_id", 0)) for obj in hidden_objects], dtype=torch.long)
    visible_pose = torch.stack([exact_1d(obj["amodal_pose_gt"], D_POSE) for obj in visible_objects], dim=0) if visible_objects else torch.zeros((0, D_POSE))
    visible_res = torch.stack([exact_1d(obj.get("amodal_res_gt", [0.0] * D_POSE), D_POSE) for obj in visible_objects], dim=0) if visible_objects else torch.zeros((0, D_POSE))
    hidden_pose = torch.stack([exact_1d(obj["pose_gt"], D_POSE) for obj in hidden_objects], dim=0) if hidden_objects else torch.zeros((0, D_POSE))

    visible_z = torch.stack([
        load_geometry_latent(geometry_root / f"{uid}.pt", allow_missing=allow_missing_geometry) for uid in visible_uid
    ], dim=0) if visible_uid else torch.zeros((0, Z_DIM))
    hidden_z = torch.stack([
        load_geometry_latent(geometry_root / f"{uid}.pt", allow_missing=allow_missing_geometry) for uid in hidden_uid
    ], dim=0) if hidden_uid else torch.zeros((0, Z_DIM))

    return ScenePacketTarget(
        layout_gt=exact_1d(metadata["layout_gt"], D_POSE),
        visible_cls_gt=visible_cls,
        visible_amodal_pose_gt=visible_pose,
        visible_amodal_res_gt=visible_res,
        visible_z_gt=visible_z,
        visible_obj_uid=pad_uid_list(visible_uid, K_VIS),
        visible_loss_mask=pad_1d([1] * len(visible_uid), K_VIS, dtype=torch.bool),
        hidden_cls_gt=hidden_cls,
        hidden_pose_gt=hidden_pose,
        hidden_z_gt=hidden_z,
        hidden_obj_uid=pad_uid_list(hidden_uid, K_HID),
        hidden_gt_mask=pad_1d([1] * len(hidden_uid), K_HID, dtype=torch.bool),
        support_gt=_optional_square(relations.get("support_gt"), N_OBJ_MAX),
        floor_gt=_optional_vector(relations.get("floor_gt"), N_OBJ_MAX),
        wall_gt=_optional_vector(relations.get("wall_gt"), N_OBJ_MAX),
        relation_valid_mask=_optional_vector(relations.get("relation_valid_mask"), N_OBJ_MAX, dtype=torch.bool),
    )


def build_meta(metadata: dict[str, Any], source_id: int) -> ScenePacketMeta:
    return ScenePacketMeta(
        sample_id=str(metadata["sample_id"]),
        scene_id=str(metadata["scene_id"]),
        room_id=str(metadata["room_id"]),
        camera_id=str(metadata["camera_id"]),
        source_id=source_id,
        camera_intrinsics=as_tensor(metadata["camera_intrinsics"], dtype=torch.float32),
        camera_extrinsics=as_tensor(metadata["camera_extrinsics"], dtype=torch.float32),
        image_path=str(metadata.get("rgb_path", "")),
        visible_obj_uid=[str(obj["uid"]) for obj in metadata.get("visible_objects", [])],
        hidden_obj_uid=[str(obj["uid"]) for obj in metadata.get("hidden_objects", [])],
    )


def load_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"metadata must be dict: {path}")
    return payload


def materialize_packets(cfg: dict[str, Any]) -> dict[str, Any]:
    index_jsonl = Path(cfg["index_jsonl"])
    scaffold_root = Path(cfg["scaffold_root"])
    geometry_root = Path(cfg["geometry_root"])
    output_root = Path(cfg["output_root"])
    source_id = int(cfg.get("source_id", 0))
    allow_missing_geometry = bool(cfg.get("allow_missing_geometry", False))
    max_samples = int(cfg.get("max_samples", 0))

    rows = read_jsonl(index_jsonl)
    if max_samples > 0:
        rows = rows[:max_samples]

    output_root.mkdir(parents=True, exist_ok=True)
    materialized = 0
    for row in rows:
        source_path = Path(row["source_path"])
        metadata = load_metadata(source_path)
        scaffold = load_scaffold(scaffold_root / f"{row['sample_id']}.pt")
        packet = ScenePacketV1(
            meta=build_meta(metadata, source_id=source_id),
            condition=build_condition(scaffold, source_id=source_id),
            target=build_target(metadata, geometry_root=geometry_root, allow_missing_geometry=allow_missing_geometry),
        )
        packet.save(output_root / f"{row['sample_id']}.pt")
        materialized += 1

    summary = {
        "num_packets": materialized,
        "index_jsonl": str(index_jsonl),
        "output_root": str(output_root),
        "source_id": source_id,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.index_jsonl is not None:
        cfg["index_jsonl"] = args.index_jsonl
    if args.scaffold_root is not None:
        cfg["scaffold_root"] = args.scaffold_root
    if args.geometry_root is not None:
        cfg["geometry_root"] = args.geometry_root
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.source_id is not None:
        cfg["source_id"] = args.source_id
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.allow_missing_geometry:
        cfg["allow_missing_geometry"] = True

    summary = materialize_packets(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
