from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from amodal_scene_diff.structures import D_MODEL, D_POSE, K_HID, K_VIS, Z_DIM

IMAGE_SIZE = 512
MAX_REL = 20
_PROJECT_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real PixARMesh packed + DepR metadata into canonical views/scaffold/geometry caches.")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--metadata-zip", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--min-visible", type=int, default=1)
    parser.add_argument("--min-hidden", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_class_id(category: str) -> int | None:
    name = category.strip().lower()
    if any(token in name for token in ["wardrobe", "closet"]):
        return 5
    if any(token in name for token in ["nightstand"]):
        return 8
    if any(token in name for token in ["tv stand", "tv_stand", "tvstand"]):
        return 9
    if any(token in name for token in ["bookcase", "bookshelf", "shelf"]):
        return 7
    if any(token in name for token in ["desk", "dressing table"]):
        return 6
    if any(token in name for token in ["cabinet", "sideboard", "armoire", "drawer chest", "wine cabinet"]):
        return 4
    if any(token in name for token in ["bed"]):
        return 3
    if any(token in name for token in ["sofa"]):
        return 2
    if any(token in name for token in ["table"]):
        return 1
    if any(token in name for token in ["chair", "stool"]):
        return 0
    return None


def safe_uid(scene_id: str, raw_instance_id: str) -> str:
    return f"{scene_id}__{raw_instance_id.replace('/', '__')}"


def _projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    key = (in_dim, out_dim, seed)
    matrix = _PROJECT_CACHE.get(key)
    if matrix is None:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((in_dim, out_dim), dtype=np.float32) / max(math.sqrt(in_dim), 1.0)
        _PROJECT_CACHE[key] = matrix
    return matrix


def project_feature(raw: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    raw = raw.astype(np.float32, copy=False)
    return raw @ _projection_matrix(raw.shape[0], out_dim, seed)


def quaternion_to_yaw(quat: list[float]) -> float:
    x, y, z, w = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def matrix_to_yaw(transform: list[list[float]]) -> float:
    mat = np.asarray(transform, dtype=np.float32)
    return float(math.atan2(float(mat[0, 2]), float(mat[0, 0])))


def corners_from_bounds(bounds: np.ndarray) -> np.ndarray:
    lo = bounds[0]
    hi = bounds[1]
    return np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float32,
    )


def world_bbox_from_visible(bounds: list[list[float]], transform: list[list[float]]) -> tuple[np.ndarray, np.ndarray, float]:
    bounds_np = np.asarray(bounds, dtype=np.float32)
    tf = np.asarray(transform, dtype=np.float32)
    corners = corners_from_bounds(bounds_np)
    corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
    world = (tf @ corners_h.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0), matrix_to_yaw(transform)


def pose8_from_bbox(bbox_min: np.ndarray, bbox_max: np.ndarray, yaw: float) -> list[float]:
    center = 0.5 * (bbox_min + bbox_max)
    size = np.maximum(bbox_max - bbox_min, 1.0e-3)
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(np.log(size[0])),
        float(np.log(size[1])),
        float(np.log(size[2])),
        float(math.sin(yaw)),
        float(math.cos(yaw)),
    ]


def layout_gt_from_row(layout: dict[str, Any]) -> tuple[list[float], dict[str, float]]:
    bbox = np.asarray(layout["bbox"], dtype=np.float32)
    mins = bbox.min(axis=0)
    maxs = bbox.max(axis=0)
    center = 0.5 * (mins + maxs)
    size = np.maximum(maxs - mins, 1.0e-3)
    layout_gt = [
        float(center[0]),
        float(center[2]),
        float(np.log(size[0])),
        float(np.log(size[2])),
        float(mins[1]),
        float(np.log(size[1])),
        0.0,
        1.0,
    ]
    bounds = {
        "xmin": float(mins[0]),
        "xmax": float(maxs[0]),
        "zmin": float(mins[2]),
        "zmax": float(maxs[2]),
        "floor_y": float(mins[1]),
        "ceil_y": float(maxs[1]),
    }
    return layout_gt, bounds


def resize_image_bytes(payload: dict[str, Any], mode: str, is_mask: bool = False) -> np.ndarray:
    image = Image.open(BytesIO(payload["bytes"]))
    if mode:
        image = image.convert(mode)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=resample)
    return np.asarray(image)


def decode_depth_obs(payload: dict[str, Any]) -> np.ndarray:
    depth = resize_image_bytes(payload, mode="F")
    if depth.max() > 0:
        depth = depth / float(depth.max())
    return depth.astype(np.float32)


def decode_visible_union_mask(payload: dict[str, Any]) -> np.ndarray:
    mask = resize_image_bytes(payload, mode="RGB", is_mask=True)
    union = (mask.sum(axis=-1) > 0).astype(np.bool_)
    return union


def depth_summary(depth_obs: np.ndarray, visible_union_mask: np.ndarray) -> np.ndarray:
    valid = depth_obs > 0
    valid_values = depth_obs[valid]
    if valid_values.size == 0:
        valid_values = np.asarray([0.0], dtype=np.float32)
    hist, _ = np.histogram(valid_values, bins=8, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist = hist / max(hist.sum(), 1.0)
    summary = np.concatenate(
        [
            hist,
            np.asarray(
                [
                    float(valid_values.mean()),
                    float(valid_values.std()),
                    float(valid_values.min()),
                    float(valid_values.max()),
                    float(valid.mean()),
                    float(np.abs(np.diff(depth_obs, axis=0)).mean() + np.abs(np.diff(depth_obs, axis=1)).mean()),
                    float(visible_union_mask.mean()),
                    float((~valid).mean()),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return summary.astype(np.float32)


def geometry_latent_from_record(record: dict[str, Any]) -> np.ndarray:
    if "bbox_min" in record and "bbox_max" in record:
        bbox_min = np.asarray(record["bbox_min"], dtype=np.float32)
        bbox_max = np.asarray(record["bbox_max"], dtype=np.float32)
        size = bbox_max - bbox_min
        yaw = float(record.get("yaw", 0.0))
    else:
        size = np.maximum(np.asarray(record.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32), 1.0e-3)
        yaw = quaternion_to_yaw(list(record.get("rot", [0.0, 0.0, 0.0, 1.0])))
    base = [
        float(size[0]),
        float(size[1]),
        float(size[2]),
        float(np.log(max(size[0], 1.0e-3))),
        float(np.log(max(size[1], 1.0e-3))),
        float(np.log(max(size[2], 1.0e-3))),
        float(math.sin(yaw)),
        float(math.cos(yaw)),
    ]
    vertices = record.get("vertices") or []
    faces = record.get("faces") or []
    if vertices:
        verts = np.asarray(vertices, dtype=np.float32)
        if verts.ndim == 2 and verts.shape[1] == 3:
            center = verts.mean(axis=0, keepdims=True)
            verts = verts - center
            scale = float(np.abs(verts).max())
            if scale > 0:
                verts = verts / scale
            sample_count = min(64, verts.shape[0])
            sample_idx = np.linspace(0, verts.shape[0] - 1, sample_count, dtype=np.int32)
            sampled = verts[sample_idx].reshape(-1)
            stats = np.concatenate([verts.mean(axis=0), verts.std(axis=0), verts.min(axis=0), verts.max(axis=0)], axis=0)
            counts = np.asarray([float(len(vertices)), float(len(faces)), float(len(faces)) / max(float(len(vertices)), 1.0)], dtype=np.float32)
            raw = np.concatenate([np.asarray(base, dtype=np.float32), stats.astype(np.float32), counts, sampled.astype(np.float32)], axis=0)
        else:
            raw = np.asarray(base, dtype=np.float32)
    else:
        raw = np.asarray(base, dtype=np.float32)
    if raw.shape[0] < Z_DIM:
        raw = np.pad(raw, (0, Z_DIM - raw.shape[0]))
    return raw[:Z_DIM].astype(np.float32)


def compute_relations(objects: list[dict[str, Any]], layout_bounds: dict[str, float]) -> dict[str, Any]:
    support = np.zeros((MAX_REL, MAX_REL), dtype=np.float32)
    floor = np.zeros((MAX_REL,), dtype=np.float32)
    wall = np.zeros((MAX_REL,), dtype=np.float32)
    valid = np.zeros((MAX_REL,), dtype=np.float32)
    n = min(len(objects), MAX_REL)
    floor_y = layout_bounds["floor_y"]

    centers = []
    sizes = []
    for i, obj in enumerate(objects[:MAX_REL]):
        pose = np.asarray(obj["pose"], dtype=np.float32)
        center = pose[:3]
        size = np.exp(pose[3:6])
        centers.append(center)
        sizes.append(size)
        valid[i] = 1.0
        bottom = center[1] - 0.5 * size[1]
        floor[i] = 1.0 if abs(float(bottom - floor_y)) < 0.15 else 0.0
        margin = min(
            abs(float(center[0] - layout_bounds["xmin"])),
            abs(float(layout_bounds["xmax"] - center[0])),
            abs(float(center[2] - layout_bounds["zmin"])),
            abs(float(layout_bounds["zmax"] - center[2])),
        )
        wall[i] = 1.0 if margin < max(0.15, 0.1 * max(float(size[0]), float(size[2]))) else 0.0

    for i in range(n):
        ci = centers[i]
        si = sizes[i]
        bottom_i = ci[1] - 0.5 * si[1]
        for j in range(n):
            if i == j:
                continue
            cj = centers[j]
            sj = sizes[j]
            top_j = cj[1] + 0.5 * sj[1]
            overlap_x = max(0.0, min(ci[0] + 0.5 * si[0], cj[0] + 0.5 * sj[0]) - max(ci[0] - 0.5 * si[0], cj[0] - 0.5 * sj[0]))
            overlap_z = max(0.0, min(ci[2] + 0.5 * si[2], cj[2] + 0.5 * sj[2]) - max(ci[2] - 0.5 * si[2], cj[2] - 0.5 * sj[2]))
            overlap_area = overlap_x * overlap_z
            area_i = max(float(si[0] * si[2]), 1.0e-6)
            area_j = max(float(sj[0] * sj[2]), 1.0e-6)
            overlap_ratio = overlap_area / max(min(area_i, area_j), 1.0e-6)
            if abs(float(bottom_i - top_j)) < 0.12 and overlap_ratio > 0.2:
                support[i, j] = 1.0

    return {
        "support_gt": support.tolist(),
        "floor_gt": floor.tolist(),
        "wall_gt": wall.tolist(),
        "relation_valid_mask": valid.tolist(),
    }


def row_batches(parquet_path: Path, max_rows: int) -> list[dict[str, Any]]:
    pf = pq.ParquetFile(parquet_path)
    rows: list[dict[str, Any]] = []
    columns = ["uid", "scene_id", "K", "wrd2cam", "objects", "layout", "depth", "panoptic_mask"]
    for batch in pf.iter_batches(batch_size=8, columns=columns):
        for row in batch.to_pylist():
            rows.append(row)
            if len(rows) >= max_rows:
                return rows
    return rows


def load_scene_metadata(zf: zipfile.ZipFile, uid: int) -> list[dict[str, Any]]:
    with zf.open(f"metadata/{uid}.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_object_cache(rows: list[dict[str, Any]], metadata_zip: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    scene_meta: dict[str, list[dict[str, Any]]] = {}
    object_cache: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with zipfile.ZipFile(metadata_zip) as zf:
        for row in rows:
            scene_id = str(row["scene_id"])
            uid = int(row["uid"])
            if scene_id not in scene_meta:
                meta_rows = load_scene_metadata(zf, uid)
                scene_meta[scene_id] = meta_rows
                for meta in meta_rows:
                    raw_id = str(meta["raw_instance_id"])
                    object_cache[scene_id].setdefault(
                        raw_id,
                        {
                            "scene_id": scene_id,
                            "raw_instance_id": raw_id,
                            "room_id": str(meta["room_id"]),
                            "category": str(meta["category"]),
                            "category_id": int(meta["category_id"]),
                            "model_id": str(meta["model_id"]),
                            "pos": [float(v) for v in meta["pos"]],
                            "rot": [float(v) for v in meta["rot"]],
                            "scale": [float(v) for v in meta["scale"]],
                        },
                    )
            objects = row["objects"]
            for idx, raw_id in enumerate(objects["raw_inst_ids"]):
                cache = object_cache[scene_id].setdefault(str(raw_id), {"scene_id": scene_id, "raw_instance_id": str(raw_id)})
                bbox_min, bbox_max, yaw = world_bbox_from_visible(objects["bounds"][idx], objects["transforms"][idx])
                cache["bbox_min"] = bbox_min.tolist()
                cache["bbox_max"] = bbox_max.tolist()
                cache["yaw"] = yaw
                cache["vertices"] = objects["vertices"][idx]
                cache["faces"] = objects["faces"][idx]
                cache["inst_id"] = int(objects["inst_ids"][idx])
                cache["pifu_id"] = int(objects["pifu_ids"][idx])
    return scene_meta, object_cache


def object_pose(cache: dict[str, Any]) -> list[float] | None:
    if "bbox_min" in cache and "bbox_max" in cache:
        return pose8_from_bbox(np.asarray(cache["bbox_min"], dtype=np.float32), np.asarray(cache["bbox_max"], dtype=np.float32), float(cache.get("yaw", 0.0)))
    pos = cache.get("pos")
    scale = cache.get("scale")
    rot = cache.get("rot")
    if pos is None or scale is None or rot is None:
        return None
    size = np.maximum(np.asarray(scale, dtype=np.float32), 1.0e-3)
    yaw = quaternion_to_yaw(list(rot))
    return [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(np.log(size[0])),
        float(np.log(size[1])),
        float(np.log(size[2])),
        float(math.sin(yaw)),
        float(math.cos(yaw)),
    ]


def build_scaffold(
    visible_objects: list[dict[str, Any]],
    layout_gt: list[float],
    hidden_count: int,
    depth_obs: np.ndarray,
    visible_union_mask: np.ndarray,
    geometry_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    visible_count = len(visible_objects)
    depth_stats = depth_summary(depth_obs, visible_union_mask)
    layout_raw = np.asarray(layout_gt, dtype=np.float32)
    global_raw = np.concatenate(
        [
            layout_raw,
            depth_stats,
            np.asarray([
                float(visible_count) / max(K_VIS, 1),
                float(hidden_count) / max(K_HID, 1),
                float(visible_union_mask.mean()),
            ], dtype=np.float32),
        ],
        axis=0,
    )
    uncertainty_raw = np.asarray(
        [
            float(hidden_count),
            float(visible_count),
            float(depth_stats[12]),
            float(depth_stats[14]),
        ],
        dtype=np.float32,
    )

    visible_tokens = []
    pose0 = []
    for obj in visible_objects[:K_VIS]:
        one_hot = np.zeros(10, dtype=np.float32)
        one_hot[int(obj["class_id"])] = 1.0
        pose = np.asarray(obj["amodal_pose_gt"], dtype=np.float32)
        z = geometry_cache[obj["uid"]][:32]
        raw = np.concatenate([one_hot, pose, z], axis=0)
        visible_tokens.append(project_feature(raw, D_MODEL, seed=17))
        pose0.append(pose)

    if visible_tokens:
        visible_tokens_cond = np.stack(visible_tokens, axis=0).astype(np.float32)
        pose0_calib = np.stack(pose0, axis=0).astype(np.float32)
        slot_confidence = np.ones((len(visible_tokens), 1), dtype=np.float32)
        lock_gate = np.ones((len(visible_tokens), 1), dtype=np.float32)
        visible_valid_mask = np.ones((len(visible_tokens),), dtype=np.bool_)
    else:
        visible_tokens_cond = np.zeros((0, D_MODEL), dtype=np.float32)
        pose0_calib = np.zeros((0, D_POSE), dtype=np.float32)
        slot_confidence = np.zeros((0, 1), dtype=np.float32)
        lock_gate = np.zeros((0, 1), dtype=np.float32)
        visible_valid_mask = np.zeros((0,), dtype=np.bool_)

    return {
        "f_global": torch.from_numpy(project_feature(global_raw, D_MODEL, seed=11)),
        "layout_token_cond": torch.from_numpy(project_feature(layout_raw, D_MODEL, seed=13)).unsqueeze(0),
        "visible_tokens_cond": torch.from_numpy(visible_tokens_cond),
        "uncertainty_token": torch.from_numpy(project_feature(uncertainty_raw, D_MODEL, seed=19)).unsqueeze(0),
        "pose0_calib": torch.from_numpy(pose0_calib),
        "layout0_calib": torch.tensor(layout_gt, dtype=torch.float32),
        "lock_gate": torch.from_numpy(lock_gate),
        "slot_confidence": torch.from_numpy(slot_confidence),
        "visible_valid_mask": torch.from_numpy(visible_valid_mask),
        "depth_obs": torch.from_numpy(depth_obs[None, ...].astype(np.float32)),
        "visible_union_mask": torch.from_numpy(visible_union_mask[None, ...].astype(np.bool_)),
    }


def export_real_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    parquet_path = Path(args.parquet_path)
    metadata_zip = Path(args.metadata_zip)
    output_root = Path(args.output_root)
    views_root = output_root / "views"
    scaffold_root = output_root / "scaffold"
    geometry_root = output_root / "geometry"

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output root already exists: {output_root}")

    views_root.mkdir(parents=True, exist_ok=True)
    scaffold_root.mkdir(parents=True, exist_ok=True)
    geometry_root.mkdir(parents=True, exist_ok=True)

    rows = row_batches(parquet_path, max_rows=args.max_rows)
    scene_meta, object_cache = build_object_cache(rows, metadata_zip)

    geometry_written: set[str] = set()
    kept = 0
    hidden_hist: list[int] = []
    visible_hist: list[int] = []

    for row in rows:
        scene_id = str(row["scene_id"])
        uid = int(row["uid"])
        sample_id = str(uid)
        full_meta = scene_meta[scene_id]
        cache_for_scene = object_cache[scene_id]
        visible_raw_ids = [str(x) for x in row["objects"]["raw_inst_ids"]]
        room_votes = Counter(cache_for_scene[raw_id]["room_id"] for raw_id in visible_raw_ids if raw_id in cache_for_scene)
        if not room_votes:
            continue
        room_id = room_votes.most_common(1)[0][0]

        layout_gt, layout_bounds = layout_gt_from_row(row["layout"])
        depth_obs = decode_depth_obs(row["depth"])
        visible_union_mask = decode_visible_union_mask(row["panoptic_mask"])
        valid_depth_ratio = float((depth_obs > 0).mean())

        room_objects = [meta for meta in full_meta if str(meta["room_id"]) == room_id]
        visible_set = set(visible_raw_ids)

        visible_objects: list[dict[str, Any]] = []
        hidden_objects: list[dict[str, Any]] = []
        geometry_cache: dict[str, np.ndarray] = {}

        visible_order = {raw_id: idx for idx, raw_id in enumerate(visible_raw_ids)}
        visible_candidates = []
        hidden_candidates = []
        for meta in room_objects:
            raw_id = str(meta["raw_instance_id"])
            cache = cache_for_scene.get(raw_id, {})
            cls_id = canonical_class_id(str(meta["category"]))
            if cls_id is None:
                continue
            pose = object_pose(cache)
            if pose is None:
                continue
            uid_str = safe_uid(scene_id, raw_id)
            record = {
                "uid": uid_str,
                "class_id": int(cls_id),
                "pose": pose,
                "raw_instance_id": raw_id,
            }
            geometry_cache[uid_str] = geometry_latent_from_record(cache)
            if raw_id in visible_set:
                visible_candidates.append((visible_order.get(raw_id, 10**9), record))
            else:
                hidden_candidates.append(record)

        visible_candidates.sort(key=lambda item: item[0])
        visible_objects = [
            {
                "uid": item[1]["uid"],
                "class_id": item[1]["class_id"],
                "amodal_pose_gt": item[1]["pose"],
                "amodal_res_gt": [0.0] * D_POSE,
            }
            for item in visible_candidates[:K_VIS]
        ]
        hidden_objects = [
            {
                "uid": item["uid"],
                "class_id": item["class_id"],
                "pose_gt": item["pose"],
            }
            for item in hidden_candidates[:K_HID]
        ]

        if len(visible_objects) < args.min_visible or len(hidden_objects) < args.min_hidden:
            continue

        relation_objects = [
            {"pose": obj["amodal_pose_gt"]} for obj in visible_objects
        ] + [
            {"pose": obj["pose_gt"]} for obj in hidden_objects
        ]
        relations = compute_relations(relation_objects, layout_bounds)

        view_payload = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "room_id": room_id,
            "camera_id": sample_id,
            "camera_intrinsics": row["K"],
            "camera_extrinsics": row["wrd2cam"],
            "layout_gt": layout_gt,
            "visible_objects": visible_objects,
            "hidden_objects": hidden_objects,
            "relations": relations,
            "stats": {
                "num_visible_total": len(visible_objects),
                "num_hidden_total": len(hidden_objects),
                "num_major_objects": len(visible_objects) + len(hidden_objects),
                "floor_ratio": 0.0,
                "wall_dominance": 0.0,
                "valid_depth_ratio": valid_depth_ratio,
            },
        }
        (views_root / f"{sample_id}.json").write_text(json.dumps(view_payload, indent=2), encoding="utf-8")

        scaffold = build_scaffold(
            visible_objects=visible_objects,
            layout_gt=layout_gt,
            hidden_count=len(hidden_objects),
            depth_obs=depth_obs,
            visible_union_mask=visible_union_mask,
            geometry_cache=geometry_cache,
        )
        torch.save(scaffold, scaffold_root / f"{sample_id}.pt")

        for uid_str, latent in geometry_cache.items():
            if uid_str in geometry_written:
                continue
            torch.save({"z_mu": torch.from_numpy(latent)}, geometry_root / f"{uid_str}.pt")
            geometry_written.add(uid_str)

        kept += 1
        hidden_hist.append(len(hidden_objects))
        visible_hist.append(len(visible_objects))

    summary = {
        "parquet_path": str(parquet_path),
        "metadata_zip": str(metadata_zip),
        "output_root": str(output_root),
        "num_input_rows": len(rows),
        "num_exported_views": kept,
        "num_geometry_files": len(geometry_written),
        "avg_visible": float(sum(visible_hist) / max(len(visible_hist), 1)),
        "avg_hidden": float(sum(hidden_hist) / max(len(hidden_hist), 1)),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = export_real_bootstrap(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
