from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh


def _load_bootstrap_utils() -> Any:
    script = Path(__file__).resolve().parent / "export_pixarmesh_depr_bootstrap.py"
    spec = importlib.util.spec_from_file_location("bootstrap_export_utils", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load bootstrap exporter from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BOOTSTRAP = _load_bootstrap_utils()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export object-level real geometry dataset for geometry VAE training.")
    parser.add_argument("--parquet-path", action="append", default=[])
    parser.add_argument("--parquet-dir", default="")
    parser.add_argument("--metadata-zip", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-rows-per-parquet", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--num-surface-points", type=int, default=2048)
    parser.add_argument("--num-query-points", type=int, default=4096)
    parser.add_argument("--near-surface-ratio", type=float, default=0.7)
    parser.add_argument("--near-surface-sigma", type=float, default=0.02)
    parser.add_argument("--uniform-range", type=float, default=1.05)
    parser.add_argument("--watertight-required", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_parquet_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.parquet_path]
    if args.parquet_dir:
        paths.extend(sorted(Path(args.parquet_dir).glob("train-*.parquet")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise RuntimeError("no parquet paths resolved; pass --parquet-path or --parquet-dir")
    return unique


def uid_rng(uid: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha1(f"{uid}:{seed}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    if mesh.faces.size > 0:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def canonicalize_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise ValueError("empty mesh")
    mesh = _cleanup_mesh(mesh)
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise ValueError("mesh invalid after cleanup")

    bounds = mesh.bounds.astype(np.float32)
    center = ((bounds[0] + bounds[1]) * 0.5).astype(np.float32)
    half_extent = float(np.max(bounds[1] - bounds[0]) * 0.5)
    if half_extent <= 1.0e-6:
        raise ValueError("mesh has near-zero extent")

    canonical_vertices = (mesh.vertices.astype(np.float32) - center[None, :]) / half_extent
    canonical_mesh = trimesh.Trimesh(vertices=canonical_vertices, faces=mesh.faces.astype(np.int64), process=False)
    canonical_mesh = _cleanup_mesh(canonical_mesh)
    if canonical_mesh.faces.size == 0 or canonical_mesh.vertices.size == 0:
        raise ValueError("canonical mesh invalid after cleanup")

    meta = {
        "bbox_center": center.tolist(),
        "canonical_scale": float(half_extent),
        "num_vertices": int(canonical_mesh.vertices.shape[0]),
        "num_faces": int(canonical_mesh.faces.shape[0]),
        "is_watertight": bool(canonical_mesh.is_watertight),
        "is_winding_consistent": bool(canonical_mesh.is_winding_consistent),
        "euler_number": int(canonical_mesh.euler_number),
    }
    return canonical_mesh, meta


def _sample_surface_points(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    area_sum = float(areas.sum())
    if area_sum <= 1.0e-12:
        raise ValueError("mesh triangles have zero total area")
    probs = areas / area_sum
    face_index = rng.choice(len(probs), size=count, replace=True, p=probs)
    tri = triangles[face_index]
    r1 = rng.random((count, 1), dtype=np.float32)
    r2 = rng.random((count, 1), dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.concatenate([1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2], axis=1).astype(np.float32)
    points = (tri * bary[:, :, None]).sum(axis=1).astype(np.float32)
    return points, face_index.astype(np.int64)


def sample_surface(mesh: trimesh.Trimesh, num_surface_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points, face_index = _sample_surface_points(mesh, num_surface_points, rng)
    normals = mesh.face_normals[np.asarray(face_index, dtype=np.int64)]
    return points.astype(np.float32), normals.astype(np.float32)


def sample_queries(
    mesh: trimesh.Trimesh,
    num_query_points: int,
    near_surface_ratio: float,
    near_surface_sigma: float,
    uniform_range: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    near_count = int(round(num_query_points * near_surface_ratio))
    near_count = max(0, min(num_query_points, near_count))
    uniform_count = max(0, num_query_points - near_count)

    near_points = np.zeros((0, 3), dtype=np.float32)
    if near_count > 0:
        near_points, _ = _sample_surface_points(mesh, near_count, rng)
        near_points = near_points + rng.normal(0.0, near_surface_sigma, size=near_points.shape).astype(np.float32)
        near_points = np.clip(near_points, -uniform_range, uniform_range)

    uniform_points = np.zeros((0, 3), dtype=np.float32)
    if uniform_count > 0:
        uniform_points = rng.uniform(-uniform_range, uniform_range, size=(uniform_count, 3)).astype(np.float32)

    query_points = np.concatenate([near_points, uniform_points], axis=0)
    if query_points.shape[0] != num_query_points:
        raise RuntimeError(f"query point count mismatch: {query_points.shape[0]} vs {num_query_points}")

    query_sdf = trimesh.proximity.signed_distance(mesh, query_points).astype(np.float32)
    query_occ = (query_sdf > 0.0).astype(np.float32)
    return query_points, query_sdf[:, None], query_occ[:, None]


def export_object(
    *,
    cache: dict[str, Any],
    uid: str,
    class_id: int,
    output_root: Path,
    num_surface_points: int,
    num_query_points: int,
    near_surface_ratio: float,
    near_surface_sigma: float,
    uniform_range: float,
    watertight_required: bool,
    seed: int,
) -> dict[str, Any] | None:
    vertices = np.asarray(cache.get("vertices") or [], dtype=np.float32)
    faces = np.asarray(cache.get("faces") or [], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return None

    canonical_mesh, mesh_meta = canonicalize_mesh(vertices, faces)
    if watertight_required and not mesh_meta["is_watertight"]:
        return None

    rng = uid_rng(uid, seed)
    surface_points, surface_normals = sample_surface(canonical_mesh, num_surface_points=num_surface_points, rng=rng)
    query_points, query_sdf, query_occ = sample_queries(
        canonical_mesh,
        num_query_points=num_query_points,
        near_surface_ratio=near_surface_ratio,
        near_surface_sigma=near_surface_sigma,
        uniform_range=uniform_range,
        rng=rng,
    )

    payload = {
        "object_uid": uid,
        "scene_id": str(cache.get("scene_id", "")),
        "raw_instance_id": str(cache.get("raw_instance_id", "")),
        "room_id": str(cache.get("room_id", "")),
        "model_id": str(cache.get("model_id", "")),
        "category": str(cache.get("category", "")),
        "class_id": int(class_id),
        "surface_points": torch.from_numpy(surface_points),
        "surface_normals": torch.from_numpy(surface_normals),
        "query_points": torch.from_numpy(query_points),
        "query_sdf": torch.from_numpy(query_sdf),
        "query_occ": torch.from_numpy(query_occ),
        "vertices": torch.from_numpy(np.asarray(canonical_mesh.vertices, dtype=np.float32)),
        "faces": torch.from_numpy(np.asarray(canonical_mesh.faces, dtype=np.int64)),
        "sdf_sign_convention": "positive_inside",
        "quality_flag": "watertight" if mesh_meta["is_watertight"] else "non_watertight",
        "mesh_meta": mesh_meta,
    }
    torch.save(payload, output_root / "objects" / f"{uid}.pt")
    return {
        "uid": uid,
        "class_id": int(class_id),
        "quality_flag": payload["quality_flag"],
        **mesh_meta,
    }


def main() -> None:
    args = parse_args()
    parquet_paths = resolve_parquet_paths(args)
    output_root = Path(args.output_root)
    (output_root / "objects").mkdir(parents=True, exist_ok=True)
    metadata_zip = Path(args.metadata_zip)

    exported = 0
    skipped_no_mesh = 0
    skipped_invalid_mesh = 0
    skipped_non_watertight = 0
    seen_uid: set[str] = set()
    manifest: list[dict[str, Any]] = []

    for parquet_path in parquet_paths:
        max_rows = int(args.max_rows_per_parquet) if int(args.max_rows_per_parquet) > 0 else 10**9
        rows = BOOTSTRAP.row_batches(parquet_path, max_rows=max_rows)
        _, object_cache = BOOTSTRAP.build_object_cache(rows, metadata_zip)
        for scene_id, scene_objects in object_cache.items():
            for raw_id, cache in scene_objects.items():
                uid = BOOTSTRAP.safe_uid(scene_id, raw_id)
                if uid in seen_uid:
                    continue
                seen_uid.add(uid)
                class_id = BOOTSTRAP.canonical_class_id(str(cache.get("category", "")))
                if class_id is None:
                    continue
                if not cache.get("vertices") or not cache.get("faces"):
                    skipped_no_mesh += 1
                    continue
                try:
                    exported_meta = export_object(
                        cache=cache,
                        uid=uid,
                        class_id=int(class_id),
                        output_root=output_root,
                        num_surface_points=int(args.num_surface_points),
                        num_query_points=int(args.num_query_points),
                        near_surface_ratio=float(args.near_surface_ratio),
                        near_surface_sigma=float(args.near_surface_sigma),
                        uniform_range=float(args.uniform_range),
                        watertight_required=bool(args.watertight_required),
                        seed=int(args.seed),
                    )
                except ValueError:
                    skipped_invalid_mesh += 1
                    continue
                except Exception:
                    skipped_invalid_mesh += 1
                    continue
                if exported_meta is None:
                    if bool(args.watertight_required):
                        skipped_non_watertight += 1
                    else:
                        skipped_invalid_mesh += 1
                    continue
                manifest.append(exported_meta)
                exported += 1
                if int(args.max_objects) > 0 and exported >= int(args.max_objects):
                    break
            if int(args.max_objects) > 0 and exported >= int(args.max_objects):
                break
        if int(args.max_objects) > 0 and exported >= int(args.max_objects):
            break

    summary = {
        "output_root": str(output_root),
        "metadata_zip": str(metadata_zip),
        "parquet_paths": [str(path) for path in parquet_paths],
        "max_rows_per_parquet": int(args.max_rows_per_parquet),
        "max_objects": int(args.max_objects),
        "num_surface_points": int(args.num_surface_points),
        "num_query_points": int(args.num_query_points),
        "near_surface_ratio": float(args.near_surface_ratio),
        "near_surface_sigma": float(args.near_surface_sigma),
        "uniform_range": float(args.uniform_range),
        "watertight_required": bool(args.watertight_required),
        "exported_objects": exported,
        "skipped_no_mesh": skipped_no_mesh,
        "skipped_invalid_mesh": skipped_invalid_mesh,
        "skipped_non_watertight": skipped_non_watertight,
        "quality_hist": {
            "watertight": sum(1 for item in manifest if item["quality_flag"] == "watertight"),
            "non_watertight": sum(1 for item in manifest if item["quality_flag"] == "non_watertight"),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
