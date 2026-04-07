from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from amodal_scene_diff.structures import D_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize real bootstrap geometry latents and rebuild visible scaffold tokens.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--geometry-subdir", default="geometry")
    parser.add_argument("--scaffold-subdir", default="scaffold")
    parser.add_argument("--views-subdir", default="views")
    parser.add_argument("--clip", type=float, default=6.0)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((in_dim, out_dim), dtype=np.float32) / max(math.sqrt(in_dim), 1.0)


_PROJ_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def project_feature(raw: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    key = (raw.shape[0], out_dim, seed)
    matrix = _PROJ_CACHE.get(key)
    if matrix is None:
        matrix = projection_matrix(raw.shape[0], out_dim, seed)
        _PROJ_CACHE[key] = matrix
    return raw.astype(np.float32, copy=False) @ matrix


def load_latent(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ["z_mu", "z", "latent"]:
            if key in payload:
                return torch.as_tensor(payload[key], dtype=torch.float32).numpy()
    return torch.as_tensor(payload, dtype=torch.float32).numpy()


def compute_stats(geometry_root: Path, eps: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    latents = [load_latent(path) for path in sorted(geometry_root.glob("*.pt"))]
    if not latents:
        raise RuntimeError(f"no geometry latents found in {geometry_root}")
    arr = np.stack(latents, axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    summary = {
        "num_geometry": int(arr.shape[0]),
        "z_dim": int(arr.shape[1]),
        "raw_global_mean": float(arr.mean()),
        "raw_global_std": float(arr.std()),
        "raw_abs_max": float(np.abs(arr).max()),
        "per_dim_std_mean": float(arr.std(axis=0).mean()),
        "per_dim_std_max": float(arr.std(axis=0).max()),
    }
    return mean, std, summary


def normalize_latent(z: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    out = (z.astype(np.float32, copy=False) - mean) / std
    if clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32)


def rebuild_visible_tokens(view_payload: dict[str, Any], geometry_map: dict[str, np.ndarray]) -> np.ndarray:
    tokens = []
    for obj in view_payload.get("visible_objects", []):
        one_hot = np.zeros(10, dtype=np.float32)
        one_hot[int(obj["class_id"])] = 1.0
        pose = np.asarray(obj["amodal_pose_gt"], dtype=np.float32)
        z = geometry_map[str(obj["uid"])][:32]
        raw = np.concatenate([one_hot, pose, z], axis=0)
        tokens.append(project_feature(raw, D_MODEL, seed=17))
    if not tokens:
        return np.zeros((0, D_MODEL), dtype=np.float32)
    return np.stack(tokens, axis=0).astype(np.float32)


def normalize_subset(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    geometry_in = input_root / args.geometry_subdir
    scaffold_in = input_root / args.scaffold_subdir
    views_in = input_root / args.views_subdir
    geometry_out = output_root / args.geometry_subdir
    scaffold_out = output_root / args.scaffold_subdir
    views_out = output_root / args.views_subdir

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output root already exists: {output_root}")

    geometry_out.mkdir(parents=True, exist_ok=True)
    scaffold_out.mkdir(parents=True, exist_ok=True)
    views_out.mkdir(parents=True, exist_ok=True)

    mean, std, raw_summary = compute_stats(geometry_in, eps=args.eps)
    geometry_map: dict[str, np.ndarray] = {}

    for path in sorted(geometry_in.glob("*.pt")):
        uid = path.stem
        z = load_latent(path)
        z_norm = normalize_latent(z, mean, std, clip=args.clip)
        geometry_map[uid] = z_norm
        torch.save({"z_mu": torch.from_numpy(z_norm)}, geometry_out / path.name)

    scaffold_files = sorted(scaffold_in.glob("*.pt"))
    view_files = sorted(views_in.glob("*.json"))
    if len(scaffold_files) != len(view_files):
        raise RuntimeError("scaffold/views count mismatch")

    for view_path in view_files:
        payload = json.loads(view_path.read_text(encoding="utf-8"))
        sample_id = str(payload["sample_id"])
        scaffold = torch.load(scaffold_in / f"{sample_id}.pt", map_location="cpu")
        visible_tokens_cond = rebuild_visible_tokens(payload, geometry_map)
        scaffold["visible_tokens_cond"] = torch.from_numpy(visible_tokens_cond)
        torch.save(scaffold, scaffold_out / f"{sample_id}.pt")
        (views_out / view_path.name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    norm_arr = np.stack(list(geometry_map.values()), axis=0)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "clip": float(args.clip),
        "raw_stats": raw_summary,
        "normalized_global_mean": float(norm_arr.mean()),
        "normalized_global_std": float(norm_arr.std()),
        "normalized_abs_max": float(np.abs(norm_arr).max()),
        "num_scaffolds": len(scaffold_files),
    }
    (output_root / "geometry_stats.json").write_text(json.dumps({
        "mean": mean.tolist(),
        "std": std.tolist(),
        "summary": summary,
    }, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = normalize_subset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
