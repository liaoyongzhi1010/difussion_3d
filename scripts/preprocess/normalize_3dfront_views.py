from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

D_POSE = 8
DEFAULT_LAYOUT_GT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
DEFAULT_CLASS_NAME_TO_ID = {
    "chair": 0,
    "table": 1,
    "sofa": 2,
    "bed": 3,
    "cabinet": 4,
    "wardrobe": 5,
    "desk": 6,
    "shelf": 7,
    "bookcase": 7,
    "nightstand": 8,
    "dresser": 8,
    "tv_stand": 9,
    "tvstand": 9,
    "console": 9,
}
DEFAULT_VIEW_LIST_KEYS = ["views", "camera_views", "samples"]
_MISSING = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize semi-structured 3D-FRONT view metadata into canonical per-view JSON files.")
    parser.add_argument("--config", default="configs/preprocess/normalize_3dfront_views.yaml")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--input-format", default=None, choices=["auto", "single_view", "jsonl_views", "room_bundle"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config at {path} must load as dict")
    return data


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"json payload must be dict: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"jsonl row must be dict at {path}:{line_num}")
            rows.append(row)
    return rows


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_by_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def first_present(payload: dict[str, Any], aliases: Iterable[str], default: Any = _MISSING) -> Any:
    for alias in aliases:
        value = get_by_path(payload, alias)
        if value is not _MISSING and value is not None:
            return value
    if default is _MISSING:
        raise KeyError(f"missing aliases {list(aliases)}")
    return default


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def ensure_pose(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != D_POSE:
        raise ValueError(f"{field_name} must be a list of length {D_POSE}")
    return [float(item) for item in value]


def ensure_matrix(value: Any, rows: int, cols: int, field_name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{field_name} must be a list with {rows} rows")
    normalized: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{field_name} rows must have length {cols}")
        normalized.append([float(item) for item in row])
    return normalized


def ensure_vector(value: Any, size: int, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field_name} must be a list of length {size}")
    return [float(item) for item in value]


def ensure_square_or_default(value: Any, size: int) -> list[list[float]]:
    output = [[0.0 for _ in range(size)] for _ in range(size)]
    if value is None:
        return output
    if not isinstance(value, list):
        raise ValueError("support relation must be a rank-2 list")
    for row_idx, row in enumerate(value[:size]):
        if not isinstance(row, list):
            raise ValueError("support relation rows must be lists")
        for col_idx, item in enumerate(row[:size]):
            output[row_idx][col_idx] = float(item)
    return output


def ensure_vector_or_default(value: Any, size: int) -> list[float]:
    output = [0.0 for _ in range(size)]
    if value is None:
        return output
    if not isinstance(value, list):
        raise ValueError("relation vector must be a list")
    for idx, item in enumerate(value[:size]):
        output[idx] = float(item)
    return output


def infer_input_format(path: Path, payload: dict[str, Any], configured_format: str) -> str:
    if configured_format != "auto":
        return configured_format
    if path.suffix.lower() == ".jsonl":
        return "jsonl_views"
    for key in DEFAULT_VIEW_LIST_KEYS:
        views = payload.get(key)
        if isinstance(views, list):
            return "room_bundle"
    return "single_view"


def normalize_class_id(raw_object: dict[str, Any], class_name_to_id: dict[str, int]) -> int:
    class_id = first_present(raw_object, ["class_id", "category_id", "label_id"], default=None)
    if class_id is not None:
        return int(class_id)
    class_name = first_present(raw_object, ["class_name", "category", "label", "semantic"], default=None)
    if class_name is None:
        raise KeyError("object is missing class_id/class_name")
    normalized = normalize_name(str(class_name))
    if normalized not in class_name_to_id:
        raise KeyError(f"unknown class name: {class_name}")
    return int(class_name_to_id[normalized])


def normalize_object(raw_object: dict[str, Any], *, kind: str, class_name_to_id: dict[str, int]) -> dict[str, Any]:
    uid = str(first_present(raw_object, ["uid", "instance_uid", "object_uid", "object_id", "id"]))
    class_id = normalize_class_id(raw_object, class_name_to_id)
    if kind == "visible":
        pose = ensure_pose(
            first_present(raw_object, ["amodal_pose_gt", "amodal_pose", "pose_gt", "pose"]),
            "visible.amodal_pose_gt",
        )
        amodal_res = ensure_pose(
            first_present(raw_object, ["amodal_res_gt", "amodal_res"], default=[0.0] * D_POSE),
            "visible.amodal_res_gt",
        )
        return {
            "uid": uid,
            "class_id": class_id,
            "amodal_pose_gt": pose,
            "amodal_res_gt": amodal_res,
        }
    pose = ensure_pose(
        first_present(raw_object, ["pose_gt", "amodal_pose_gt", "amodal_pose", "pose"]),
        "hidden.pose_gt",
    )
    return {
        "uid": uid,
        "class_id": class_id,
        "pose_gt": pose,
    }


def normalize_relations(payload: dict[str, Any], total_objects: int) -> dict[str, Any]:
    relations = first_present(payload, ["relations", "relation_labels"], default={})
    if not isinstance(relations, dict):
        raise TypeError("relations must be a dict when present")
    return {
        "support_gt": ensure_square_or_default(
            first_present(relations, ["support_gt", "support", "support_matrix"], default=None),
            total_objects,
        ),
        "floor_gt": ensure_vector_or_default(
            first_present(relations, ["floor_gt", "floor", "floor_contact"], default=None),
            total_objects,
        ),
        "wall_gt": ensure_vector_or_default(
            first_present(relations, ["wall_gt", "wall", "wall_contact"], default=None),
            total_objects,
        ),
        "relation_valid_mask": ensure_vector_or_default(
            first_present(relations, ["relation_valid_mask", "valid_mask", "relation_mask"], default=None),
            total_objects,
        ),
    }


def normalize_stats(
    payload: dict[str, Any],
    visible_objects: list[dict[str, Any]],
    hidden_objects: list[dict[str, Any]],
    stats_fallback: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    stats = first_present(payload, ["stats", "view_stats", "quality"], default={})
    if not isinstance(stats, dict):
        raise TypeError("stats must be a dict when present")

    total_major_default = len(visible_objects) + len(hidden_objects)
    raw_visible = list(first_present(payload, ["visible_objects", "visible_instances"], default=[]))
    raw_hidden = list(first_present(payload, ["hidden_objects", "hidden_instances"], default=[]))
    for raw_object in raw_visible + raw_hidden:
        if isinstance(raw_object, dict) and raw_object.get("is_major") is False:
            total_major_default -= 1

    def stat_value(names: list[str], default: Any = _MISSING) -> Any:
        try:
            return first_present(stats, names)
        except KeyError:
            for name in names:
                if name in stats_fallback and stats_fallback[name] is not None:
                    return stats_fallback[name]
            if default is _MISSING:
                raise
            return default

    num_visible_total = int(stat_value(["num_visible_total"], default=len(visible_objects)))
    num_hidden_total = int(stat_value(["num_hidden_total"], default=len(hidden_objects)))
    num_major_objects = int(stat_value(["num_major_objects"], default=total_major_default))

    if strict:
        floor_ratio = float(stat_value(["floor_ratio"]))
        wall_dominance = float(stat_value(["wall_dominance"]))
        valid_depth_ratio = float(stat_value(["valid_depth_ratio"]))
    else:
        floor_ratio = float(stat_value(["floor_ratio"], default=0.0))
        wall_dominance = float(stat_value(["wall_dominance"], default=1.0))
        valid_depth_ratio = float(stat_value(["valid_depth_ratio"], default=0.0))

    return {
        "num_visible_total": num_visible_total,
        "num_hidden_total": num_hidden_total,
        "num_major_objects": num_major_objects,
        "floor_ratio": floor_ratio,
        "wall_dominance": wall_dominance,
        "valid_depth_ratio": valid_depth_ratio,
    }


def normalize_view_payload(
    payload: dict[str, Any],
    *,
    class_name_to_id: dict[str, int],
    stats_fallback: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    sample_id = str(first_present(payload, ["sample_id", "view_id", "image_id"]))
    scene_id = str(first_present(payload, ["scene_id", "scene_uid"]))
    room_id = str(first_present(payload, ["room_id", "room_uid"]))
    camera_id = str(first_present(payload, ["camera_id"], default=sample_id))
    rgb_path = str(first_present(payload, ["rgb_path", "image_path", "rgb", "image.file"], default=""))

    camera_intrinsics = ensure_matrix(
        first_present(payload, ["camera_intrinsics", "intrinsics", "camera.intrinsics"]),
        3,
        3,
        "camera_intrinsics",
    )
    camera_extrinsics = ensure_matrix(
        first_present(payload, ["camera_extrinsics", "extrinsics", "camera.extrinsics"]),
        4,
        4,
        "camera_extrinsics",
    )
    layout_gt = ensure_vector(
        first_present(payload, ["layout_gt", "layout.pose", "layout_token"], default=DEFAULT_LAYOUT_GT),
        D_POSE,
        "layout_gt",
    )

    raw_visible_objects = first_present(payload, ["visible_objects", "visible_instances"], default=[])
    raw_hidden_objects = first_present(payload, ["hidden_objects", "hidden_instances"], default=[])
    if not isinstance(raw_visible_objects, list) or not isinstance(raw_hidden_objects, list):
        raise TypeError("visible_objects and hidden_objects must be lists")

    visible_objects = [normalize_object(raw_object, kind="visible", class_name_to_id=class_name_to_id) for raw_object in raw_visible_objects]
    hidden_objects = [normalize_object(raw_object, kind="hidden", class_name_to_id=class_name_to_id) for raw_object in raw_hidden_objects]
    relations = normalize_relations(payload, total_objects=len(visible_objects) + len(hidden_objects))
    stats = normalize_stats(
        payload,
        visible_objects=visible_objects,
        hidden_objects=hidden_objects,
        stats_fallback=stats_fallback,
        strict=strict,
    )

    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "room_id": room_id,
        "camera_id": camera_id,
        "rgb_path": rgb_path,
        "camera_intrinsics": camera_intrinsics,
        "camera_extrinsics": camera_extrinsics,
        "layout_gt": layout_gt,
        "visible_objects": visible_objects,
        "hidden_objects": hidden_objects,
        "relations": relations,
        "stats": stats,
    }


def iter_view_payloads(path: Path, payload: dict[str, Any], input_format: str) -> Iterable[tuple[dict[str, Any], str]]:
    if input_format == "single_view":
        yield payload, "single_view"
        return

    if input_format == "room_bundle":
        view_list_key = None
        for key in DEFAULT_VIEW_LIST_KEYS:
            if isinstance(payload.get(key), list):
                view_list_key = key
                break
        if view_list_key is None:
            raise KeyError(f"room bundle is missing one of view list keys: {DEFAULT_VIEW_LIST_KEYS}")
        bundle_defaults = {key: value for key, value in payload.items() if key != view_list_key}
        for view_payload in payload[view_list_key]:
            if not isinstance(view_payload, dict):
                raise TypeError(f"bundle view must be dict: {path}")
            yield deep_merge(bundle_defaults, view_payload), "room_bundle"
        return

    if input_format == "jsonl_views":
        for row in read_jsonl(path):
            yield row, "jsonl_views"
        return

    raise ValueError(f"unsupported input_format: {input_format}")


def discover_input_paths(input_root: Path, patterns: list[str], recursive: bool) -> list[Path]:
    discovered: set[Path] = set()
    for pattern in patterns:
        iterator = input_root.rglob(pattern) if recursive else input_root.glob(pattern)
        for path in iterator:
            if path.is_file():
                discovered.add(path)
    return sorted(discovered)


def normalize_views(cfg: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    input_format = str(cfg.get("input_format", "auto"))
    patterns = list(cfg.get("input_globs", ["*.json", "*.jsonl"]))
    recursive = bool(cfg.get("recursive", True))
    overwrite = bool(cfg.get("overwrite", False))
    strict = bool(cfg.get("strict", True))
    stats_fallback = dict(cfg.get("stats_fallback", {}))
    class_name_to_id = {
        normalize_name(str(key)): int(value)
        for key, value in dict(cfg.get("class_name_to_id", DEFAULT_CLASS_NAME_TO_ID)).items()
    }

    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")

    input_paths = discover_input_paths(input_root, patterns=patterns, recursive=recursive)
    format_counter: Counter[str] = Counter()
    sample_ids: list[str] = []
    failures: list[str] = []
    normalized_views: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    for path in input_paths:
        try:
            if path.suffix.lower() == ".jsonl":
                inferred = "jsonl_views" if input_format == "auto" else input_format
                for raw_payload, source_format in iter_view_payloads(path, {}, inferred):
                    canonical = normalize_view_payload(
                        raw_payload,
                        class_name_to_id=class_name_to_id,
                        stats_fallback=stats_fallback,
                        strict=strict,
                    )
                    sample_id = canonical["sample_id"]
                    if sample_id in seen_sample_ids:
                        raise ValueError(f"duplicate sample_id detected: {sample_id}")
                    seen_sample_ids.add(sample_id)
                    normalized_views.append(canonical)
                    sample_ids.append(sample_id)
                    format_counter[source_format] += 1
                continue

            payload = read_json(path)
            inferred = infer_input_format(path, payload, input_format)
            for raw_payload, source_format in iter_view_payloads(path, payload, inferred):
                canonical = normalize_view_payload(
                    raw_payload,
                    class_name_to_id=class_name_to_id,
                    stats_fallback=stats_fallback,
                    strict=strict,
                )
                sample_id = canonical["sample_id"]
                if sample_id in seen_sample_ids:
                    raise ValueError(f"duplicate sample_id detected: {sample_id}")
                seen_sample_ids.add(sample_id)
                normalized_views.append(canonical)
                sample_ids.append(sample_id)
                format_counter[source_format] += 1
        except Exception as exc:  # pragma: no cover - kept for CLI robustness
            failures.append(f"{path}: {exc}")

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_input_files": len(input_paths),
        "num_written_views": len(normalized_views),
        "format_breakdown": dict(format_counter),
        "sample_ids_preview": sample_ids[:10],
        "num_failures": len(failures),
        "failures": failures[:20],
        "strict": strict,
    }

    if dry_run:
        return summary

    output_root.mkdir(parents=True, exist_ok=True)
    for canonical in normalized_views:
        output_path = output_root / f"{canonical['sample_id']}.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
        output_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.input_root is not None:
        cfg["input_root"] = args.input_root
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.input_format is not None:
        cfg["input_format"] = args.input_format
    if args.overwrite:
        cfg["overwrite"] = True

    summary = normalize_views(cfg, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
