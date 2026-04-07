from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ViewRecord:
    sample_id: str
    scene_id: str
    room_id: str
    camera_id: str
    num_visible_total: int
    num_hidden_total: int
    num_major_objects: int
    floor_ratio: float
    wall_dominance: float
    valid_depth_ratio: float
    source_path: str
    payload: dict[str, Any]

    def as_index_entry(self, split: str) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "room_id": self.room_id,
            "camera_id": self.camera_id,
            "split": split,
            "num_visible_total": self.num_visible_total,
            "num_hidden_total": self.num_hidden_total,
            "num_major_objects": self.num_major_objects,
            "floor_ratio": self.floor_ratio,
            "wall_dominance": self.wall_dominance,
            "valid_depth_ratio": self.valid_depth_ratio,
            "source_path": self.source_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build room-level 3D-FRONT view index and splits.")
    parser.add_argument("--config", default="configs/preprocess/3dfront_index.yaml")
    parser.add_argument("--views-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-room-splits", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"config at {path} must load as dict")
    return data


def stable_hash_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def assign_room_split(room_id: str, split_ratio: list[float], seed: int) -> str:
    train_r, val_r, test_r = split_ratio
    value = stable_hash_fraction(room_id, seed)
    if value < train_r:
        return "train"
    if value < train_r + val_r:
        return "val"
    return "test"


def require_fields(payload: dict[str, Any], names: list[str], section_name: str, path: Path) -> None:
    missing = [name for name in names if name not in payload]
    if missing:
        raise KeyError(f"missing {section_name} fields {missing} in {path}")


def parse_view_record(path: Path, cfg: dict[str, Any]) -> ViewRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"view metadata must be dict: {path}")

    stats = payload.get("stats") or {}
    if not isinstance(stats, dict):
        raise TypeError(f"stats must be dict in {path}")

    fields_cfg = cfg.get("fields", {})
    require_fields(payload, list(fields_cfg.get("required_meta", [])), "meta", path)
    require_fields(stats, list(fields_cfg.get("required_stats", [])), "stats", path)

    return ViewRecord(
        sample_id=str(payload["sample_id"]),
        scene_id=str(payload["scene_id"]),
        room_id=str(payload["room_id"]),
        camera_id=str(payload["camera_id"]),
        num_visible_total=int(stats.get("num_visible_total", 0)),
        num_hidden_total=int(stats.get("num_hidden_total", 0)),
        num_major_objects=int(stats.get("num_major_objects", 0)),
        floor_ratio=float(stats.get("floor_ratio", 0.0)),
        wall_dominance=float(stats.get("wall_dominance", 1.0)),
        valid_depth_ratio=float(stats.get("valid_depth_ratio", 0.0)),
        source_path=str(path),
        payload=payload,
    )


def passes_filters(record: ViewRecord, cfg: dict[str, Any]) -> bool:
    filters = cfg.get("filters", {})
    if record.num_visible_total < int(filters.get("min_visible_total", 0)):
        return False
    if record.num_major_objects < int(filters.get("min_major_objects", 0)):
        return False
    if record.valid_depth_ratio < float(filters.get("min_valid_depth_ratio", 0.0)):
        return False
    if record.floor_ratio < float(filters.get("min_floor_ratio", 0.0)):
        return False
    if record.wall_dominance > float(filters.get("max_wall_dominance", 1.0)):
        return False
    if bool(filters.get("require_hidden_labels", False)) and record.num_hidden_total <= 0:
        return False
    return True


def collect_records(views_root: Path, cfg: dict[str, Any]) -> tuple[list[ViewRecord], list[Path]]:
    kept: list[ViewRecord] = []
    rejected: list[Path] = []
    ignore_names = set(cfg.get("ignore_filenames", ["summary.json"]))
    for path in sorted(views_root.glob("*.json")):
        if path.name in ignore_names:
            continue
        record = parse_view_record(path, cfg)
        if passes_filters(record, cfg):
            kept.append(record)
        else:
            rejected.append(path)
    return kept, rejected


def room_split_map(records: list[ViewRecord], split_ratio: list[float], seed: int) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for record in records:
        mapping.setdefault(record.room_id, assign_room_split(record.room_id, split_ratio, seed))
    return mapping


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_summary(records: list[ViewRecord], rejected: list[Path], room_to_split: dict[str, str]) -> dict[str, Any]:
    split_counts = {split: 0 for split in ["train", "val", "test"]}
    room_counts = {split: 0 for split in ["train", "val", "test"]}
    for split in room_to_split.values():
        room_counts[split] += 1
    for record in records:
        split_counts[room_to_split[record.room_id]] += 1

    return {
        "num_kept_views": len(records),
        "num_rejected_views": len(rejected),
        "num_rooms": len(room_to_split),
        "views_per_split": split_counts,
        "rooms_per_split": room_counts,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    views_root = Path(args.views_root or cfg["views_root"])
    output_root = Path(args.output_root or cfg["output_root"])
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    split_ratio = list(cfg.get("room_split", [0.8, 0.1, 0.1]))
    if len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("room_split must contain 3 values summing to 1")

    if not views_root.exists():
        raise FileNotFoundError(f"views root does not exist: {views_root}")

    records, rejected = collect_records(views_root, cfg)
    room_to_split = room_split_map(records, split_ratio=split_ratio, seed=seed)
    summary = build_summary(records, rejected, room_to_split)

    index_rows = [record.as_index_entry(room_to_split[record.room_id]) for record in records]
    split_to_sample_ids = {
        split: [row["sample_id"] for row in index_rows if row["split"] == split]
        for split in ["train", "val", "test"]
    }
    split_to_room_ids = {
        split: sorted([room_id for room_id, room_split in room_to_split.items() if room_split == split])
        for split in ["train", "val", "test"]
    }

    if args.dry_run:
        print(json.dumps({"summary": summary, "output_root": str(output_root)}, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "view_index.jsonl", index_rows)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "split_to_sample_ids.json", split_to_sample_ids)
    if args.write_room_splits:
        write_json(output_root / "split_to_room_ids.json", split_to_room_ids)

    print(json.dumps({"summary": summary, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
