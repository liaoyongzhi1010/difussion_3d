from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

CANONICAL_PACKET_ROOT = Path("outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets")
CANONICAL_SPLIT_JSON = Path("outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json")
CANONICAL_ROOM_SPLIT_JSON = Path("outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_room_ids.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight harness for the single-view paper mainline.")
    parser.add_argument("--mode", required=True, choices=["train", "eval_state", "eval_render", "export"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--split-json", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--allow-noncanonical-packet-root", action="store_true")
    parser.add_argument("--allow-noncanonical-split", action="store_true")
    parser.add_argument("--save-summary", default="")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"yaml config at {path} must load as dict")
    return data


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _check_room_split_disjoint(room_split_path: Path) -> dict[str, Any]:
    if not room_split_path.exists():
        return {"exists": False, "disjoint": None, "counts": {}}
    payload = json.loads(room_split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"room split json must be dict: {room_split_path}")
    rooms = {key: set(value) for key, value in payload.items() if isinstance(value, list)}
    overlaps = {}
    for first in sorted(rooms):
        for second in sorted(rooms):
            if first >= second:
                continue
            overlaps[f"{first}__{second}"] = len(rooms[first] & rooms[second])
    return {
        "exists": True,
        "disjoint": all(count == 0 for count in overlaps.values()),
        "counts": {key: len(value) for key, value in rooms.items()},
        "overlaps": overlaps,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    runtime_cfg = load_yaml(args.runtime_config)

    packet_dir = normalize_path(args.packet_dir)
    config_path = normalize_path(args.config)
    data_config_path = normalize_path(args.data_config)
    runtime_config_path = normalize_path(args.runtime_config)
    split_json = normalize_path(args.split_json) if args.split_json else normalize_path(data_cfg.get("split_json", CANONICAL_SPLIT_JSON))
    output_root = normalize_path(args.output_root) if args.output_root else None

    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("observation_backbone", {})
    room_split_info = _check_room_split_disjoint(CANONICAL_ROOM_SPLIT_JSON.resolve())

    violations: list[str] = []

    if str(model_cfg.get("arch", "")) != "single_view_scene_v1":
        violations.append("model.arch must remain single_view_scene_v1 for the paper mainline")
    if "single_view_visible_direct_hidden_diffusion" not in str(model_cfg.get("name", Path(args.config).stem)):
        violations.append("config must stay on the single_view_visible_direct_hidden_diffusion family")
    if str(backbone_cfg.get("type", "")).lower() not in {"transformers_dinov2", "transformers_dinov2_hybrid"}:
        violations.append("observation_backbone.type must use an official DINOv2-based backbone, not patch_vit")
    if "dinov2" not in str(backbone_cfg.get("model_name", "")).lower():
        violations.append("observation_backbone.model_name must stay on a DINOv2 family checkpoint")
    if bool(backbone_cfg.get("allow_pseudo_rgb", False)):
        violations.append("allow_pseudo_rgb must remain false on the paper mainline")
    if int(model_cfg.get("obs_channels", 0)) < 4:
        violations.append("obs_channels must keep RGB plus depth-aware observation channels")

    canonical_packet_root = CANONICAL_PACKET_ROOT.resolve()
    if not args.allow_noncanonical_packet_root and packet_dir != canonical_packet_root:
        violations.append(
            f"packet_dir must stay on the canonical real-data mainline root: {canonical_packet_root}"
        )
    if not args.allow_noncanonical_split and split_json != CANONICAL_SPLIT_JSON.resolve():
        violations.append(
            f"split_json must stay on the canonical room-disjoint split file: {CANONICAL_SPLIT_JSON.resolve()}"
        )
    if room_split_info["exists"] and not room_split_info["disjoint"]:
        violations.append("room-level split file is not disjoint across train/val/test")

    summary = {
        "mode": args.mode,
        "config": str(config_path),
        "data_config": str(data_config_path),
        "runtime_config": str(runtime_config_path),
        "packet_dir": str(packet_dir),
        "split_json": str(split_json),
        "output_root": str(output_root) if output_root is not None else "",
        "route_pillars": {
            "single_view_mainline": True,
            "visible_direct_hidden_diffusion": True,
            "official_dinov2_backbone": str(backbone_cfg.get("type", "")).lower() in {"transformers_dinov2", "transformers_dinov2_hybrid"},
            "real_rgb_required": not bool(backbone_cfg.get("allow_pseudo_rgb", False)),
            "canonical_real_data_root": packet_dir == canonical_packet_root,
            "room_disjoint_split": bool(room_split_info.get("disjoint")) if room_split_info.get("exists") else None,
        },
        "room_split": room_split_info,
        "violations": violations,
        "ok": len(violations) == 0,
    }

    if args.save_summary:
        save_path = Path(args.save_summary)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
