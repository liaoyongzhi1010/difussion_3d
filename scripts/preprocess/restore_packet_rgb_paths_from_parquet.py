from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore real RGB image paths into single-view packets from the original parquet shards.")
    parser.add_argument("--parquet-dir", default="outputs/real_data/hf_cache/data")
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--views-dir", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-summary", default="outputs/debug/restore_packet_rgb_summary.json")
    return parser.parse_args()


def discover_packet_paths(packet_dir: Path, max_samples: int) -> list[Path]:
    packet_paths = sorted(packet_dir.glob("*.pt"))
    if max_samples > 0:
        packet_paths = packet_paths[:max_samples]
    return packet_paths


def iter_matching_rows(parquet_paths: list[Path], wanted_ids: set[str]) -> Iterable[tuple[str, bytes, str]]:
    remaining = set(wanted_ids)
    for parquet_path in parquet_paths:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=16, columns=["uid", "image"]):
            for row in batch.to_pylist():
                sample_id = str(row.get("uid"))
                if sample_id not in remaining:
                    continue
                image_payload = row.get("image") or {}
                image_bytes = image_payload.get("bytes") if isinstance(image_payload, dict) else None
                if not image_bytes:
                    continue
                yield sample_id, image_bytes, str(parquet_path)
                remaining.remove(sample_id)
                if not remaining:
                    return


def update_packet(packet_path: Path, image_path: Path) -> None:
    packet = torch.load(packet_path, map_location="cpu")
    packet = dict(packet)
    meta = dict(packet.get("meta", {}))
    meta["image_path"] = str(image_path)
    packet["meta"] = meta
    condition = dict(packet.get("condition", {}))
    condition.pop("rgb_obs", None)
    packet["condition"] = condition
    torch.save(packet, packet_path)


def update_view_json(view_path: Path, image_path: Path) -> None:
    payload = json.loads(view_path.read_text(encoding="utf-8"))
    payload["rgb_path"] = str(image_path)
    payload["image_path"] = str(image_path)
    view_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    packet_dir = Path(args.packet_dir)
    if not packet_dir.exists():
        raise FileNotFoundError(f"packet dir does not exist: {packet_dir}")
    views_dir = Path(args.views_dir) if args.views_dir else None
    if views_dir is not None and not views_dir.exists():
        raise FileNotFoundError(f"views dir does not exist: {views_dir}")

    image_root = Path(args.image_root) if args.image_root else packet_dir.parent / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    packet_paths = discover_packet_paths(packet_dir, max_samples=args.max_samples)
    wanted_ids = {path.stem for path in packet_paths}
    parquet_paths = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet shards found under {args.parquet_dir}")

    restored = 0
    restored_from: dict[str, str] = {}
    for sample_id, image_bytes, parquet_path in iter_matching_rows(parquet_paths, wanted_ids):
        image_path = image_root / f"{sample_id}.png"
        if not image_path.exists():
            image_path.write_bytes(image_bytes)
        update_packet(packet_dir / f"{sample_id}.pt", image_path)
        if views_dir is not None:
            view_path = views_dir / f"{sample_id}.json"
            if view_path.exists():
                update_view_json(view_path, image_path)
        restored += 1
        restored_from[sample_id] = parquet_path

    missing = sorted(wanted_ids.difference(restored_from.keys()))
    summary = {
        "packet_dir": str(packet_dir),
        "views_dir": str(views_dir) if views_dir is not None else "",
        "image_root": str(image_root),
        "num_packets_requested": len(packet_paths),
        "num_restored": restored,
        "num_missing": len(missing),
        "missing_sample_ids": missing[:100],
    }
    summary_path = Path(args.save_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
