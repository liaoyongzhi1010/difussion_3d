from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite scene packets with learned geometry z_mu latents.")
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--save-summary", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def replace_latents(z_tensor: torch.Tensor, uids: list[str], latent_dir: Path, cache: dict[str, torch.Tensor], stats: dict[str, int]) -> torch.Tensor:
    if z_tensor.numel() == 0:
        return z_tensor
    updated = z_tensor.clone()
    for index, uid in enumerate(uids):
        if uid == "__pad__":
            continue
        if uid not in cache:
            latent_path = latent_dir / f"{uid}.pt"
            if latent_path.exists():
                payload = torch.load(latent_path, map_location="cpu")
                cache[uid] = payload["z_mu"].float().view(-1)
            else:
                stats["missing_latents"] += 1
                continue
        updated[index] = cache[uid]
        stats["replaced_latents"] += 1
    return updated


def copy_small_metadata(src_root: Path, dst_root: Path) -> None:
    for name in ["summary.json", "coverage_summary.json"]:
        src = src_root / name
        if src.exists():
            shutil.copy2(src, dst_root / name)
    if (src_root / "index").exists():
        shutil.copytree(src_root / "index", dst_root / "index", dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    latent_dir = Path(args.latent_dir)
    summary_path = Path(args.save_summary)

    src_packet_dir = src_root / "packets"
    dst_packet_dir = dst_root / "packets"
    dst_geometry_dir = dst_root / "geometry"
    dst_packet_dir.mkdir(parents=True, exist_ok=True)
    dst_geometry_dir.mkdir(parents=True, exist_ok=True)
    copy_small_metadata(src_root, dst_root)

    latent_paths = sorted(latent_dir.glob("*.pt"))
    for latent_path in latent_paths:
        target = dst_geometry_dir / latent_path.name
        if args.overwrite or not target.exists():
            shutil.copy2(latent_path, target)

    packet_paths = sorted(src_packet_dir.glob("*.pt"))
    latent_cache: dict[str, torch.Tensor] = {}
    stats = {
        "num_packets": len(packet_paths),
        "rewritten_packets": 0,
        "skipped_existing_packets": 0,
        "replaced_latents": 0,
        "missing_latents": 0,
    }

    for packet_index, src_packet in enumerate(packet_paths, start=1):
        dst_packet = dst_packet_dir / src_packet.name
        if dst_packet.exists() and not args.overwrite:
            stats["skipped_existing_packets"] += 1
            continue
        payload = torch.load(src_packet, map_location="cpu")
        target = payload.get("target", {})
        target["visible_z_gt"] = replace_latents(target["visible_z_gt"], list(target["visible_obj_uid"]), latent_dir, latent_cache, stats)
        target["hidden_z_gt"] = replace_latents(target["hidden_z_gt"], list(target["hidden_obj_uid"]), latent_dir, latent_cache, stats)
        payload["target"] = target
        torch.save(payload, dst_packet)
        stats["rewritten_packets"] += 1

        if packet_index % 100 == 0 or packet_index == len(packet_paths):
            summary = {
                "src_root": str(src_root),
                "dst_root": str(dst_root),
                "latent_dir": str(latent_dir),
                **stats,
                "completed": packet_index == len(packet_paths),
            }
            save_json(summary_path, summary)
            print(json.dumps(summary), flush=True)

    packets_summary_src = src_root / "packets" / "summary.json"
    if packets_summary_src.exists():
        shutil.copy2(packets_summary_src, dst_packet_dir / "summary.json")

    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "latent_dir": str(latent_dir),
        **stats,
        "completed": True,
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
