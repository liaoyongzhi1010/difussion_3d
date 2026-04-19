"""Render 3D-FRONT rooms into amodal-scene-diff packets.

For each scene JSON, this script:
    1. Parses the scene graph (rooms, furniture instances, CAD references).
    2. Picks a fixed set of camera poses per room.
    3. Rasterizes RGB + depth + per-object visibility masks via trimesh +
       pyrender (EGL offscreen).
    4. Emits one packet .pt per (room, view), matching
       PixarMeshPacketDataset's schema.

This file is intentionally a runnable entrypoint — not imported by the package.
It depends on `trimesh`, `pyrender`, `Pillow`. Install them into the env that
runs preprocessing (not the training env, which can remain lean).

Skeleton; fleshed out as rendering integration proceeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-out", type=Path, required=True)
    parser.add_argument("--samples-per-room", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-rooms", type=int, default=-1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.layout_root.exists() or not args.future_root.exists():
        raise SystemExit(
            "3D-FRONT / 3D-FUTURE roots not found. Run 3dfront.download.sh first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Renderer integration is intentionally deferred — we ship the control flow
    # so users can stage their pipeline, then fill the TODOs. Concretely:
    #
    #   scene_jsons = sorted(args.layout_root.glob("*.json"))
    #   for scene_json in scene_jsons:
    #       scene = ThreeDFrontScene.load(scene_json, future_root=args.future_root)
    #       for room in scene.rooms:
    #           cameras = sample_cameras(room, n=args.samples_per_room)
    #           for cam in cameras:
    #               packet = render_packet(room, cam, image_size=args.image_size)
    #               torch.save(packet, args.output_dir / f"{packet['meta']['sample_id']}.pt")
    #
    # The renderer is postponed to the next session because it needs EGL setup
    # (headless) and is not in scope for the restructure commit.

    sample_ids: list[str] = sorted(p.stem for p in args.output_dir.glob("*.pt"))
    n = len(sample_ids)
    if n == 0:
        raise SystemExit(
            "no packets found in output-dir; wire the TODO render loop before running."
        )
    n_train = int(args.train_ratio * n)
    n_val = int(args.val_ratio * n)
    split = {
        "train": sample_ids[:n_train],
        "val": sample_ids[n_train : n_train + n_val],
        "test": sample_ids[n_train + n_val :],
    }
    args.split_out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"wrote split with train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
