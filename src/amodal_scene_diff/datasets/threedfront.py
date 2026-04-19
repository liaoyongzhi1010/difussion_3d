"""3D-FRONT loader — public benchmark for indoor scene reconstruction.

3D-FRONT is EULA-gated (Alibaba Tianchi). The user signs the agreement and
downloads the raw archives using the helper scripts under
`docs/superpowers/specs/datasets/`. This module:

    1. Loads extracted per-room .json layouts and per-object CAD models.
    2. Renders (or reads pre-rendered) single-view RGB/depth packets in the same
       schema consumed by SingleViewSceneDiffusion.

Until rendering is triggered, only manifest scanning is performed so the class
can be imported without the raw data present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .pixarmesh import build_observation_image


@dataclass(frozen=True)
class ThreeDFrontPaths:
    layout_root: Path            # 3D-FRONT/ (scene .json files)
    future_root: Path            # 3D-FUTURE-model/ (CAD models)
    rendered_root: Path          # our pre-rendered packets per scene-view
    split_file: Path             # yaml/json listing of sample_ids per split


class ThreeDFrontPacketDataset(Dataset[dict[str, Any]]):
    """Reads pre-rendered 3D-FRONT packets conforming to the pixarmesh schema.

    Raw loading / rendering is delegated to
    `docs/superpowers/specs/datasets/3dfront.extract.py`. That script writes
    one `.pt` file per sample under `rendered_root/{sample_id}.pt`, matching
    the layout `PixarMeshPacketDataset` consumes.
    """

    def __init__(
        self,
        paths: ThreeDFrontPaths,
        split: str,
        *,
        image_size: int = 512,
        preload: bool = False,
    ) -> None:
        self.paths = paths
        self.split = split
        self.image_size = int(image_size)

        if not paths.rendered_root.exists():
            raise FileNotFoundError(
                f"rendered 3D-FRONT packets not found at {paths.rendered_root}. "
                "Run docs/superpowers/specs/datasets/3dfront.extract.py first."
            )
        if not paths.split_file.exists():
            raise FileNotFoundError(
                f"split manifest missing at {paths.split_file}. "
                "Generate it with the 3dfront.extract.py --split-out flag."
            )
        manifest = json.loads(paths.split_file.read_text(encoding="utf-8"))
        if split not in manifest:
            raise KeyError(f"split {split!r} not in {list(manifest)} from {paths.split_file}")
        self.sample_ids: list[str] = list(manifest[split])
        self.packet_paths: list[Path] = [paths.rendered_root / f"{sid}.pt" for sid in self.sample_ids]
        missing = [p for p in self.packet_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} packets listed in split manifest are missing; "
                f"first missing: {missing[0]}"
            )
        self._cache: list[dict[str, Any]] | None = [
            self._load_packet(p) for p in self.packet_paths
        ] if preload else None

    def __len__(self) -> int:
        return len(self.packet_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_packet(self.packet_paths[index])

    def _load_packet(self, path: Path) -> dict[str, Any]:
        packet = torch.load(path, map_location="cpu")
        if not isinstance(packet, dict):
            raise TypeError(f"packet at {path} must be a dict, got {type(packet)!r}")
        condition = dict(packet.get("condition", {}))
        obs_image, rgb_available = build_observation_image(packet=packet, image_size=self.image_size)
        condition["obs_image"] = obs_image
        condition["rgb_available"] = torch.tensor(bool(rgb_available), dtype=torch.bool)
        packet = dict(packet)
        packet["condition"] = condition
        return packet


__all__ = ["ThreeDFrontPaths", "ThreeDFrontPacketDataset"]
