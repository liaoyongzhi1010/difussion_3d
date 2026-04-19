"""ScanNet loader — public benchmark for real indoor scans.

Requires a Stanford ScanNet access token. Download is handled by
`docs/superpowers/specs/datasets/scannet.download.sh`, which fetches the
requested scenes into `data/external/scannet/` and runs `scannet.extract.py`
to convert each scene into our packet format.
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
class ScanNetPaths:
    raw_root: Path              # data/external/scannet/scans/
    rendered_root: Path         # our pre-rendered packets
    split_file: Path            # json: {"train": [...], "val": [...]}


class ScanNetPacketDataset(Dataset[dict[str, Any]]):
    """Reads pre-rendered ScanNet packets conforming to the pixarmesh schema."""

    def __init__(
        self,
        paths: ScanNetPaths,
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
                f"rendered ScanNet packets not found at {paths.rendered_root}. "
                "Run docs/superpowers/specs/datasets/scannet.download.sh + scannet.extract.py."
            )
        if not paths.split_file.exists():
            raise FileNotFoundError(f"split manifest missing at {paths.split_file}")
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


__all__ = ["ScanNetPaths", "ScanNetPacketDataset"]
