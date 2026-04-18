from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from amodal_scene_diff.structures import SingleViewConditionBatch, SingleViewSceneBatch

from .collate import collate_scene_packets

Tensor = torch.Tensor


class PixarMeshPacketDataset(Dataset[dict[str, Any]]):
    """Load single-view observations directly from packets.

    Preferred source order:
    1. packet['observation']['rgb'] if already cached
    2. meta.image_path if it points to a real file
    3. pseudo-RGB synthesized from depth + visible mask
    """

    def __init__(self, packet_paths: list[Path], *, preload_packets: bool = False, image_size: int = 512) -> None:
        self.packet_paths = packet_paths
        self.preload_packets = bool(preload_packets)
        self.image_size = int(image_size)
        self.total_bytes = sum(path.stat().st_size for path in self.packet_paths)
        self._cache: list[dict[str, Any]] | None = None
        if self.preload_packets:
            self._cache = [self._load_packet(path) for path in self.packet_paths]

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
        meta = dict(packet.get("meta", {}))
        obs_image, rgb_available = build_observation_image(packet=packet, image_size=self.image_size)
        condition["obs_image"] = obs_image
        condition["rgb_available"] = torch.tensor(bool(rgb_available), dtype=torch.bool)
        packet = dict(packet)
        packet["condition"] = condition
        packet["meta"] = meta
        return packet


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _single_channel(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    tensor = _as_tensor(value, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected single-channel image [1,H,W], got {tuple(tensor.shape)}")
    return tensor


def _normalize_depth(depth_obs: Tensor) -> Tensor:
    depth = depth_obs.float().clone()
    valid = depth > 0
    if bool(valid.any()):
        values = depth[valid]
        dmin = values.min()
        dmax = values.max()
        if float((dmax - dmin).abs().item()) > 1.0e-6:
            depth[valid] = (values - dmin) / (dmax - dmin)
        else:
            depth[valid] = 0.0
    depth[~valid] = 0.0
    return depth.clamp(0.0, 1.0)


def _depth_gradient(depth: Tensor) -> Tensor:
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[..., :, 1:] = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    grad_y[..., 1:, :] = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    return (grad_x + grad_y).clamp(0.0, 1.0)


def _load_rgb_from_path(image_path: str | Path, image_size: int) -> Tensor | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_observation_image(packet: dict[str, Any], image_size: int) -> tuple[Tensor, bool]:
    condition = dict(packet.get("condition", {}))
    meta = dict(packet.get("meta", {}))

    depth_obs = _single_channel(condition.get("depth_obs", torch.zeros(1, image_size, image_size)))
    visible_union_mask = _single_channel(
        condition.get("visible_union_mask", torch.zeros(1, image_size, image_size)),
        dtype=torch.float32,
    )
    if int(depth_obs.shape[-1]) != image_size or int(depth_obs.shape[-2]) != image_size:
        depth_obs = F.interpolate(depth_obs.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    if int(visible_union_mask.shape[-1]) != image_size or int(visible_union_mask.shape[-2]) != image_size:
        visible_union_mask = F.interpolate(visible_union_mask.unsqueeze(0), size=(image_size, image_size), mode="nearest").squeeze(0)

    rgb_obs = condition.get("rgb_obs")
    rgb_tensor: Tensor | None = None
    if rgb_obs is not None:
        rgb_tensor = _as_tensor(rgb_obs, dtype=torch.float32)
        if rgb_tensor.ndim != 3 or rgb_tensor.shape[0] != 3:
            raise ValueError(f"rgb_obs must have shape [3,H,W], got {tuple(rgb_tensor.shape)}")
        if rgb_tensor.shape[-2:] != (image_size, image_size):
            rgb_tensor = F.interpolate(rgb_tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        rgb_tensor = rgb_tensor.clamp(0.0, 1.0)
    else:
        rgb_tensor = _load_rgb_from_path(meta.get("image_path", ""), image_size=image_size)

    depth_norm = _normalize_depth(depth_obs)
    mask = visible_union_mask.float().clamp(0.0, 1.0)
    gradient = _depth_gradient(depth_norm)

    if rgb_tensor is not None:
        obs_image = torch.cat([rgb_tensor, depth_norm], dim=0)
        return obs_image, True

    obs_image = torch.cat([depth_norm, mask, depth_norm * mask, gradient], dim=0)
    return obs_image, False


def collate_pixarmesh_packets(samples: list[dict[str, Any]]) -> SingleViewSceneBatch:
    base = collate_scene_packets(samples)
    obs_image = torch.stack([
        _as_tensor(sample["condition"]["obs_image"], dtype=torch.float32) for sample in samples
    ])
    rgb_available = torch.tensor(
        [bool(sample["condition"].get("rgb_available", False)) for sample in samples],
        dtype=torch.bool,
    )
    cond = SingleViewConditionBatch(
        obs_image=obs_image,
        depth_obs=base.cond.depth_obs,
        visible_union_mask=base.cond.visible_union_mask.float(),
        rgb_available=rgb_available,
        source_id=base.cond.source_id,
    )
    batch = SingleViewSceneBatch(cond=cond, target=base.target, meta=base.meta)
    batch.validate()
    return batch
