"""Minimal training loop for the v5 line.

Does not replace the legacy `scripts/train/train_single_view_scene.py`, which
continues to drive v3/v4 experiments. This loop is the entry point referenced
by the design spec acceptance criterion:

    python -m amodal_scene_diff.engine.train_loop \
        --config configs/experiments/main/v5_dit_detr_dinov2_large.yaml \
        --train-steps 2

so the integration path is exercised end-to-end on every refactor.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from amodal_scene_diff.datasets import PixarMeshPacketDataset, collate_pixarmesh_packets
from amodal_scene_diff.diffusion import SingleViewSceneDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=-1,
                        help="Override cfg.training.max_steps (use <=0 to respect config)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    # resolve `include:` chain (top-level list of config paths merged in order)
    includes = cfg.pop("include", [])
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = (config_path.parent / inc).resolve() if not Path(inc).is_absolute() else Path(inc)
        merged = _merge(merged, load_config(inc_path))
    return _merge(merged, cfg)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, float]] = field(default_factory=list)


def build_dataloader(cfg: dict[str, Any]) -> DataLoader:
    data_cfg = cfg["data"]
    packet_dir = Path(data_cfg["packet_cache_root"])
    packet_paths = sorted(packet_dir.glob("*.pt"))
    if not packet_paths:
        raise FileNotFoundError(f"no packets found under {packet_dir}")
    dataset = PixarMeshPacketDataset(
        packet_paths=packet_paths,
        image_size=int(data_cfg.get("image_size", 512)),
    )
    runtime = cfg.get("runtime", {})
    return DataLoader(
        dataset,
        batch_size=int(runtime.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(runtime.get("num_workers", 0)),
        collate_fn=collate_pixarmesh_packets,
        drop_last=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_all(args.seed)

    runtime = cfg.get("runtime", {})
    device = str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = args.output_dir or Path(runtime.get("output_dir", "outputs/v5_smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loader = build_dataloader(cfg)
    model = SingleViewSceneDiffusion.from_config(cfg).to(device)

    training_cfg = cfg.get("training", {})
    max_steps = args.train_steps if args.train_steps > 0 else int(training_cfg.get("max_steps", 2000))
    lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = args.log_every

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    state = TrainState()

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        state.step = int(ckpt.get("step", 0))

    model.train()
    loader_iter = iter(loader)
    t0 = time.time()
    while state.step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = batch.to(device)

        losses = model.compute_losses(batch)
        loss = losses["loss_total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        state.step += 1

        if state.step % log_every == 0 or state.step == 1:
            entry = {k: float(v.detach().cpu().item()) for k, v in losses.items() if torch.is_tensor(v)}
            entry["step"] = state.step
            entry["elapsed_s"] = time.time() - t0
            state.history.append(entry)
            print(json.dumps(entry), flush=True)

    ckpt_path = output_dir / "latest.pt"
    torch.save({"model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "step": state.step, "cfg": cfg}, ckpt_path)
    (output_dir / "history.json").write_text(json.dumps(state.history, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", "steps": state.step, "checkpoint": str(ckpt_path)}))


if __name__ == "__main__":
    main()
