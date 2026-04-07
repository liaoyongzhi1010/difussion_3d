from __future__ import annotations

import argparse
import contextlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from amodal_scene_diff.datasets import collate_scene_packets
from amodal_scene_diff.models.diffusion import (
    SceneConditionedTransformer,
    SceneDenoisingTransformer,
    SceneDiffusionStub,
)
from amodal_scene_diff.structures import D_MODEL, D_POSE, Z_DIM


class ScenePacketDataset(Dataset[dict[str, Any]]):
    """Dataset wrapper for materialized ScenePacketV1 `.pt` files.

    For the current real-data bootstrap stage, the full packet directory is small
    enough to preload into RAM. That removes repeated per-step `torch.load` calls
    which were starving the GPU.
    """

    def __init__(
        self,
        packet_paths: list[Path],
        *,
        preload_packets: bool = False,
        strip_spatial_conditioning: bool = False,
        spatial_placeholder_size: int = 1,
    ) -> None:
        self.packet_paths = packet_paths
        self.preload_packets = bool(preload_packets)
        self.strip_spatial_conditioning = bool(strip_spatial_conditioning)
        self.spatial_placeholder_size = max(1, int(spatial_placeholder_size))
        self.total_bytes = sum(path.stat().st_size for path in self.packet_paths)
        self._cache: list[dict[str, Any]] | None = None
        if self.preload_packets:
            self._cache = [self._load_packet(path) for path in self.packet_paths]

    def __len__(self) -> int:
        return len(self.packet_paths)

    def _load_packet(self, path: Path) -> dict[str, Any]:
        packet = torch.load(path, map_location="cpu")
        if not isinstance(packet, dict):
            raise TypeError(f"packet at {path} must be a dict, got {type(packet)!r}")
        return self._normalize_packet(packet)

    def _normalize_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        if not self.strip_spatial_conditioning:
            return packet

        normalized = dict(packet)
        condition = dict(packet.get("condition", {}))
        size = self.spatial_placeholder_size

        depth_obs = condition.get("depth_obs")
        if depth_obs is not None:
            depth_tensor = torch.as_tensor(depth_obs, dtype=torch.float32)
            condition["depth_obs"] = torch.zeros((1, size, size), dtype=depth_tensor.dtype)

        visible_union_mask = condition.get("visible_union_mask")
        if visible_union_mask is not None:
            condition["visible_union_mask"] = torch.zeros((1, size, size), dtype=torch.bool)

        normalized["condition"] = condition
        return normalized

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_packet(self.packet_paths[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or dry-run scene diffusion v1.")
    parser.add_argument("--config", default="configs/diffusion/base.yaml")
    parser.add_argument("--data-config", default="configs/data/3dfront_v1.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/dev.yaml")
    parser.add_argument("--packet-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run-dummy-batch", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--save-summary", default="outputs/debug/scene_diffusion_dry_run.json")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--save-every-steps", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"yaml config at {path} must load as a dict")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_dummy_packet(sample_index: int, image_size: int) -> dict[str, Any]:
    vis_count = 2 + (sample_index % 3)
    hid_count = sample_index % 2
    source_id = sample_index % 3

    return {
        "meta": {
            "sample_id": f"dummy_{sample_index:04d}",
            "scene_id": f"scene_{sample_index // 2:04d}",
            "room_id": f"room_{sample_index // 2:04d}",
            "camera_id": f"cam_{sample_index:04d}",
            "source_id": source_id,
            "camera_intrinsics": torch.eye(3),
            "camera_extrinsics": torch.eye(4),
        },
        "condition": {
            "f_global": torch.zeros(D_MODEL),
            "layout_token_cond": torch.zeros(1, D_MODEL),
            "visible_tokens_cond": torch.randn(vis_count, D_MODEL),
            "uncertainty_token": torch.zeros(1, D_MODEL),
            "pose0_calib": torch.randn(vis_count, D_POSE),
            "layout0_calib": torch.zeros(D_POSE),
            "lock_gate": torch.ones(vis_count, 1),
            "slot_confidence": torch.full((vis_count, 1), 0.75),
            "visible_valid_mask": torch.ones(vis_count, dtype=torch.bool),
            "depth_obs": torch.zeros(1, image_size, image_size),
            "visible_union_mask": torch.zeros(1, image_size, image_size, dtype=torch.bool),
        },
        "target": {
            "layout_gt": torch.zeros(D_POSE),
            "visible_cls_gt": torch.arange(vis_count, dtype=torch.long) % 10,
            "visible_amodal_pose_gt": torch.randn(vis_count, D_POSE),
            "visible_amodal_res_gt": torch.zeros(vis_count, D_POSE),
            "visible_z_gt": torch.randn(vis_count, Z_DIM),
            "visible_obj_uid": [f"v{sample_index}_{i}" for i in range(vis_count)],
            "visible_loss_mask": torch.ones(vis_count, dtype=torch.bool),
            "hidden_cls_gt": torch.arange(hid_count, dtype=torch.long) % 10,
            "hidden_pose_gt": torch.randn(hid_count, D_POSE),
            "hidden_z_gt": torch.randn(hid_count, Z_DIM),
            "hidden_obj_uid": [f"h{sample_index}_{i}" for i in range(hid_count)],
            "hidden_gt_mask": torch.ones(hid_count, dtype=torch.bool),
        },
    }


def build_dummy_packets(batch_size: int, image_size: int) -> list[dict[str, Any]]:
    return [build_dummy_packet(index, image_size=image_size) for index in range(batch_size)]


def discover_packet_paths(packet_dir: str | Path, max_samples: int | None) -> list[Path]:
    if not packet_dir:
        return []
    root = Path(packet_dir)
    if not root.exists():
        raise FileNotFoundError(f"packet directory does not exist: {root}")
    packet_paths = sorted(root.glob("*.pt"))
    if max_samples is not None:
        packet_paths = packet_paths[:max_samples]
    return packet_paths


def summarize_batch(batch: Any) -> dict[str, Any]:
    return {
        "batch_size": batch.batch_size,
        "source_ids": batch.cond.source_id.tolist(),
        "visible_tokens_cond_shape": list(batch.cond.visible_tokens_cond.shape),
        "hidden_z_gt_shape": list(batch.target.hidden_z_gt.shape),
        "relation_valid_mask_shape": list(batch.target.relation_valid_mask.shape),
        "depth_obs_shape": list(batch.cond.depth_obs.shape),
        "first_sample_id": batch.meta.sample_ids[0],
        "first_visible_uid_tail": batch.meta.visible_obj_uid[0][-2:],
        "first_hidden_uid_tail": batch.meta.hidden_obj_uid[0][-2:],
    }


def make_dataloader_from_packets(
    packet_paths: list[Path],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    drop_last: bool,
    preload_packets: bool,
    strip_spatial_conditioning: bool,
    spatial_placeholder_size: int,
    persistent_workers: bool,
    prefetch_factor: int,
) -> tuple[ScenePacketDataset, DataLoader]:
    dataset = ScenePacketDataset(
        packet_paths,
        preload_packets=preload_packets,
        strip_spatial_conditioning=strip_spatial_conditioning,
        spatial_placeholder_size=spatial_placeholder_size,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_scene_packets,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return dataset, DataLoader(dataset, **loader_kwargs)


def _cycle_loader(loader: DataLoader) -> Iterable[Any]:
    while True:
        for batch in loader:
            yield batch


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    noise_cfg = cfg.get("noise", {})
    loss_cfg = cfg.get("loss", {})
    arch = str(model_cfg.get("arch", "stub"))
    if arch == "stub":
        return SceneDiffusionStub()
    if arch == "scene_transformer_v0":
        return SceneConditionedTransformer(
            d_model=int(model_cfg.get("d_model", D_MODEL)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_ratio=float(model_cfg.get("ffn_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if arch == "scene_denoiser_v1":
        return SceneDenoisingTransformer(
            d_model=int(model_cfg.get("d_model", D_MODEL)),
            num_layers=int(model_cfg.get("num_layers", 12)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_ratio=float(model_cfg.get("ffn_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            train_timesteps=int(noise_cfg.get("train_timesteps", 1000)),
            beta_schedule=str(noise_cfg.get("beta_schedule", "cosine")),
            prediction_type=str(noise_cfg.get("prediction_type", "v_prediction")),
            layout_weight=float(loss_cfg.get("lambda_layout", 1.0)),
            visible_weight=float(loss_cfg.get("lambda_vis_align", loss_cfg.get("lambda_diff_geo", 1.0))),
            hidden_weight=float(loss_cfg.get("lambda_diff_geo", 1.0)),
            hidden_exist_weight=float(loss_cfg.get("lambda_exist", 1.0)),
            hidden_cls_weight=float(loss_cfg.get("lambda_cls", 1.0)),
            support_weight=float(loss_cfg.get("lambda_support", 0.5)),
            floor_weight=float(loss_cfg.get("lambda_floor", 0.25)),
            wall_weight=float(loss_cfg.get("lambda_wall", 0.25)),
            visible_mode=str(model_cfg.get("visible_mode", "diffusion")),
            attention_mode=str(model_cfg.get("attention_mode", "full")),
        )
    raise ValueError(f"Unsupported model arch: {arch}")


def _resolve_amp_dtype(device: str, mixed_precision: str | None) -> torch.dtype | None:
    if not device.startswith("cuda"):
        return None
    if not mixed_precision:
        return None
    mode = str(mixed_precision).lower()
    if mode in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mode in {"fp16", "float16", "half"}:
        return torch.float16
    return None


def _autocast_context(device: str, amp_dtype: torch.dtype | None) -> contextlib.AbstractContextManager[Any]:
    if amp_dtype is None:
        return contextlib.nullcontext()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    cfg: dict[str, Any],
    dataset_info: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": int(step),
        "cfg": cfg,
        "dataset_info": dataset_info,
        "metrics": metrics,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else {},
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        payload["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    cfg: dict[str, Any],
    dataset_info: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(model, optimizer, scaler, step, cfg, dataset_info, metrics)
    step_path = checkpoint_dir / f"step_{step:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    shutil.copyfile(step_path, latest_path)
    return step_path


def _load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _optimizer_to(optimizer, device)
    scaler_state = checkpoint.get("scaler_state") or {}
    if scaler.is_enabled() and scaler_state:
        scaler.load_state_dict(scaler_state)

    rng_state = checkpoint.get("rng_state", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if device.startswith("cuda") and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return checkpoint


def run_train_steps(
    batch_stream: Iterable[Any],
    train_steps: int,
    device: str,
    lr: float,
    cfg: dict[str, Any],
    dataset_info: dict[str, Any],
    checkpoint_dir: str | Path = "",
    resume_from: str | Path = "",
    save_every_steps: int | None = None,
) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg).to(device)
    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    runtime_cfg = cfg.get("runtime", {})

    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    grad_accum = max(1, int(training_cfg.get("grad_accum", 1)))
    clip_grad_norm = float(training_cfg.get("clip_grad_norm", 0.0))
    mixed_precision = training_cfg.get("mixed_precision")
    amp_dtype = _resolve_amp_dtype(device, mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and device.startswith("cuda")))
    non_blocking = bool(runtime_cfg.get("non_blocking_device_transfer", bool(runtime_cfg.get("pin_memory", False))))
    log_every_steps = max(1, int(logging_cfg.get("log_every_steps", 100)))
    save_every = max(1, int(save_every_steps if save_every_steps is not None else logging_cfg.get("save_every_steps", 1000)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)

    start_step = 0
    resume_from_path = Path(resume_from) if resume_from else None
    if resume_from_path is not None:
        checkpoint = _load_checkpoint(resume_from_path, model, optimizer, scaler, device)
        start_step = int(checkpoint.get("step", 0))

    last_batch = None
    last_metrics: dict[str, float] = {}
    data_wait_total = 0.0
    compute_total = 0.0
    window_data_wait = 0.0
    window_compute = 0.0
    start_time = time.perf_counter()
    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None
    last_checkpoint_path = ""

    end_step = start_step + train_steps
    for step_index in range(start_step, end_step):
        global_step = step_index + 1
        data_start = time.perf_counter()
        batch = next(batch_stream)
        data_wait = time.perf_counter() - data_start
        batch = batch.to(device, non_blocking=non_blocking)
        last_batch = batch

        compute_start = time.perf_counter()
        with _autocast_context(device, amp_dtype):
            losses = model.compute_losses(batch)
            loss_total = losses["loss_total"] / grad_accum

        if scaler.is_enabled():
            scaler.scale(loss_total).backward()
        else:
            loss_total.backward()

        should_step = (global_step % grad_accum == 0) or (global_step == end_step)
        if should_step:
            if clip_grad_norm > 0.0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - compute_start
        data_wait_total += data_wait
        compute_total += compute_time
        window_data_wait += data_wait
        window_compute += compute_time

        last_metrics = {key: float(value.detach().float().cpu().item()) for key, value in losses.items()}
        last_metrics["step"] = float(global_step)

        if (global_step % log_every_steps == 0) or (global_step == end_step):
            window_steps = min(log_every_steps, max(1, global_step - max(start_step, global_step - log_every_steps)))
            log_payload = {
                "step": global_step,
                "steps_per_sec": window_steps / max(window_compute + window_data_wait, 1.0e-8),
                "avg_data_wait_sec": window_data_wait / max(window_steps, 1),
                "avg_compute_sec": window_compute / max(window_steps, 1),
                "max_cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if device.startswith("cuda") else 0.0,
                "metrics": last_metrics,
            }
            print(json.dumps(log_payload), flush=True)
            window_data_wait = 0.0
            window_compute = 0.0

        if checkpoint_dir_path is not None and ((global_step % save_every == 0) or (global_step == end_step)):
            saved_path = _save_checkpoint(
                checkpoint_dir=checkpoint_dir_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                cfg=cfg,
                dataset_info=dataset_info,
                metrics=last_metrics,
            )
            last_checkpoint_path = str(saved_path)
            print(json.dumps({"event": "checkpoint_saved", "step": global_step, "path": last_checkpoint_path}), flush=True)

    if last_batch is None:
        raise RuntimeError("train_steps must be greater than 0")

    total_runtime = time.perf_counter() - start_time
    summary = summarize_batch(last_batch)
    summary["train_steps"] = train_steps
    summary["start_step"] = start_step
    summary["end_step"] = end_step
    summary["num_parameters"] = model.num_parameters
    summary["model_arch"] = str(cfg["model"].get("arch", "stub"))
    summary["losses"] = last_metrics
    summary["runtime_sec"] = total_runtime
    summary["avg_data_wait_sec"] = data_wait_total / max(train_steps, 1)
    summary["avg_compute_sec"] = compute_total / max(train_steps, 1)
    summary["steps_per_sec"] = train_steps / max(total_runtime, 1.0e-8)
    summary["max_cuda_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if device.startswith("cuda") else 0.0
    summary["dataset_info"] = dataset_info
    summary["resume_from"] = str(resume_from_path) if resume_from_path is not None else ""
    summary["last_checkpoint_path"] = last_checkpoint_path
    return summary


def main() -> None:
    args = parse_args()
    diffusion_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    runtime_cfg = load_yaml(args.runtime_config)
    cfg = deep_merge(deep_merge(diffusion_cfg, {"data": data_cfg}), {"runtime": runtime_cfg})

    seed = args.seed if args.seed is not None else int(cfg["runtime"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = args.batch_size if args.batch_size is not None else int(cfg["runtime"].get("batch_size", 2))
    max_samples = args.max_samples if args.max_samples is not None else int(cfg["runtime"].get("max_samples", 8))
    image_size = int(cfg["data"].get("image_size", 512))
    device = str(cfg["runtime"].get("device", "cpu"))
    lr = float(cfg["training"].get("lr", 2.0e-4))

    use_dummy = bool(args.dry_run_dummy_batch or (cfg["runtime"].get("dry_run_dummy_batch", False) and not args.packet_dir))
    packet_paths: list[Path] = []
    dataset_info = {"type": "dummy", "num_packet_files": 0}
    if not use_dummy:
        packet_paths = discover_packet_paths(
            args.packet_dir or cfg["data"].get("packet_cache_root", ""),
            max_samples=max_samples,
        )
        dataset_info = {
            "type": "packet_dir",
            "num_packet_files": len(packet_paths),
        }

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir and args.train_steps > 0 and not use_dummy:
        checkpoint_dir = str(Path(args.save_summary).parent / "checkpoints")

    if args.train_steps > 0:
        if use_dummy:
            def dummy_stream() -> Iterable[Any]:
                while True:
                    yield collate_scene_packets(build_dummy_packets(batch_size=batch_size, image_size=image_size))

            summary = run_train_steps(
                dummy_stream(),
                train_steps=args.train_steps,
                device=device,
                lr=lr,
                cfg=cfg,
                dataset_info=dataset_info,
                checkpoint_dir=checkpoint_dir,
                resume_from=args.resume_from,
                save_every_steps=args.save_every_steps,
            )
            summary["mode"] = "dummy_train"
        elif packet_paths:
            strip_spatial_conditioning = bool(not cfg["model"].get("uses_spatial_conditioning", False))
            preload_packets = bool(cfg["runtime"].get("preload_packets", False))
            runtime_num_workers = int(cfg["runtime"].get("num_workers", 0))
            num_workers = 0 if preload_packets else runtime_num_workers
            dataset, loader = make_dataloader_from_packets(
                packet_paths=packet_paths,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=bool(cfg["runtime"].get("pin_memory", False)) and num_workers > 0,
                shuffle=True,
                drop_last=False,
                preload_packets=preload_packets,
                strip_spatial_conditioning=strip_spatial_conditioning,
                spatial_placeholder_size=int(cfg["runtime"].get("spatial_placeholder_size", 1)),
                persistent_workers=bool(cfg["runtime"].get("persistent_workers", False)) and num_workers > 0,
                prefetch_factor=int(cfg["runtime"].get("prefetch_factor", 2)),
            )
            dataset_info.update(
                {
                    "packet_total_mb": round(dataset.total_bytes / 1024 / 1024, 3),
                    "preload_packets": dataset.preload_packets,
                    "strip_spatial_conditioning": dataset.strip_spatial_conditioning,
                    "spatial_placeholder_size": dataset.spatial_placeholder_size,
                    "effective_num_workers": num_workers,
                }
            )
            summary = run_train_steps(
                _cycle_loader(loader),
                train_steps=args.train_steps,
                device=device,
                lr=lr,
                cfg=cfg,
                dataset_info=dataset_info,
                checkpoint_dir=checkpoint_dir,
                resume_from=args.resume_from,
                save_every_steps=args.save_every_steps,
            )
            summary["mode"] = "packet_dir_train"
        else:
            raise RuntimeError("No training source available for train_steps > 0")
    else:
        if use_dummy:
            batch = collate_scene_packets(build_dummy_packets(batch_size=batch_size, image_size=image_size))
            summary = summarize_batch(batch)
            summary["mode"] = "dummy"
        elif packet_paths:
            strip_spatial_conditioning = bool(not cfg["model"].get("uses_spatial_conditioning", False))
            preload_packets = bool(cfg["runtime"].get("preload_packets", False))
            runtime_num_workers = int(cfg["runtime"].get("num_workers", 0))
            num_workers = 0 if preload_packets else runtime_num_workers
            dataset, loader = make_dataloader_from_packets(
                packet_paths=packet_paths,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=bool(cfg["runtime"].get("pin_memory", False)) and num_workers > 0,
                shuffle=False,
                drop_last=False,
                preload_packets=preload_packets,
                strip_spatial_conditioning=strip_spatial_conditioning,
                spatial_placeholder_size=int(cfg["runtime"].get("spatial_placeholder_size", 1)),
                persistent_workers=bool(cfg["runtime"].get("persistent_workers", False)) and num_workers > 0,
                prefetch_factor=int(cfg["runtime"].get("prefetch_factor", 2)),
            )
            summary = summarize_batch(next(iter(loader)))
            summary["mode"] = "packet_dir"
            summary["num_packet_files"] = len(packet_paths)
            summary["dataset_info"] = {
                "packet_total_mb": round(dataset.total_bytes / 1024 / 1024, 3),
                "preload_packets": dataset.preload_packets,
                "strip_spatial_conditioning": dataset.strip_spatial_conditioning,
                "spatial_placeholder_size": dataset.spatial_placeholder_size,
                "effective_num_workers": num_workers,
            }
        else:
            raise RuntimeError(
                "No training source available. Pass --dry-run-dummy-batch or provide --packet-dir with ScenePacket `.pt` files."
            )

    save_path = Path(args.save_summary)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
