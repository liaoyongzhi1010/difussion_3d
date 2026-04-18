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
from torch.utils.data import DataLoader

from amodal_scene_diff.datasets import SingleViewPacketDataset, collate_single_view_packets
from amodal_scene_diff.models.diffusion import SingleViewReconstructionDiffusion
from amodal_scene_diff.structures import (
    C_OBJ,
    D_POSE,
    K_HID,
    K_VIS,
    N_OBJ_MAX,
    Z_DIM,
    SceneMetaBatch,
    SceneTargetBatch,
    SingleViewConditionBatch,
    SingleViewSceneBatch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper mainline: visible direct reconstruction + hidden diffusion from a single view.")
    parser.add_argument("--config", default="configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml")
    parser.add_argument("--data-config", default="configs/data/pixarmesh_single_view_main.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--packet-dir", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run-dummy-batch", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--save-summary", default="outputs/debug/single_view_scene_train_summary.json")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--val-packet-dir", default="")
    parser.add_argument("--val-sample-id-json", default="")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=0)
    parser.add_argument("--val-every-steps", type=int, default=0)
    parser.add_argument("--val-num-inference-steps", type=int, default=20)
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


def load_model_state_compat(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    remapped = dict(state_dict)
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    alias_pairs = [
        ("observation_encoder.norm.", "observation_encoder.rgb_norm."),
        ("observation_encoder.proj.", "observation_encoder.rgb_proj."),
        ("observation_encoder.rgb_norm.", "observation_encoder.norm."),
        ("observation_encoder.rgb_proj.", "observation_encoder.proj."),
    ]
    for src_prefix, dst_prefix in alias_pairs:
        src_keys = [key for key in list(remapped.keys()) if key.startswith(src_prefix)]
        if not src_keys:
            continue
        if any(key.startswith(dst_prefix) for key in remapped):
            continue
        if not any(key.startswith(dst_prefix) for key in model_keys):
            continue
        for key in src_keys:
            remapped[dst_prefix + key[len(src_prefix):]] = remapped[key]
    filtered = {key: value for key, value in remapped.items() if key in model_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys after compatibility remap: {unexpected}")
    if missing:
        raise RuntimeError(f"missing checkpoint keys after compatibility remap: {missing}")


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
    load_model_state_compat(model, checkpoint["model_state"])
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


def _cycle_loader(loader: DataLoader) -> Iterable[Any]:
    while True:
        for batch in loader:
            yield batch


def _load_requested_sample_ids(sample_id_json: str | Path, split: str) -> list[str]:
    if not sample_id_json:
        return []
    payload = json.loads(Path(sample_id_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if not split:
            raise TypeError("sample-id json is a dict; --split is required")
        selected = payload.get(split)
        if not isinstance(selected, list):
            raise KeyError(f"split {split!r} not found in {sample_id_json}")
        return [str(item) for item in selected]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise TypeError(f"unsupported sample-id json format: {type(payload)!r}")


def _filter_packet_paths(packet_paths: list[Path], sample_ids: list[str], max_samples: int | None) -> list[Path]:
    if sample_ids:
        packet_map = {path.stem: path for path in packet_paths}
        packet_paths = [packet_map[sample_id] for sample_id in sample_ids if sample_id in packet_map]
    if max_samples is not None and max_samples > 0:
        packet_paths = packet_paths[:max_samples]
    return packet_paths


def _masked_mse_per_scene(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numer = (((pred - target) ** 2) * mask).flatten(1).sum(dim=-1)
    denom = mask.flatten(1).sum(dim=-1).clamp_min(1.0)
    return numer / denom


def _class_accuracy_per_scene(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    correct = ((pred == labels).float() * mask.float()).sum(dim=-1)
    denom = mask.float().sum(dim=-1).clamp_min(1.0)
    return correct / denom


@torch.no_grad()
def evaluate_single_view_state(
    model: SingleViewReconstructionDiffusion,
    loader: DataLoader,
    device: str,
    num_inference_steps: int,
) -> dict[str, float]:
    model.eval()
    aggregate = {
        "layout_mse": 0.0,
        "visible_mse": 0.0,
        "hidden_mse": 0.0,
        "visible_exist_acc": 0.0,
        "visible_cls_acc": 0.0,
        "hidden_exist_acc": 0.0,
        "hidden_cls_acc": 0.0,
    }
    total_scenes = 0
    for batch in loader:
        batch = batch.to(device)
        gt_states = model.continuous_state_targets(batch)
        visible_mask = batch.target.visible_loss_mask.float().unsqueeze(-1)
        visible_mask_2d = batch.target.visible_loss_mask.float()
        hidden_mask = batch.target.hidden_gt_mask.float().unsqueeze(-1)
        hidden_mask_2d = batch.target.hidden_gt_mask.float()

        sample = model.sample_posterior(batch, num_sampling_steps=num_inference_steps)
        layout_mse = ((sample["layout"] - gt_states["layout"]) ** 2).mean(dim=-1)
        visible_mse = _masked_mse_per_scene(sample["visible"], gt_states["visible"], visible_mask)
        hidden_mse = _masked_mse_per_scene(sample["hidden"], gt_states["hidden"], hidden_mask)
        visible_exist_acc = ((sample["visible_exist_probs"] > 0.5).float() == visible_mask_2d).float().mean(dim=-1)
        visible_cls_acc = _class_accuracy_per_scene(sample["visible_cls_logits"], batch.target.visible_cls_gt, batch.target.visible_loss_mask)
        hidden_exist_acc = ((sample["hidden_exist_probs"] > 0.5).float() == hidden_mask_2d).float().mean(dim=-1)
        hidden_cls_acc = _class_accuracy_per_scene(sample["hidden_cls_logits"], batch.target.hidden_cls_gt, batch.target.hidden_gt_mask)

        aggregate["layout_mse"] += float(layout_mse.sum().item())
        aggregate["visible_mse"] += float(visible_mse.sum().item())
        aggregate["hidden_mse"] += float(hidden_mse.sum().item())
        aggregate["visible_exist_acc"] += float(visible_exist_acc.sum().item())
        aggregate["visible_cls_acc"] += float(visible_cls_acc.sum().item())
        aggregate["hidden_exist_acc"] += float(hidden_exist_acc.sum().item())
        aggregate["hidden_cls_acc"] += float(hidden_cls_acc.sum().item())
        total_scenes += batch.batch_size

    model.train()
    return {key: value / max(total_scenes, 1) for key, value in aggregate.items()}


def build_model(cfg: dict[str, Any]) -> SingleViewReconstructionDiffusion:
    return SingleViewReconstructionDiffusion.from_config(cfg)


def discover_packet_paths(packet_dir: str | Path, max_samples: int | None) -> list[Path]:
    if not packet_dir:
        return []
    root = Path(packet_dir)
    if not root.exists():
        raise FileNotFoundError(f"packet directory does not exist: {root}")
    packet_paths = sorted(root.glob("*.pt"))
    if max_samples is not None and max_samples > 0:
        packet_paths = packet_paths[:max_samples]
    return packet_paths


def build_dummy_batch(batch_size: int, image_size: int, obs_channels: int) -> SingleViewSceneBatch:
    cond = SingleViewConditionBatch(
        obs_image=torch.randn(batch_size, obs_channels, image_size, image_size),
        depth_obs=torch.rand(batch_size, 1, image_size, image_size),
        visible_union_mask=torch.randint(0, 2, (batch_size, 1, image_size, image_size), dtype=torch.float32),
        rgb_available=torch.zeros(batch_size, dtype=torch.bool),
        source_id=torch.arange(batch_size, dtype=torch.long) % 3,
    )
    target = SceneTargetBatch(
        layout_gt=torch.randn(batch_size, D_POSE),
        visible_cls_gt=torch.randint(0, C_OBJ, (batch_size, K_VIS), dtype=torch.long),
        visible_amodal_pose_gt=torch.randn(batch_size, K_VIS, D_POSE),
        visible_amodal_res_gt=torch.zeros(batch_size, K_VIS, D_POSE),
        visible_z_gt=torch.randn(batch_size, K_VIS, Z_DIM),
        visible_loss_mask=torch.rand(batch_size, K_VIS) > 0.25,
        hidden_cls_gt=torch.randint(0, C_OBJ, (batch_size, K_HID), dtype=torch.long),
        hidden_pose_gt=torch.randn(batch_size, K_HID, D_POSE),
        hidden_z_gt=torch.randn(batch_size, K_HID, Z_DIM),
        hidden_gt_mask=torch.rand(batch_size, K_HID) > 0.35,
        support_gt=torch.rand(batch_size, N_OBJ_MAX, N_OBJ_MAX),
        floor_gt=torch.rand(batch_size, N_OBJ_MAX),
        wall_gt=torch.rand(batch_size, N_OBJ_MAX),
        relation_valid_mask=torch.rand(batch_size, N_OBJ_MAX) > 0.1,
    )
    meta = SceneMetaBatch(
        sample_ids=[f"dummy_{index:04d}" for index in range(batch_size)],
        scene_ids=[f"scene_{index:04d}" for index in range(batch_size)],
        room_ids=[f"room_{index:04d}" for index in range(batch_size)],
        camera_ids=[f"cam_{index:04d}" for index in range(batch_size)],
        camera_intrinsics=torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
        camera_extrinsics=torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),
        visible_obj_uid=[[f"v{index}_{slot}" for slot in range(K_VIS)] for index in range(batch_size)],
        hidden_obj_uid=[[f"h{index}_{slot}" for slot in range(K_HID)] for index in range(batch_size)],
    )
    batch = SingleViewSceneBatch(cond=cond, target=target, meta=meta)
    batch.validate()
    return batch


def make_dataloader(
    packet_paths: list[Path],
    *,
    batch_size: int,
    image_size: int,
    preload_packets: bool,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    drop_last: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> tuple[SingleViewPacketDataset, DataLoader]:
    dataset = SingleViewPacketDataset(packet_paths, preload_packets=preload_packets, image_size=image_size)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_single_view_packets,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return dataset, DataLoader(dataset, **loader_kwargs)


def summarize_batch(batch: SingleViewSceneBatch) -> dict[str, Any]:
    return {
        "batch_size": batch.batch_size,
        "obs_image_shape": list(batch.cond.obs_image.shape),
        "depth_obs_shape": list(batch.cond.depth_obs.shape),
        "rgb_available": [bool(v) for v in batch.cond.rgb_available.tolist()],
        "layout_gt_shape": list(batch.target.layout_gt.shape),
        "visible_z_gt_shape": list(batch.target.visible_z_gt.shape),
        "hidden_z_gt_shape": list(batch.target.hidden_z_gt.shape),
        "first_sample_id": batch.meta.sample_ids[0],
        "first_visible_uid_tail": batch.meta.visible_obj_uid[0][-2:],
        "first_hidden_uid_tail": batch.meta.hidden_obj_uid[0][-2:],
    }


def run_train_steps(
    batch_stream: Iterable[SingleViewSceneBatch],
    train_steps: int,
    device: str,
    lr: float,
    cfg: dict[str, Any],
    dataset_info: dict[str, Any],
    checkpoint_dir: str | Path = "",
    resume_from: str | Path = "",
    save_every_steps: int | None = None,
    val_loader: DataLoader | None = None,
    val_every_steps: int = 0,
    val_num_inference_steps: int = 20,
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
    last_val_metrics: dict[str, float] = {}
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

        if val_loader is not None and val_every_steps > 0 and ((global_step % val_every_steps == 0) or (global_step == end_step)):
            last_val_metrics = evaluate_single_view_state(
                model,
                val_loader,
                device=device,
                num_inference_steps=val_num_inference_steps,
            )
            print(json.dumps({"event": "val_metrics", "step": global_step, "metrics": last_val_metrics}), flush=True)

        if checkpoint_dir_path is not None and ((global_step % save_every == 0) or (global_step == end_step)):
            checkpoint_metrics = dict(last_metrics)
            checkpoint_metrics.update({f"val_{key}": value for key, value in last_val_metrics.items()})
            saved_path = _save_checkpoint(
                checkpoint_dir=checkpoint_dir_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                cfg=cfg,
                dataset_info=dataset_info,
                metrics=checkpoint_metrics,
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
    summary["model_arch"] = str(cfg["model"].get("arch", "single_view_scene_v1"))
    summary["losses"] = last_metrics
    summary["runtime_sec"] = total_runtime
    summary["avg_data_wait_sec"] = data_wait_total / max(train_steps, 1)
    summary["avg_compute_sec"] = compute_total / max(train_steps, 1)
    summary["steps_per_sec"] = train_steps / max(total_runtime, 1.0e-8)
    summary["max_cuda_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if device.startswith("cuda") else 0.0
    summary["dataset_info"] = dataset_info
    summary["val_metrics"] = last_val_metrics
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
    obs_channels = int(cfg["model"].get("obs_channels", 4))
    device = str(cfg["runtime"].get("device", "cpu"))
    lr = float(cfg["training"].get("lr", 2.0e-4))

    use_dummy = bool(args.dry_run_dummy_batch or (cfg["runtime"].get("dry_run_dummy_batch", False) and not args.packet_dir))
    packet_paths: list[Path] = []
    val_packet_paths: list[Path] = []
    dataset_info = {"type": "dummy", "num_packet_files": 0}
    if not use_dummy:
        packet_root = args.packet_dir or cfg["data"].get("packet_cache_root", "")
        packet_paths = discover_packet_paths(packet_root, max_samples=max_samples)
        dataset_info = {"type": "single_view_packet", "num_packet_files": len(packet_paths)}

        val_packet_root = args.val_packet_dir or packet_root
        val_sample_json = args.val_sample_id_json or str(cfg["data"].get("split_json", ""))
        val_sample_ids = _load_requested_sample_ids(val_sample_json, args.val_split) if val_sample_json else []
        val_packet_paths = _filter_packet_paths(
            discover_packet_paths(val_packet_root, None),
            val_sample_ids,
            args.val_max_samples if args.val_max_samples > 0 else None,
        )
        if val_packet_paths:
            dataset_info["num_val_packet_files"] = len(val_packet_paths)
            dataset_info["val_split"] = args.val_split

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir and args.train_steps > 0 and not use_dummy:
        checkpoint_dir = str(Path(args.save_summary).parent / "checkpoints_single_view_scene")

    if args.train_steps > 0:
        if use_dummy:
            def dummy_stream() -> Iterable[SingleViewSceneBatch]:
                while True:
                    yield build_dummy_batch(batch_size=batch_size, image_size=image_size, obs_channels=obs_channels)

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
                val_loader=None,
                val_every_steps=args.val_every_steps,
                val_num_inference_steps=args.val_num_inference_steps,
            )
            summary["mode"] = "dummy_train"
        elif packet_paths:
            preload_packets = bool(cfg["runtime"].get("preload_packets", False))
            runtime_num_workers = int(cfg["runtime"].get("num_workers", 0))
            num_workers = 0 if preload_packets else runtime_num_workers
            dataset, loader = make_dataloader(
                packet_paths,
                batch_size=batch_size,
                image_size=image_size,
                preload_packets=preload_packets,
                num_workers=num_workers,
                pin_memory=bool(cfg["runtime"].get("pin_memory", False)) and num_workers > 0,
                shuffle=True,
                drop_last=False,
                persistent_workers=bool(cfg["runtime"].get("persistent_workers", False)) and num_workers > 0,
                prefetch_factor=int(cfg["runtime"].get("prefetch_factor", 2)),
            )
            val_loader = None
            if val_packet_paths and args.val_every_steps > 0:
                _, val_loader = make_dataloader(
                    val_packet_paths,
                    batch_size=int(args.val_batch_size or batch_size),
                    image_size=image_size,
                    preload_packets=True,
                    num_workers=0,
                    pin_memory=False,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=False,
                    prefetch_factor=2,
                )
            dataset_info.update(
                {
                    "packet_total_mb": round(dataset.total_bytes / 1024 / 1024, 3),
                    "preload_packets": dataset.preload_packets,
                    "effective_num_workers": num_workers,
                    "image_size": image_size,
                    "obs_channels": obs_channels,
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
                val_loader=val_loader,
                val_every_steps=args.val_every_steps,
                val_num_inference_steps=args.val_num_inference_steps,
            )
            summary["mode"] = "single_view_packet_train"
        else:
            raise RuntimeError("train-steps requested but no packet data was found")
    else:
        batch = build_dummy_batch(batch_size=batch_size, image_size=image_size, obs_channels=obs_channels)
        summary = summarize_batch(batch)
        summary["mode"] = "dummy_summary"
        summary["dataset_info"] = dataset_info

    summary_path = Path(args.save_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
