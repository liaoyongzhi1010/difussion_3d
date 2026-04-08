from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _load_train_utils() -> Any:
    train_script = Path(__file__).resolve().parent / "train_scene_diffusion.py"
    spec = importlib.util.spec_from_file_location("scene_train_utils", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load training utilities from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_UTILS = _load_train_utils()


class PosteriorSelector(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        hidden_dim = max(8, int(hidden_dim))
        depth = max(1, int(depth))
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(width, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a posterior sample selector on top of a frozen scene generator.")
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--config", default="configs/diffusion/base.yaml")
    parser.add_argument("--data-config", default="configs/data/3dfront_v1.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--sample-id-json", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--num-posterior-samples", type=int, default=8)
    parser.add_argument("--eval-num-posterior-samples", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=2.0)
    parser.add_argument("--selector-hidden-dim", type=int, default=128)
    parser.add_argument("--selector-depth", type=int, default=2)
    parser.add_argument("--selector-dropout", type=float, default=0.1)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-summary", default="outputs/debug/posterior_selector_summary.json")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--save-every-steps", type=int, default=200)
    return parser.parse_args()


def _masked_mse_per_scene(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numer = (((pred - target) ** 2) * mask).flatten(1).sum(dim=-1)
    denom = mask.flatten(1).sum(dim=-1).clamp_min(1.0)
    return numer / denom


def _binary_confidence_per_scene(probs: torch.Tensor) -> torch.Tensor:
    return (probs - 0.5).abs().mul(2.0).mean(dim=-1)


def _class_confidence_per_scene(class_probs: torch.Tensor, exist_probs: torch.Tensor) -> torch.Tensor:
    max_probs = class_probs.amax(dim=-1)
    numer = (max_probs * exist_probs).sum(dim=-1)
    denom = exist_probs.sum(dim=-1).clamp_min(1.0)
    return numer / denom


def _relation_confidence_per_scene(
    support_logits: torch.Tensor,
    floor_logits: torch.Tensor,
    wall_logits: torch.Tensor,
    object_presence: torch.Tensor,
) -> torch.Tensor:
    support_conf = (torch.sigmoid(support_logits) - 0.5).abs().mul(2.0)
    pair_weight = object_presence.unsqueeze(-1) * object_presence.unsqueeze(-2)
    diag_mask = ~torch.eye(pair_weight.shape[-1], dtype=torch.bool, device=pair_weight.device).unsqueeze(0)
    pair_weight = pair_weight * diag_mask.float()
    support_numer = (support_conf * pair_weight).flatten(1).sum(dim=-1)
    support_denom = pair_weight.flatten(1).sum(dim=-1).clamp_min(1.0)
    support_score = support_numer / support_denom

    floor_conf = (torch.sigmoid(floor_logits) - 0.5).abs().mul(2.0)
    wall_conf = (torch.sigmoid(wall_logits) - 0.5).abs().mul(2.0)
    unary_score = ((floor_conf + wall_conf) * 0.5 * object_presence).sum(dim=-1) / object_presence.sum(dim=-1).clamp_min(1.0)
    return 0.5 * (support_score + unary_score)


def _masked_state_mse_per_scene(samples: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numer = (((samples - reference) ** 2) * mask).flatten(2).sum(dim=-1)
    denom = mask.flatten(2).sum(dim=-1).clamp_min(1.0)
    return numer / denom


def _load_requested_sample_ids(sample_id_json: str | Path, split: str) -> list[str]:
    if not sample_id_json:
        return []
    payload = json.loads(Path(sample_id_json).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        selected = payload.get(split)
        if not isinstance(selected, list):
            raise KeyError(f"split {split!r} not found in {sample_id_json}")
        return [str(item) for item in selected]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise TypeError(f"unsupported sample-id json format: {type(payload)!r}")


def _filter_packet_paths(packet_paths: list[Path], sample_ids: list[str], max_samples: int | None) -> list[Path]:
    if not sample_ids:
        return packet_paths[:max_samples] if max_samples is not None else packet_paths
    packet_map = {path.stem: path for path in packet_paths}
    selected = [packet_map[sample_id] for sample_id in sample_ids if sample_id in packet_map]
    return selected[:max_samples] if max_samples is not None else selected


def _cycle_loader(loader: Iterable[Any]) -> Iterable[Any]:
    while True:
        for batch in loader:
            yield batch


def _make_loader(
    packet_dir: str,
    sample_id_json: str,
    split: str,
    batch_size: int,
    max_samples: int,
    cfg: dict[str, Any],
    *,
    shuffle: bool,
) -> tuple[Any, Any, list[Path]]:
    all_packet_paths = TRAIN_UTILS.discover_packet_paths(packet_dir, None)
    requested_sample_ids = _load_requested_sample_ids(sample_id_json, split) if sample_id_json else []
    selected_paths = _filter_packet_paths(all_packet_paths, requested_sample_ids, max_samples if max_samples > 0 else None)
    strip_spatial_conditioning = bool(not cfg["model"].get("uses_spatial_conditioning", False))
    dataset, loader = TRAIN_UTILS.make_dataloader_from_packets(
        packet_paths=selected_paths,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=shuffle,
        drop_last=False,
        preload_packets=True,
        strip_spatial_conditioning=strip_spatial_conditioning,
        spatial_placeholder_size=int(cfg["runtime"].get("spatial_placeholder_size", 1)),
        persistent_workers=False,
        prefetch_factor=2,
    )
    return dataset, loader, all_packet_paths


def _gather_selected(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(values.shape[0], device=values.device)
    return values[batch_indices, indices]


@torch.no_grad()
def _build_candidate_batch(
    generator: nn.Module,
    batch: Any,
    *,
    num_posterior_samples: int,
    num_inference_steps: int,
) -> dict[str, torch.Tensor]:
    gt_states = generator.continuous_state_targets(batch)
    visible_mask = batch.target.visible_loss_mask.float().unsqueeze(-1)
    visible_presence = batch.cond.visible_valid_mask.float()
    hidden_mask = batch.target.hidden_gt_mask.float().unsqueeze(-1)

    layout_errors: list[torch.Tensor] = []
    visible_errors: list[torch.Tensor] = []
    hidden_errors: list[torch.Tensor] = []
    hidden_exist_probs: list[torch.Tensor] = []
    exist_confidence_scores: list[torch.Tensor] = []
    semantic_confidence_scores: list[torch.Tensor] = []
    relation_confidence_scores: list[torch.Tensor] = []
    hidden_samples: list[torch.Tensor] = []

    for _ in range(num_posterior_samples):
        sample = generator.sample_posterior(batch, num_sampling_steps=num_inference_steps)
        layout_errors.append(((sample["layout"] - gt_states["layout"]) ** 2).mean(dim=-1))
        visible_errors.append(_masked_mse_per_scene(sample["visible"], gt_states["visible"], visible_mask))
        hidden_errors.append(_masked_mse_per_scene(sample["hidden"], gt_states["hidden"], hidden_mask))

        exist_probs = sample["hidden_exist_probs"]
        hidden_exist_probs.append(exist_probs)
        exist_confidence_scores.append(_binary_confidence_per_scene(exist_probs))
        semantic_confidence_scores.append(_class_confidence_per_scene(sample["hidden_cls_probs"], exist_probs))
        object_presence = torch.cat([visible_presence, exist_probs], dim=-1)
        relation_confidence_scores.append(
            _relation_confidence_per_scene(
                sample["support_logits"],
                sample["floor_logits"],
                sample["wall_logits"],
                object_presence,
            )
        )
        hidden_samples.append(sample["hidden"])

    layout_stack = torch.stack(layout_errors, dim=0)
    visible_stack = torch.stack(visible_errors, dim=0)
    hidden_stack = torch.stack(hidden_errors, dim=0)
    hidden_exist_stack = torch.stack(hidden_exist_probs, dim=0)
    hidden_sample_stack = torch.stack(hidden_samples, dim=0)
    exist_conf_stack = torch.stack(exist_confidence_scores, dim=0)
    semantic_conf_stack = torch.stack(semantic_confidence_scores, dim=0)
    relation_conf_stack = torch.stack(relation_confidence_scores, dim=0)

    consensus_weight = hidden_exist_stack.mean(dim=0, keepdim=True).unsqueeze(-1)
    consensus_hidden_mse = _masked_state_mse_per_scene(
        hidden_sample_stack,
        hidden_sample_stack.mean(dim=0, keepdim=True),
        consensus_weight,
    )

    features = torch.stack(
        [
            exist_conf_stack,
            semantic_conf_stack,
            relation_conf_stack,
            -consensus_hidden_mse,
            hidden_exist_stack.mean(dim=-1),
        ],
        dim=-1,
    ).permute(1, 0, 2).contiguous()
    feature_mean = features.mean(dim=1, keepdim=True)
    feature_std = features.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    features = (features - feature_mean) / feature_std

    oracle_cost = (hidden_stack + 0.05 * visible_stack + 0.05 * layout_stack).permute(1, 0).contiguous()
    return {
        "features": features,
        "oracle_cost": oracle_cost,
        "layout_mse": layout_stack.permute(1, 0).contiguous(),
        "visible_mse": visible_stack.permute(1, 0).contiguous(),
        "hidden_mse": hidden_stack.permute(1, 0).contiguous(),
    }


def _selector_metrics_from_logits(candidates: dict[str, torch.Tensor], logits: torch.Tensor) -> dict[str, float]:
    selected_indices = logits.argmax(dim=-1)
    oracle_indices = candidates["oracle_cost"].argmin(dim=-1)

    selected_hidden = _gather_selected(candidates["hidden_mse"], selected_indices)
    selected_visible = _gather_selected(candidates["visible_mse"], selected_indices)
    selected_layout = _gather_selected(candidates["layout_mse"], selected_indices)
    mean_hidden = candidates["hidden_mse"].mean(dim=-1)
    best_hidden = candidates["hidden_mse"].min(dim=-1).values
    gap = (mean_hidden - best_hidden).clamp_min(1.0e-6)
    gap_closed = ((mean_hidden - selected_hidden) / gap).clamp(0.0, 1.0)

    return {
        "oracle_top1_acc": float((selected_indices == oracle_indices).float().mean().item()),
        "selected_hidden_mse": float(selected_hidden.mean().item()),
        "selected_visible_mse": float(selected_visible.mean().item()),
        "selected_layout_mse": float(selected_layout.mean().item()),
        "mean_hidden_mse": float(mean_hidden.mean().item()),
        "best_hidden_mse": float(best_hidden.mean().item()),
        "oracle_hidden_gap_closed": float(gap_closed.mean().item()),
    }


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    payload = {
        "step": int(step),
        "cfg": cfg,
        "metrics": metrics,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(model, optimizer, step, cfg, metrics)
    step_path = checkpoint_dir / f"step_{step:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    shutil.copyfile(step_path, latest_path)
    return step_path


def _save_best_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(model, optimizer, step, cfg, metrics)
    best_path = checkpoint_dir / "best.pt"
    torch.save(payload, best_path)
    return best_path


def _load_checkpoint(checkpoint_path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    rng_state = checkpoint.get("rng_state", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if device.startswith("cuda") and "cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    return checkpoint


@torch.no_grad()
def _evaluate_selector(
    selector: nn.Module,
    generator: nn.Module,
    loader: Iterable[Any],
    device: str,
    *,
    num_posterior_samples: int,
    num_inference_steps: int,
    max_batches: int,
) -> dict[str, float]:
    selector.eval()
    aggregate = {
        "oracle_top1_acc": 0.0,
        "selected_hidden_mse": 0.0,
        "selected_visible_mse": 0.0,
        "selected_layout_mse": 0.0,
        "mean_hidden_mse": 0.0,
        "best_hidden_mse": 0.0,
        "oracle_hidden_gap_closed": 0.0,
        "loss": 0.0,
    }
    total_batches = 0
    total_scenes = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = batch.to(device)
        candidates = _build_candidate_batch(
            generator,
            batch,
            num_posterior_samples=num_posterior_samples,
            num_inference_steps=num_inference_steps,
        )
        logits = selector(candidates["features"])
        target_probs = torch.softmax(-candidates["oracle_cost"] / 2.0, dim=-1)
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        metrics = _selector_metrics_from_logits(candidates, logits)
        batch_size = int(batch.batch_size)
        for key, value in metrics.items():
            aggregate[key] += value * batch_size
        aggregate["loss"] += float(loss.item()) * batch_size
        total_batches += 1
        total_scenes += batch_size
    if total_scenes == 0:
        return {}
    aggregate = {key: value / total_scenes for key, value in aggregate.items()}
    aggregate["num_batches"] = float(total_batches)
    aggregate["num_scenes"] = float(total_scenes)
    return aggregate


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    diffusion_cfg = TRAIN_UTILS.load_yaml(args.config)
    data_cfg = TRAIN_UTILS.load_yaml(args.data_config)
    runtime_cfg = TRAIN_UTILS.load_yaml(args.runtime_config)
    cfg = TRAIN_UTILS.deep_merge(TRAIN_UTILS.deep_merge(diffusion_cfg, {"data": data_cfg}), {"runtime": runtime_cfg})

    generator_checkpoint = torch.load(args.generator_checkpoint, map_location="cpu")
    checkpoint_cfg = generator_checkpoint.get("cfg")
    if isinstance(checkpoint_cfg, dict):
        cfg = TRAIN_UTILS.deep_merge(checkpoint_cfg, {"runtime": cfg.get("runtime", {})})

    device = str(cfg["runtime"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    train_dataset, train_loader, all_packet_paths = _make_loader(
        args.packet_dir,
        args.sample_id_json,
        args.train_split,
        args.batch_size,
        args.max_train_samples,
        cfg,
        shuffle=True,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("selector training found zero train packets")

    val_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    val_loader = None
    if args.val_split:
        _, val_loader, _ = _make_loader(
            args.packet_dir,
            args.sample_id_json,
            args.val_split,
            args.batch_size,
            args.max_eval_samples,
            cfg,
            shuffle=False,
        )
    test_loader = None
    if args.test_split:
        _, test_loader, _ = _make_loader(
            args.packet_dir,
            args.sample_id_json,
            args.test_split,
            args.batch_size,
            args.max_eval_samples,
            cfg,
            shuffle=False,
        )

    generator = TRAIN_UTILS.build_model(cfg).to(device)
    generator.load_state_dict(generator_checkpoint["model_state"])
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    selector = PosteriorSelector(
        input_dim=5,
        hidden_dim=args.selector_hidden_dim,
        depth=args.selector_depth,
        dropout=args.selector_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume_from:
        resume_payload = _load_checkpoint(args.resume_from, selector, optimizer, device)
        start_step = int(resume_payload.get("step", 0))

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    train_stream = _cycle_loader(train_loader)
    log_every = max(1, min(50, args.eval_every_steps))
    save_every = max(1, args.save_every_steps)
    eval_every = max(1, args.eval_every_steps)
    start_time = time.perf_counter()
    last_metrics: dict[str, float] = {}
    last_checkpoint_path = ""
    best_checkpoint_path = ""
    best_val_metrics: dict[str, float] = {}
    best_step = 0
    best_val_hidden_mse = float("inf")

    if args.resume_from:
        resume_metrics = resume_payload.get("metrics", {})
        if isinstance(resume_metrics, dict) and "val_selected_hidden_mse" in resume_metrics:
            best_val_hidden_mse = float(resume_metrics["val_selected_hidden_mse"])
            best_step = start_step
            best_checkpoint_path = str(args.resume_from)

    for step_index in range(start_step, start_step + args.train_steps):
        global_step = step_index + 1
        batch = next(train_stream)
        batch = batch.to(device)

        with torch.no_grad():
            candidates = _build_candidate_batch(
                generator,
                batch,
                num_posterior_samples=args.num_posterior_samples,
                num_inference_steps=args.num_inference_steps,
            )

        selector.train()
        logits = selector(candidates["features"])
        target_probs = torch.softmax(-candidates["oracle_cost"] / float(args.target_temperature), dim=-1)
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(selector.parameters(), float(args.clip_grad_norm))
        optimizer.step()

        last_metrics = _selector_metrics_from_logits(candidates, logits)
        last_metrics["loss"] = float(loss.item())
        last_metrics["step"] = float(global_step)

        if (global_step % log_every == 0) or (global_step == start_step + args.train_steps):
            payload = {
                "step": global_step,
                "max_cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if device.startswith("cuda") else 0.0,
                "metrics": last_metrics,
            }
            print(json.dumps(payload), flush=True)

        if val_loader is not None and ((global_step % eval_every == 0) or (global_step == start_step + args.train_steps)):
            val_metrics = _evaluate_selector(
                selector,
                generator,
                val_loader,
                device,
                num_posterior_samples=args.eval_num_posterior_samples,
                num_inference_steps=args.num_inference_steps,
                max_batches=args.max_eval_batches,
            )
            print(json.dumps({"event": "val_eval", "step": global_step, "metrics": val_metrics}), flush=True)
            current_val_hidden = val_metrics.get("selected_hidden_mse") if val_metrics else None
            if current_val_hidden is not None and float(current_val_hidden) < best_val_hidden_mse:
                best_val_hidden_mse = float(current_val_hidden)
                best_val_metrics = dict(val_metrics)
                best_step = global_step
                if checkpoint_dir is not None:
                    payload_metrics = dict(last_metrics)
                    payload_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
                    best_path = _save_best_checkpoint(checkpoint_dir, selector, optimizer, global_step, cfg, payload_metrics)
                    best_checkpoint_path = str(best_path)
                    print(
                        json.dumps(
                            {
                                "event": "best_checkpoint_saved",
                                "step": global_step,
                                "path": best_checkpoint_path,
                                "selected_hidden_mse": best_val_hidden_mse,
                            }
                        ),
                        flush=True,
                    )

        if checkpoint_dir is not None and ((global_step % save_every == 0) or (global_step == start_step + args.train_steps)):
            payload_metrics = dict(last_metrics)
            if val_metrics:
                payload_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            saved_path = _save_checkpoint(checkpoint_dir, selector, optimizer, global_step, cfg, payload_metrics)
            last_checkpoint_path = str(saved_path)
            print(json.dumps({"event": "checkpoint_saved", "step": global_step, "path": last_checkpoint_path}), flush=True)

    if test_loader is not None:
        if best_checkpoint_path:
            _load_checkpoint(best_checkpoint_path, selector, optimizer, device)
        test_metrics = _evaluate_selector(
            selector,
            generator,
            test_loader,
            device,
            num_posterior_samples=args.eval_num_posterior_samples,
            num_inference_steps=args.num_inference_steps,
            max_batches=args.max_eval_batches,
        )
        print(json.dumps({"event": "test_eval", "step": start_step + args.train_steps, "metrics": test_metrics}), flush=True)

    runtime_sec = time.perf_counter() - start_time
    summary = {
        "generator_checkpoint": str(Path(args.generator_checkpoint)),
        "generator_step": int(generator_checkpoint.get("step", 0)),
        "train_steps": int(args.train_steps),
        "start_step": int(start_step),
        "end_step": int(start_step + args.train_steps),
        "selector_num_parameters": int(selector.num_parameters),
        "packet_dir": str(args.packet_dir),
        "num_available_packet_files": int(len(all_packet_paths)),
        "num_train_packet_files": int(len(train_dataset)),
        "num_posterior_samples": int(args.num_posterior_samples),
        "eval_num_posterior_samples": int(args.eval_num_posterior_samples),
        "num_inference_steps": int(args.num_inference_steps),
        "runtime_sec": float(runtime_sec),
        "max_cuda_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2) if device.startswith("cuda") else 0.0,
        "train_metrics": last_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_step": int(best_step),
        "best_val_metrics": best_val_metrics,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "resume_from": str(args.resume_from),
    }

    save_path = Path(args.save_summary)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
