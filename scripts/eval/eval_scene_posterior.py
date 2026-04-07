from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def _load_train_utils() -> Any:
    train_script = Path(__file__).resolve().parents[1] / "train" / "train_scene_diffusion.py"
    spec = importlib.util.spec_from_file_location("scene_train_utils", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load training utilities from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_UTILS = _load_train_utils()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior sampling/eval for scene_denoiser_v1 checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--config", default="configs/diffusion/base.yaml")
    parser.add_argument("--data-config", default="configs/data/3dfront_v1.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--num-posterior-samples", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--sample-id-json", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--save-summary", default="outputs/debug/posterior_eval_summary.json")
    return parser.parse_args()


def _masked_mse_per_scene(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numer = (((pred - target) ** 2) * mask).flatten(1).sum(dim=-1)
    denom = mask.flatten(1).sum(dim=-1).clamp_min(1.0)
    return numer / denom


def _pairwise_diversity_per_scene(samples: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    if len(samples) < 2:
        return torch.zeros((samples[0].shape[0],), device=samples[0].device)
    pair_metrics: list[torch.Tensor] = []
    for first in range(len(samples)):
        for second in range(first + 1, len(samples)):
            numer = (((samples[first] - samples[second]) ** 2) * mask).flatten(1).sum(dim=-1)
            denom = mask.flatten(1).sum(dim=-1).clamp_min(1.0)
            pair_metrics.append(torch.sqrt(numer / denom))
    return torch.stack(pair_metrics, dim=0).mean(dim=0)


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


def _zscore_across_samples(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (values - mean) / std


def _gather_selected_per_scene(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(values.shape[1], device=values.device)
    return values[indices, batch_indices]


def _selector_metric_totals() -> dict[str, float]:
    return {
        "layout_mse": 0.0,
        "visible_mse": 0.0,
        "hidden_mse": 0.0,
        "hidden_exist_acc": 0.0,
        "hidden_cls_acc": 0.0,
        "hidden_exist_brier": 0.0,
        "oracle_hidden_gap_closed": 0.0,
    }


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
    if not sample_ids:
        return packet_paths[:max_samples] if max_samples is not None else packet_paths
    packet_map = {path.stem: path for path in packet_paths}
    selected = [packet_map[sample_id] for sample_id in sample_ids if sample_id in packet_map]
    return selected[:max_samples] if max_samples is not None else selected


def main() -> None:
    args = parse_args()
    diffusion_cfg = TRAIN_UTILS.load_yaml(args.config)
    data_cfg = TRAIN_UTILS.load_yaml(args.data_config)
    runtime_cfg = TRAIN_UTILS.load_yaml(args.runtime_config)
    cfg = TRAIN_UTILS.deep_merge(TRAIN_UTILS.deep_merge(diffusion_cfg, {"data": data_cfg}), {"runtime": runtime_cfg})

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_cfg = checkpoint.get("cfg")
    if isinstance(checkpoint_cfg, dict):
        cfg = TRAIN_UTILS.deep_merge(checkpoint_cfg, {"runtime": cfg.get("runtime", {})})

    device = str(cfg["runtime"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    all_packet_paths = TRAIN_UTILS.discover_packet_paths(args.packet_dir, None)
    requested_sample_ids = _load_requested_sample_ids(args.sample_id_json, args.split)
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None
    packet_paths = _filter_packet_paths(all_packet_paths, requested_sample_ids, max_samples)
    if not packet_paths:
        raise RuntimeError("posterior eval found zero packet files after filtering")

    strip_spatial_conditioning = bool(not cfg["model"].get("uses_spatial_conditioning", False))
    preload_packets = True
    dataset, loader = TRAIN_UTILS.make_dataloader_from_packets(
        packet_paths=packet_paths,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        preload_packets=preload_packets,
        strip_spatial_conditioning=strip_spatial_conditioning,
        spatial_placeholder_size=int(cfg["runtime"].get("spatial_placeholder_size", 1)),
        persistent_workers=False,
        prefetch_factor=2,
    )

    model = TRAIN_UTILS.build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    aggregate = {
        "layout_mse": 0.0,
        "visible_mse": 0.0,
        "hidden_mse": 0.0,
        "best_layout_mse": 0.0,
        "best_visible_mse": 0.0,
        "best_hidden_mse": 0.0,
        "hidden_diversity": 0.0,
        "visible_diversity": 0.0,
        "hidden_exist_acc": 0.0,
        "hidden_cls_acc": 0.0,
        "hidden_exist_brier": 0.0,
    }
    selector_aggregate = {
        "exist_confidence": _selector_metric_totals(),
        "semantic_confidence": _selector_metric_totals(),
        "relation_confidence": _selector_metric_totals(),
        "consensus_hidden": _selector_metric_totals(),
        "joint_confidence": _selector_metric_totals(),
    }
    total_scenes = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            gt_states = model.continuous_state_targets(batch)
            visible_mask = batch.target.visible_loss_mask.float().unsqueeze(-1)
            visible_presence = batch.cond.visible_valid_mask.float()
            hidden_mask = batch.target.hidden_gt_mask.float().unsqueeze(-1)
            hidden_mask_2d = batch.target.hidden_gt_mask.float()

            layout_errors: list[torch.Tensor] = []
            visible_errors: list[torch.Tensor] = []
            hidden_errors: list[torch.Tensor] = []
            hidden_exist_accs: list[torch.Tensor] = []
            hidden_cls_accs: list[torch.Tensor] = []
            hidden_exist_briers: list[torch.Tensor] = []
            hidden_exist_probs: list[torch.Tensor] = []
            visible_samples: list[torch.Tensor] = []
            hidden_samples: list[torch.Tensor] = []
            exist_confidence_scores: list[torch.Tensor] = []
            semantic_confidence_scores: list[torch.Tensor] = []
            relation_confidence_scores: list[torch.Tensor] = []

            for _ in range(args.num_posterior_samples):
                sample = model.sample_posterior(batch, num_sampling_steps=args.num_inference_steps)
                layout_errors.append(((sample["layout"] - gt_states["layout"]) ** 2).mean(dim=-1))
                visible_errors.append(_masked_mse_per_scene(sample["visible"], gt_states["visible"], visible_mask))
                hidden_errors.append(_masked_mse_per_scene(sample["hidden"], gt_states["hidden"], hidden_mask))

                exist_probs = sample["hidden_exist_probs"]
                hidden_exist_probs.append(exist_probs)
                hidden_exist_accs.append(((exist_probs > 0.5).float() == hidden_mask_2d).float().mean(dim=-1))
                hidden_exist_briers.append(((exist_probs - hidden_mask_2d) ** 2).mean(dim=-1))

                pred_cls = sample["hidden_cls_logits"].argmax(dim=-1)
                cls_correct = ((pred_cls == batch.target.hidden_cls_gt).float() * hidden_mask_2d).sum(dim=-1)
                cls_denom = hidden_mask_2d.sum(dim=-1).clamp_min(1.0)
                hidden_cls_accs.append(cls_correct / cls_denom)

                visible_samples.append(sample["visible"])
                hidden_samples.append(sample["hidden"])
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

            layout_stack = torch.stack(layout_errors, dim=0)
            visible_stack = torch.stack(visible_errors, dim=0)
            hidden_stack = torch.stack(hidden_errors, dim=0)
            hidden_exist_stack = torch.stack(hidden_exist_probs, dim=0)
            hidden_exist_acc_stack = torch.stack(hidden_exist_accs, dim=0)
            hidden_cls_acc_stack = torch.stack(hidden_cls_accs, dim=0)
            hidden_exist_brier_stack = torch.stack(hidden_exist_briers, dim=0)
            exist_conf_stack = torch.stack(exist_confidence_scores, dim=0)
            semantic_conf_stack = torch.stack(semantic_confidence_scores, dim=0)
            relation_conf_stack = torch.stack(relation_confidence_scores, dim=0)
            hidden_sample_stack = torch.stack(hidden_samples, dim=0)

            best_layout = layout_stack.min(dim=0).values
            best_visible = visible_stack.min(dim=0).values
            best_hidden = hidden_stack.min(dim=0).values
            mean_layout = layout_stack.mean(dim=0)
            mean_visible = visible_stack.mean(dim=0)
            mean_hidden = hidden_stack.mean(dim=0)
            mean_exist_acc = hidden_exist_acc_stack.mean(dim=0)
            mean_cls_acc = hidden_cls_acc_stack.mean(dim=0)
            mean_exist_prob = hidden_exist_stack.mean(dim=0)
            exist_brier = ((mean_exist_prob - hidden_mask_2d) ** 2).mean(dim=-1)
            hidden_diversity = _pairwise_diversity_per_scene(hidden_samples, hidden_mask)
            visible_diversity = _pairwise_diversity_per_scene(visible_samples, visible_mask)
            consensus_weight = hidden_exist_stack.mean(dim=0, keepdim=True).unsqueeze(-1)
            consensus_hidden_mse = _masked_state_mse_per_scene(
                hidden_sample_stack,
                hidden_sample_stack.mean(dim=0, keepdim=True),
                consensus_weight,
            )

            selector_scores = {
                "exist_confidence": exist_conf_stack,
                "semantic_confidence": semantic_conf_stack,
                "relation_confidence": relation_conf_stack,
                "consensus_hidden": -consensus_hidden_mse,
            }
            selector_scores["joint_confidence"] = (
                0.5 * _zscore_across_samples(exist_conf_stack)
                + _zscore_across_samples(semantic_conf_stack)
                - _zscore_across_samples(consensus_hidden_mse)
            )

            aggregate["layout_mse"] += float(mean_layout.sum().item())
            aggregate["visible_mse"] += float(mean_visible.sum().item())
            aggregate["hidden_mse"] += float(mean_hidden.sum().item())
            aggregate["best_layout_mse"] += float(best_layout.sum().item())
            aggregate["best_visible_mse"] += float(best_visible.sum().item())
            aggregate["best_hidden_mse"] += float(best_hidden.sum().item())
            aggregate["hidden_diversity"] += float(hidden_diversity.sum().item())
            aggregate["visible_diversity"] += float(visible_diversity.sum().item())
            aggregate["hidden_exist_acc"] += float(mean_exist_acc.sum().item())
            aggregate["hidden_cls_acc"] += float(mean_cls_acc.sum().item())
            aggregate["hidden_exist_brier"] += float(exist_brier.sum().item())

            oracle_gap = (mean_hidden - best_hidden).clamp_min(1.0e-6)
            for selector_name, selector_score in selector_scores.items():
                selected_indices = selector_score.argmax(dim=0)
                selected_layout = _gather_selected_per_scene(layout_stack, selected_indices)
                selected_visible = _gather_selected_per_scene(visible_stack, selected_indices)
                selected_hidden = _gather_selected_per_scene(hidden_stack, selected_indices)
                selected_exist_acc = _gather_selected_per_scene(hidden_exist_acc_stack, selected_indices)
                selected_cls_acc = _gather_selected_per_scene(hidden_cls_acc_stack, selected_indices)
                selected_exist_brier = _gather_selected_per_scene(hidden_exist_brier_stack, selected_indices)
                gap_closed = ((mean_hidden - selected_hidden) / oracle_gap).clamp(0.0, 1.0)

                selector_aggregate[selector_name]["layout_mse"] += float(selected_layout.sum().item())
                selector_aggregate[selector_name]["visible_mse"] += float(selected_visible.sum().item())
                selector_aggregate[selector_name]["hidden_mse"] += float(selected_hidden.sum().item())
                selector_aggregate[selector_name]["hidden_exist_acc"] += float(selected_exist_acc.sum().item())
                selector_aggregate[selector_name]["hidden_cls_acc"] += float(selected_cls_acc.sum().item())
                selector_aggregate[selector_name]["hidden_exist_brier"] += float(selected_exist_brier.sum().item())
                selector_aggregate[selector_name]["oracle_hidden_gap_closed"] += float(gap_closed.sum().item())
            total_scenes += int(batch.batch_size)

    if total_scenes == 0:
        raise RuntimeError("posterior eval found zero scenes")

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "num_eval_scenes": total_scenes,
        "num_packet_files": len(packet_paths),
        "num_available_packet_files": len(all_packet_paths),
        "num_posterior_samples": int(args.num_posterior_samples),
        "num_inference_steps": int(args.num_inference_steps),
        "split": str(args.split),
        "sample_id_json": str(args.sample_id_json),
        "dataset_info": {
            "packet_total_mb": round(dataset.total_bytes / 1024 / 1024, 3),
            "preload_packets": dataset.preload_packets,
            "strip_spatial_conditioning": dataset.strip_spatial_conditioning,
        },
        "metrics": {key: value / total_scenes for key, value in aggregate.items()},
        "selector_metrics": {
            name: {key: value / total_scenes for key, value in totals.items()}
            for name, totals in selector_aggregate.items()
        },
    }

    save_path = Path(args.save_summary)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
