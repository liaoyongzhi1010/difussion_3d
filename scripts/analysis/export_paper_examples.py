
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from amodal_scene_diff.datasets import collate_scene_packets

ROOT = Path(__file__).resolve().parents[2]
PANEL_SIZE = 360
HEADER_H = 40
FIGURE_BG = (248, 248, 244)
VISIBLE_COLOR = (44, 98, 217, 110)
HIDDEN_GT_COLOR = (244, 140, 6, 115)
HIDDEN_PRED_COLOR = (220, 52, 87, 115)
ROOM_COLOR = (48, 48, 48, 255)
TEXT_COLOR = (24, 24, 24, 255)


def _load_train_utils() -> Any:
    train_script = ROOT / "scripts/train/train_scene_diffusion.py"
    spec = importlib.util.spec_from_file_location("scene_train_utils", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load training utilities from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_UTILS = _load_train_utils()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-facing qualitative examples for the visible-locked generator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--config", default="configs/diffusion/visible_locked_occbias_v0625.yaml")
    parser.add_argument("--data-config", default="configs/data/3dfront_v1.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--sample-id-json", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--output-dir", default="examples/figures/visible_locked_occbias_v0625_main")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-posterior-samples", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--selector", default="joint_confidence", choices=["joint_confidence", "exist_confidence", "semantic_confidence", "relation_confidence"])
    return parser.parse_args()


def _load_requested_sample_ids(sample_id_json: str | Path, split: str) -> list[str]:
    if not sample_id_json:
        return []
    payload = json.loads((ROOT / Path(sample_id_json)).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        selected = payload.get(split)
        if not isinstance(selected, list):
            raise KeyError(f"split {split!r} not found in {sample_id_json}")
        return [str(item) for item in selected]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise TypeError(f"unsupported sample-id json format: {type(payload)!r}")


def _filter_packet_paths(packet_paths: list[Path], sample_ids: list[str]) -> list[Path]:
    if not sample_ids:
        return packet_paths
    packet_map = {path.stem: path for path in packet_paths}
    return [packet_map[sample_id] for sample_id in sample_ids if sample_id in packet_map]


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


def _decode_layout_bounds(layout_pose: torch.Tensor) -> dict[str, float]:
    cx, cz, log_sx, log_sz, floor_y, log_sy, _, _ = [float(item) for item in layout_pose.tolist()]
    sx = math.exp(log_sx)
    sz = math.exp(log_sz)
    sy = math.exp(log_sy)
    return {
        "xmin": cx - 0.5 * sx,
        "xmax": cx + 0.5 * sx,
        "zmin": cz - 0.5 * sz,
        "zmax": cz + 0.5 * sz,
        "floor_y": floor_y,
        "ceil_y": floor_y + sy,
    }


def _pose_to_corners_xz(pose: torch.Tensor) -> list[tuple[float, float]]:
    cx, _, cz, log_sx, _, log_sz, sin_yaw, cos_yaw = [float(item) for item in pose.tolist()]
    sx = math.exp(log_sx)
    sz = math.exp(log_sz)
    yaw = math.atan2(sin_yaw, cos_yaw)
    half_x = 0.5 * sx
    half_z = 0.5 * sz
    corners = [(-half_x, -half_z), (half_x, -half_z), (half_x, half_z), (-half_x, half_z)]
    out: list[tuple[float, float]] = []
    sin_v = math.sin(yaw)
    cos_v = math.cos(yaw)
    for local_x, local_z in corners:
        world_x = cx + cos_v * local_x - sin_v * local_z
        world_z = cz + sin_v * local_x + cos_v * local_z
        out.append((world_x, world_z))
    return out


def _map_point(bounds: dict[str, float], point: tuple[float, float], size: int) -> tuple[float, float]:
    x, z = point
    pad = 16.0
    width = max(bounds["xmax"] - bounds["xmin"], 1.0e-4)
    depth = max(bounds["zmax"] - bounds["zmin"], 1.0e-4)
    usable = size - 2.0 * pad
    px = pad + (x - bounds["xmin"]) / width * usable
    py = size - (pad + (z - bounds["zmin"]) / depth * usable)
    return px, py


def _depth_to_panel(depth_obs: Any, title: str) -> Image.Image:
    depth = torch.as_tensor(depth_obs, dtype=torch.float32).squeeze().cpu()
    depth_min = float(depth.min().item())
    depth_max = float(depth.max().item())
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = torch.zeros_like(depth)
    depth_img = (depth.mul(255.0).clamp(0, 255).byte().numpy())
    image = Image.fromarray(depth_img, mode="L").convert("RGB")
    image = ImageOps.contain(image, (PANEL_SIZE, PANEL_SIZE))
    canvas = Image.new("RGBA", (PANEL_SIZE, PANEL_SIZE + HEADER_H), FIGURE_BG)
    canvas.paste(image, ((PANEL_SIZE - image.width) // 2, HEADER_H + (PANEL_SIZE - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=TEXT_COLOR, font=ImageFont.load_default())
    return canvas.convert("RGB")


def _render_topdown(
    title: str,
    layout_pose: torch.Tensor,
    visible_pose: torch.Tensor,
    visible_mask: torch.Tensor,
    hidden_pose: torch.Tensor,
    hidden_mask: torch.Tensor,
    subtitle: str = "",
) -> Image.Image:
    bounds = _decode_layout_bounds(layout_pose)
    canvas = Image.new("RGBA", (PANEL_SIZE, PANEL_SIZE + HEADER_H), FIGURE_BG)
    draw = ImageDraw.Draw(canvas, "RGBA")
    room_rect = [
        _map_point(bounds, (bounds["xmin"], bounds["zmax"]), PANEL_SIZE),
        _map_point(bounds, (bounds["xmax"], bounds["zmin"]), PANEL_SIZE),
    ]
    draw.rectangle(
        [room_rect[0][0], room_rect[0][1] + HEADER_H, room_rect[1][0], room_rect[1][1] + HEADER_H],
        outline=ROOM_COLOR,
        width=2,
    )

    for index in range(int(visible_mask.shape[0])):
        if not bool(visible_mask[index].item()):
            continue
        polygon = [_map_point(bounds, point, PANEL_SIZE) for point in _pose_to_corners_xz(visible_pose[index])]
        polygon = [(x, y + HEADER_H) for x, y in polygon]
        draw.polygon(polygon, fill=VISIBLE_COLOR, outline=(44, 98, 217, 255))

    for index in range(int(hidden_mask.shape[0])):
        if not bool(hidden_mask[index].item()):
            continue
        polygon = [_map_point(bounds, point, PANEL_SIZE) for point in _pose_to_corners_xz(hidden_pose[index])]
        polygon = [(x, y + HEADER_H) for x, y in polygon]
        color = HIDDEN_GT_COLOR if title.endswith("GT") else HIDDEN_PRED_COLOR
        outline = (194, 98, 0, 255) if title.endswith("GT") else (176, 23, 55, 255)
        draw.polygon(polygon, fill=color, outline=outline)

    font = ImageFont.load_default()
    draw.text((12, 12), title, fill=TEXT_COLOR, font=font)
    if subtitle:
        draw.text((12, 24), subtitle, fill=TEXT_COLOR, font=font)
    return canvas.convert("RGB")


def _compose_triptych(sample_id: str, panels: list[Image.Image], footer: str) -> Image.Image:
    gap = 12
    width = len(panels) * PANEL_SIZE + (len(panels) - 1) * gap
    height = PANEL_SIZE + HEADER_H + 48
    canvas = Image.new("RGB", (width, height), FIGURE_BG)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += PANEL_SIZE + gap
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, height - 34), f"sample_id={sample_id}", fill=TEXT_COLOR, font=font)
    draw.text((10, height - 20), footer, fill=TEXT_COLOR, font=font)
    return canvas


def _make_contact_sheet(images: list[Image.Image]) -> Image.Image:
    columns = 2 if len(images) > 1 else 1
    rows = math.ceil(len(images) / columns)
    gap = 16
    tile_w, tile_h = images[0].size
    width = columns * tile_w + (columns - 1) * gap
    height = rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), FIGURE_BG)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x = col * (tile_w + gap)
        y = row * (tile_h + gap)
        canvas.paste(image, (x, y))
    return canvas


def _sample_selected_prediction(
    model: torch.nn.Module,
    batch: Any,
    selector: str,
    num_posterior_samples: int,
    num_inference_steps: int,
) -> dict[str, Any]:
    gt_states = model.continuous_state_targets(batch)
    visible_mask = batch.target.visible_loss_mask.float().unsqueeze(-1)
    visible_presence = batch.cond.visible_valid_mask.float()
    hidden_mask = batch.target.hidden_gt_mask.float().unsqueeze(-1)
    hidden_mask_2d = batch.target.hidden_gt_mask.float()

    visible_samples: list[torch.Tensor] = []
    hidden_samples: list[torch.Tensor] = []
    hidden_exist_probs: list[torch.Tensor] = []
    hidden_errors: list[torch.Tensor] = []
    exist_confidence_scores: list[torch.Tensor] = []
    semantic_confidence_scores: list[torch.Tensor] = []
    relation_confidence_scores: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(num_posterior_samples):
            sample = model.sample_posterior(batch, num_sampling_steps=num_inference_steps)
            visible_samples.append(sample["visible"])
            hidden_samples.append(sample["hidden"])
            hidden_exist_probs.append(sample["hidden_exist_probs"])
            numer = (((sample["hidden"] - gt_states["hidden"]) ** 2) * hidden_mask).flatten(1).sum(dim=-1)
            denom = hidden_mask.flatten(1).sum(dim=-1).clamp_min(1.0)
            hidden_errors.append(numer / denom)
            exist_probs = sample["hidden_exist_probs"]
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

    hidden_stack = torch.stack(hidden_errors, dim=0)
    hidden_sample_stack = torch.stack(hidden_samples, dim=0)
    hidden_exist_stack = torch.stack(hidden_exist_probs, dim=0)
    exist_conf_stack = torch.stack(exist_confidence_scores, dim=0)
    semantic_conf_stack = torch.stack(semantic_confidence_scores, dim=0)
    relation_conf_stack = torch.stack(relation_confidence_scores, dim=0)
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
        "joint_confidence": 0.5 * _zscore_across_samples(exist_conf_stack)
        + _zscore_across_samples(semantic_conf_stack)
        - _zscore_across_samples(consensus_hidden_mse),
    }
    selected_index = int(selector_scores[selector][:, 0].argmax().item())
    selected_visible = visible_samples[selected_index][0].detach().cpu()
    selected_hidden = hidden_samples[selected_index][0].detach().cpu()
    selected_exist = hidden_exist_probs[selected_index][0].detach().cpu()
    selected_hidden_mse = float(hidden_stack[selected_index, 0].item())
    return {
        "selected_index": selected_index,
        "selected_visible": selected_visible,
        "selected_hidden": selected_hidden,
        "selected_hidden_exist_probs": selected_exist,
        "selected_hidden_mse": selected_hidden_mse,
    }


def _choose_packet_paths(args: argparse.Namespace) -> list[Path]:
    if args.sample_ids.strip():
        requested = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    else:
        requested = _load_requested_sample_ids(args.sample_id_json, args.split)
    packet_paths = TRAIN_UTILS.discover_packet_paths(args.packet_dir, None)
    filtered = _filter_packet_paths(packet_paths, requested)
    candidates: list[tuple[int, int, str, Path]] = []
    for path in filtered:
        packet = torch.load(path, map_location="cpu")
        sample_id = str(packet.get("meta", {}).get("sample_id", path.stem))
        target = dict(packet.get("target", {}))
        hidden_uids = target.get("hidden_obj_uid") or packet.get("meta", {}).get("hidden_obj_uid") or []
        visible_uids = target.get("visible_obj_uid") or packet.get("meta", {}).get("visible_obj_uid") or []
        hidden_count = len(hidden_uids)
        visible_count = len(visible_uids)
        candidates.append((hidden_count, visible_count, sample_id, path))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    non_empty = [item for item in candidates if item[0] > 0]
    chosen = non_empty if non_empty else candidates
    return [item[3] for item in chosen[: max(int(args.top_k), 1)]]


def main() -> None:
    args = parse_args()
    diffusion_cfg = TRAIN_UTILS.load_yaml(args.config)
    data_cfg = TRAIN_UTILS.load_yaml(args.data_config)
    runtime_cfg = TRAIN_UTILS.load_yaml(args.runtime_config)
    cfg = TRAIN_UTILS.deep_merge(TRAIN_UTILS.deep_merge(diffusion_cfg, {"data": data_cfg}), {"runtime": runtime_cfg})

    checkpoint = torch.load(ROOT / Path(args.checkpoint), map_location="cpu")
    checkpoint_cfg = checkpoint.get("cfg")
    if isinstance(checkpoint_cfg, dict):
        cfg = TRAIN_UTILS.deep_merge(checkpoint_cfg, {"runtime": cfg.get("runtime", {})})

    device = str(cfg["runtime"].get("device", "cpu"))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"

    model = TRAIN_UTILS.build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    packet_paths = _choose_packet_paths(args)
    strip_spatial = bool(not cfg["model"].get("uses_spatial_conditioning", False))
    output_dir = ROOT / Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    contact_images: list[Image.Image] = []

    for path in packet_paths:
        raw_packet = torch.load(path, map_location="cpu")
        dataset = TRAIN_UTILS.ScenePacketDataset(
            [path],
            preload_packets=True,
            strip_spatial_conditioning=strip_spatial,
            spatial_placeholder_size=int(cfg["runtime"].get("spatial_placeholder_size", 1)),
        )
        batch = collate_scene_packets([dataset[0]]).to(device)
        selected = _sample_selected_prediction(
            model=model,
            batch=batch,
            selector=args.selector,
            num_posterior_samples=int(args.num_posterior_samples),
            num_inference_steps=int(args.num_inference_steps),
        )

        sample_id = str(raw_packet.get("meta", {}).get("sample_id", path.stem))
        depth_panel = _depth_to_panel(raw_packet.get("condition", {}).get("depth_obs"), title="Observed depth")
        gt_panel = _render_topdown(
            title="Top-down GT",
            layout_pose=batch.target.layout_gt[0].detach().cpu(),
            visible_pose=batch.target.visible_amodal_pose_gt[0].detach().cpu(),
            visible_mask=batch.cond.visible_valid_mask[0].detach().cpu(),
            hidden_pose=batch.target.hidden_pose_gt[0].detach().cpu(),
            hidden_mask=batch.target.hidden_gt_mask[0].detach().cpu(),
            subtitle=f"visible={int(batch.cond.visible_valid_mask[0].sum().item())} hidden={int(batch.target.hidden_gt_mask[0].sum().item())}",
        )
        pred_visible_pose = batch.cond.pose0_calib[0].detach().cpu() + selected["selected_visible"][:, : TRAIN_UTILS.D_POSE]
        pred_hidden_pose = selected["selected_hidden"][:, : TRAIN_UTILS.D_POSE]
        pred_hidden_mask = (selected["selected_hidden_exist_probs"] > 0.5).detach().cpu()
        pred_panel = _render_topdown(
            title="Top-down prediction",
            layout_pose=batch.target.layout_gt[0].detach().cpu(),
            visible_pose=pred_visible_pose,
            visible_mask=batch.cond.visible_valid_mask[0].detach().cpu(),
            hidden_pose=pred_hidden_pose,
            hidden_mask=pred_hidden_mask,
            subtitle=f"selector={args.selector} pick={selected['selected_index']} hidden_mse={selected['selected_hidden_mse']:.3f}",
        )
        figure = _compose_triptych(
            sample_id=sample_id,
            panels=[depth_panel, gt_panel, pred_panel],
            footer=f"checkpoint={Path(args.checkpoint).name} p={args.num_posterior_samples} s={args.num_inference_steps}",
        )
        figure_path = output_dir / f"{sample_id}.png"
        figure.save(figure_path)
        contact_images.append(figure)
        manifest_records.append(
            {
                "sample_id": sample_id,
                "packet_path": str(path),
                "figure_path": str(figure_path.relative_to(ROOT)),
                "visible_count": int(batch.cond.visible_valid_mask[0].sum().item()),
                "hidden_count": int(batch.target.hidden_gt_mask[0].sum().item()),
                "selected_index": int(selected["selected_index"]),
                "selected_hidden_mse": round(float(selected["selected_hidden_mse"]), 4),
            }
        )

    if not contact_images:
        raise RuntimeError("no figures were exported")

    contact_sheet = _make_contact_sheet(contact_images)
    contact_sheet_path = output_dir / "contact_sheet.png"
    contact_sheet.save(contact_sheet_path)

    manifest = {
        "checkpoint": str(Path(args.checkpoint)),
        "selector": str(args.selector),
        "num_posterior_samples": int(args.num_posterior_samples),
        "num_inference_steps": int(args.num_inference_steps),
        "output_dir": str(Path(args.output_dir)),
        "contact_sheet": str(contact_sheet_path.relative_to(ROOT)),
        "records": manifest_records,
    }
    manifest_path = ROOT / Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
