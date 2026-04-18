from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.metrics import structural_similarity

try:
    import lpips
except ImportError:  # pragma: no cover - optional dependency
    lpips = None


ROOT = Path(__file__).resolve().parents[2]
PANEL_SIZE = 320
HEADER_H = 42
FIGURE_BG = (248, 248, 244)
TEXT_COLOR = (24, 24, 24)
ROOM_COLOR = (56, 56, 56)
VISIBLE_COLOR = (44, 98, 217)
HIDDEN_COLOR = (244, 140, 6)
ROOM_LINE_WIDTH = 2


def _load_train_utils() -> Any:
    train_script = Path(__file__).resolve().parents[1] / "train" / "train_single_view_scene.py"
    spec = importlib.util.spec_from_file_location("single_view_train_utils", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load training utilities from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_UTILS = _load_train_utils()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render top-down semantic views for the single-view paper pipeline and compute PSNR/SSIM/LPIPS."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--config", default="configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml")
    parser.add_argument("--data-config", default="configs/data/pixarmesh_single_view_main.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-posterior-samples", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--sample-id-json", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--disable-lpips", action="store_true")
    parser.add_argument("--output-dir", default="outputs/debug/single_view_render_eval")
    parser.add_argument("--save-summary", default="")
    return parser.parse_args()


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
    if max_samples is not None:
        packet_paths = packet_paths[:max_samples]
    return packet_paths


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


def _scene_world_bounds(
    layout_pose: torch.Tensor,
    visible_pose: torch.Tensor,
    visible_mask: torch.Tensor,
    hidden_pose: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> dict[str, float]:
    bounds = _decode_layout_bounds(layout_pose)
    xs = [bounds["xmin"], bounds["xmax"]]
    zs = [bounds["zmin"], bounds["zmax"]]
    for pose_bank, pose_mask in ((visible_pose, visible_mask), (hidden_pose, hidden_mask)):
        for index in range(int(pose_mask.shape[0])):
            if not bool(pose_mask[index].item()):
                continue
            for x, z in _pose_to_corners_xz(pose_bank[index]):
                xs.append(x)
                zs.append(z)
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "zmin": min(zs),
        "zmax": max(zs),
    }


def _union_view_bounds(*scene_bounds: dict[str, float]) -> dict[str, float]:
    xmin = min(item["xmin"] for item in scene_bounds)
    xmax = max(item["xmax"] for item in scene_bounds)
    zmin = min(item["zmin"] for item in scene_bounds)
    zmax = max(item["zmax"] for item in scene_bounds)
    pad_x = max((xmax - xmin) * 0.06, 0.05)
    pad_z = max((zmax - zmin) * 0.06, 0.05)
    return {
        "xmin": xmin - pad_x,
        "xmax": xmax + pad_x,
        "zmin": zmin - pad_z,
        "zmax": zmax + pad_z,
    }


def _map_point(bounds: dict[str, float], point: tuple[float, float], size: int) -> tuple[float, float]:
    x, z = point
    pad = 8.0
    width = max(bounds["xmax"] - bounds["xmin"], 1.0e-4)
    depth = max(bounds["zmax"] - bounds["zmin"], 1.0e-4)
    usable = size - 2.0 * pad
    px = pad + (x - bounds["xmin"]) / width * usable
    py = size - (pad + (z - bounds["zmin"]) / depth * usable)
    return px, py


def _draw_scene(
    *,
    draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    bounds: dict[str, float],
    size: int,
    layout_pose: torch.Tensor,
    visible_pose: torch.Tensor,
    visible_mask: torch.Tensor,
    hidden_pose: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> None:
    room = _decode_layout_bounds(layout_pose)
    room_rect = [
        _map_point(bounds, (room["xmin"], room["zmax"]), size),
        _map_point(bounds, (room["xmax"], room["zmin"]), size),
    ]
    draw.rectangle(
        [room_rect[0][0], room_rect[0][1], room_rect[1][0], room_rect[1][1]],
        outline=ROOM_COLOR,
        width=ROOM_LINE_WIDTH,
    )
    for pose_bank, pose_mask, color in (
        (visible_pose, visible_mask, VISIBLE_COLOR),
        (hidden_pose, hidden_mask, HIDDEN_COLOR),
    ):
        for index in range(int(pose_mask.shape[0])):
            if not bool(pose_mask[index].item()):
                continue
            polygon = [_map_point(bounds, point, size) for point in _pose_to_corners_xz(pose_bank[index])]
            draw.polygon(polygon, fill=color, outline=color)
            mask_draw.polygon(polygon, fill=255, outline=255)


def _render_semantic_scene(
    *,
    size: int,
    bounds: dict[str, float],
    layout_pose: torch.Tensor,
    visible_pose: torch.Tensor,
    visible_mask: torch.Tensor,
    hidden_pose: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    canvas = Image.new("RGB", (size, size), FIGURE_BG)
    mask_img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    mask_draw = ImageDraw.Draw(mask_img)
    _draw_scene(
        draw=draw,
        mask_draw=mask_draw,
        bounds=bounds,
        size=size,
        layout_pose=layout_pose,
        visible_pose=visible_pose,
        visible_mask=visible_mask,
        hidden_pose=hidden_pose,
        hidden_mask=hidden_mask,
    )
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return rgb, mask


def _render_hidden_scene(
    *,
    size: int,
    bounds: dict[str, float],
    layout_pose: torch.Tensor,
    hidden_pose: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    empty_visible = torch.zeros((0, 8), dtype=layout_pose.dtype)
    empty_mask = torch.zeros((0,), dtype=torch.bool)
    return _render_semantic_scene(
        size=size,
        bounds=bounds,
        layout_pose=layout_pose,
        visible_pose=empty_visible,
        visible_mask=empty_mask,
        hidden_pose=hidden_pose,
        hidden_mask=hidden_mask,
    )


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1.0e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, channel_axis=-1, data_range=1.0))


def _mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    intersection = float(np.logical_and(pred_bin, target_bin).sum())
    union = float(np.logical_or(pred_bin, target_bin).sum())
    if union <= 0.0:
        return 1.0
    return intersection / union


def _make_lpips_metric(device: str, net: str, disabled: bool) -> Any | None:
    if disabled or lpips is None:
        return None
    metric = lpips.LPIPS(net=net)
    metric = metric.to(device)
    metric.eval()
    return metric


def _lpips_batch(
    metric: Any | None,
    pred_images: list[np.ndarray],
    target_images: list[np.ndarray],
    device: str,
) -> list[float]:
    if metric is None:
        return [float("nan")] * len(pred_images)
    pred = torch.from_numpy(np.stack(pred_images, axis=0)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(np.stack(target_images, axis=0)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    pred = pred.mul(2.0).sub(1.0)
    target = target.mul(2.0).sub(1.0)
    with torch.no_grad():
        values = metric(pred, target).view(-1).detach().cpu().tolist()
    return [float(item) for item in values]


def _array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _obs_to_panel(obs_image: torch.Tensor, title: str) -> Image.Image:
    obs = obs_image.detach().cpu()
    if obs.shape[0] >= 3:
        image = np.clip(obs[:3].permute(1, 2, 0).numpy(), 0.0, 1.0)
        panel = _array_to_image(image)
    else:
        depth = obs.squeeze().numpy()
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        panel = Image.fromarray(np.clip(depth * 255.0, 0.0, 255.0).astype(np.uint8), mode="L").convert("RGB")
    panel = ImageOps.contain(panel, (PANEL_SIZE, PANEL_SIZE))
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE + HEADER_H), FIGURE_BG)
    canvas.paste(panel, ((PANEL_SIZE - panel.width) // 2, HEADER_H + (PANEL_SIZE - panel.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=TEXT_COLOR, font=ImageFont.load_default())
    return canvas


def _render_panel(image: np.ndarray, title: str, subtitle: str = "") -> Image.Image:
    panel = ImageOps.contain(_array_to_image(image), (PANEL_SIZE, PANEL_SIZE))
    canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE + HEADER_H), FIGURE_BG)
    canvas.paste(panel, ((PANEL_SIZE - panel.width) // 2, HEADER_H + (PANEL_SIZE - panel.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 10), title, fill=TEXT_COLOR, font=font)
    if subtitle:
        draw.text((12, 24), subtitle, fill=TEXT_COLOR, font=font)
    return canvas


def _compose_figure(sample_id: str, panels: list[Image.Image], footer: str) -> Image.Image:
    gap = 10
    width = len(panels) * PANEL_SIZE + (len(panels) - 1) * gap
    height = PANEL_SIZE + HEADER_H + 52
    canvas = Image.new("RGB", (width, height), FIGURE_BG)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += PANEL_SIZE + gap
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, height - 34), f"sample_id={sample_id}", fill=TEXT_COLOR, font=font)
    draw.text((10, height - 18), footer, fill=TEXT_COLOR, font=font)
    return canvas


def _make_contact_sheet(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("cannot build contact sheet from zero images")
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


def _aggregate_mean(values: list[float]) -> float | None:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def main() -> None:
    args = parse_args()
    if args.seed >= 0:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

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
        raise RuntimeError("render eval found zero packet files after filtering")

    _, loader = TRAIN_UTILS.make_dataloader(
        packet_paths,
        batch_size=args.batch_size,
        image_size=int(cfg["data"].get("image_size", 512)),
        preload_packets=True,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=2,
    )

    model = TRAIN_UTILS.build_model(cfg).to(device)
    TRAIN_UTILS.load_model_state_compat(model, checkpoint["model_state"])
    model.eval()

    lpips_metric = _make_lpips_metric(device=device, net=args.lpips_net, disabled=bool(args.disable_lpips))
    output_dir = Path(args.output_dir)
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    save_summary = Path(args.save_summary) if args.save_summary else output_dir / "render_eval_summary.json"

    metric_totals: dict[str, list[float]] = {
        "scene_psnr": [],
        "scene_ssim": [],
        "scene_lpips": [],
        "scene_mask_iou": [],
        "hidden_psnr": [],
        "hidden_ssim": [],
        "hidden_lpips": [],
        "hidden_mask_iou": [],
    }
    per_scene_records: list[dict[str, Any]] = []
    contact_images: list[Image.Image] = []
    total_scenes = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            gt_layout = batch.target.layout_gt.detach().cpu()
            gt_visible = batch.target.visible_amodal_pose_gt.detach().cpu()
            gt_visible_mask = batch.target.visible_loss_mask.detach().cpu()
            gt_hidden = batch.target.hidden_pose_gt.detach().cpu()
            gt_hidden_mask = batch.target.hidden_gt_mask.detach().cpu()
            obs_image = batch.cond.obs_image.detach().cpu()

            scene_metric_bank: list[list[dict[str, float]]] = [[] for _ in range(batch_size)]
            hidden_metric_bank: list[list[dict[str, float]]] = [[] for _ in range(batch_size)]
            preview_payloads: list[dict[str, Any]] = [{} for _ in range(batch_size)]

            for sample_index in range(max(1, int(args.num_posterior_samples))):
                sample = model.sample_posterior(batch, num_sampling_steps=args.num_inference_steps)
                pred_layout = sample["layout"].detach().cpu()
                pred_visible = sample["visible"][..., : TRAIN_UTILS.D_POSE].detach().cpu()
                pred_hidden = sample["hidden"][..., : TRAIN_UTILS.D_POSE].detach().cpu()
                pred_hidden_mask = (sample["hidden_exist_probs"] > 0.5).detach().cpu()

                pred_full_images: list[np.ndarray] = []
                gt_full_images: list[np.ndarray] = []
                pred_hidden_images: list[np.ndarray] = []
                gt_hidden_images: list[np.ndarray] = []

                for row in range(batch_size):
                    gt_scene_bounds = _scene_world_bounds(
                        gt_layout[row],
                        gt_visible[row],
                        gt_visible_mask[row],
                        gt_hidden[row],
                        gt_hidden_mask[row],
                    )
                    pred_scene_bounds = _scene_world_bounds(
                        pred_layout[row],
                        pred_visible[row],
                        gt_visible_mask[row],
                        pred_hidden[row],
                        pred_hidden_mask[row],
                    )
                    unified_bounds = _union_view_bounds(gt_scene_bounds, pred_scene_bounds)
                    gt_full, gt_full_mask = _render_semantic_scene(
                        size=args.render_size,
                        bounds=unified_bounds,
                        layout_pose=gt_layout[row],
                        visible_pose=gt_visible[row],
                        visible_mask=gt_visible_mask[row],
                        hidden_pose=gt_hidden[row],
                        hidden_mask=gt_hidden_mask[row],
                    )
                    gt_hidden_only, gt_hidden_only_mask = _render_hidden_scene(
                        size=args.render_size,
                        bounds=unified_bounds,
                        layout_pose=gt_layout[row],
                        hidden_pose=gt_hidden[row],
                        hidden_mask=gt_hidden_mask[row],
                    )
                    pred_full, pred_full_mask = _render_semantic_scene(
                        size=args.render_size,
                        bounds=unified_bounds,
                        layout_pose=pred_layout[row],
                        visible_pose=pred_visible[row],
                        visible_mask=gt_visible_mask[row],
                        hidden_pose=pred_hidden[row],
                        hidden_mask=pred_hidden_mask[row],
                    )
                    pred_hidden_only, pred_hidden_only_mask = _render_hidden_scene(
                        size=args.render_size,
                        bounds=unified_bounds,
                        layout_pose=pred_layout[row],
                        hidden_pose=pred_hidden[row],
                        hidden_mask=pred_hidden_mask[row],
                    )
                    pred_full_images.append(pred_full)
                    gt_full_images.append(gt_full)
                    pred_hidden_images.append(pred_hidden_only)
                    gt_hidden_images.append(gt_hidden_only)
                    scene_metric_bank[row].append(
                        {
                            "psnr": _psnr(pred_full, gt_full),
                            "ssim": _ssim(pred_full, gt_full),
                            "mask_iou": _mask_iou(pred_full_mask, gt_full_mask),
                        }
                    )
                    hidden_metric_bank[row].append(
                        {
                            "psnr": _psnr(pred_hidden_only, gt_hidden_only),
                            "ssim": _ssim(pred_hidden_only, gt_hidden_only),
                            "mask_iou": _mask_iou(pred_hidden_only_mask, gt_hidden_only_mask),
                        }
                    )
                    if sample_index == 0:
                        preview_payloads[row] = {
                            "pred_full": pred_full,
                            "gt_full": gt_full,
                            "pred_hidden": pred_hidden_only,
                            "gt_hidden": gt_hidden_only,
                            "pred_hidden_count": int(pred_hidden_mask[row].sum().item()),
                        }

                scene_lpips_values = _lpips_batch(lpips_metric, pred_full_images, gt_full_images, device=device)
                hidden_lpips_values = _lpips_batch(lpips_metric, pred_hidden_images, gt_hidden_images, device=device)
                for row in range(batch_size):
                    scene_metric_bank[row][-1]["lpips"] = scene_lpips_values[row]
                    hidden_metric_bank[row][-1]["lpips"] = hidden_lpips_values[row]

            for row, sample_id in enumerate(batch.meta.sample_ids):
                scene_means = {
                    key: _aggregate_mean([metrics[key] for metrics in scene_metric_bank[row]])
                    for key in scene_metric_bank[row][0].keys()
                }
                hidden_means = {
                    key: _aggregate_mean([metrics[key] for metrics in hidden_metric_bank[row]])
                    for key in hidden_metric_bank[row][0].keys()
                }
                metric_totals["scene_psnr"].append(float(scene_means["psnr"]))
                metric_totals["scene_ssim"].append(float(scene_means["ssim"]))
                metric_totals["scene_mask_iou"].append(float(scene_means["mask_iou"]))
                metric_totals["hidden_psnr"].append(float(hidden_means["psnr"]))
                metric_totals["hidden_ssim"].append(float(hidden_means["ssim"]))
                metric_totals["hidden_mask_iou"].append(float(hidden_means["mask_iou"]))
                if scene_means["lpips"] is not None:
                    metric_totals["scene_lpips"].append(float(scene_means["lpips"]))
                if hidden_means["lpips"] is not None:
                    metric_totals["hidden_lpips"].append(float(hidden_means["lpips"]))

                panels = [
                    _obs_to_panel(obs_image[row], title="Observed RGBD"),
                    _render_panel(
                        preview_payloads[row]["gt_full"],
                        title="GT top-down",
                        subtitle=f"hidden={int(gt_hidden_mask[row].sum().item())}",
                    ),
                    _render_panel(
                        preview_payloads[row]["pred_full"],
                        title="Pred top-down",
                        subtitle=f"hidden={preview_payloads[row]['pred_hidden_count']}",
                    ),
                    _render_panel(
                        preview_payloads[row]["pred_hidden"],
                        title="Pred hidden only",
                        subtitle=(
                            f"PSNR={scene_means['psnr']:.2f} "
                            f"SSIM={scene_means['ssim']:.3f} "
                            f"LPIPS={scene_means['lpips']:.3f}" if scene_means["lpips"] is not None else
                            f"PSNR={scene_means['psnr']:.2f} SSIM={scene_means['ssim']:.3f}"
                        ),
                    ),
                ]
                figure = _compose_figure(
                    sample_id=sample_id,
                    panels=panels,
                    footer=(
                        f"metric_space=topdown_semantic_render p={int(args.num_posterior_samples)} "
                        f"s={int(args.num_inference_steps)}"
                    ),
                )
                figure_path = renders_dir / f"{sample_id}.png"
                figure.save(figure_path)
                contact_images.append(figure)
                per_scene_records.append(
                    {
                        "sample_id": sample_id,
                        "scene_id": batch.meta.scene_ids[row],
                        "room_id": batch.meta.room_ids[row],
                        "camera_id": batch.meta.camera_ids[row],
                        "num_posterior_samples": int(args.num_posterior_samples),
                        "render_path": str(figure_path.relative_to(output_dir)),
                        "metric_space": "topdown_semantic_render",
                        "scene_psnr": scene_means["psnr"],
                        "scene_ssim": scene_means["ssim"],
                        "scene_lpips": scene_means["lpips"],
                        "scene_mask_iou": scene_means["mask_iou"],
                        "hidden_psnr": hidden_means["psnr"],
                        "hidden_ssim": hidden_means["ssim"],
                        "hidden_lpips": hidden_means["lpips"],
                        "hidden_mask_iou": hidden_means["mask_iou"],
                        "gt_hidden_count": int(gt_hidden_mask[row].sum().item()),
                        "pred_hidden_count": int(preview_payloads[row]["pred_hidden_count"]),
                    }
                )
                total_scenes += 1

    contact_sheet_path = output_dir / "contact_sheet.png"
    if contact_images:
        _make_contact_sheet(contact_images).save(contact_sheet_path)

    summary = {
        "checkpoint": args.checkpoint,
        "packet_dir": args.packet_dir,
        "config": args.config,
        "data_config": args.data_config,
        "runtime_config": args.runtime_config,
        "num_scenes": total_scenes,
        "num_posterior_samples": int(args.num_posterior_samples),
        "num_inference_steps": int(args.num_inference_steps),
        "render_size": int(args.render_size),
        "metric_space": "topdown_semantic_render",
        "lpips_enabled": bool(lpips_metric is not None),
        "lpips_backend": args.lpips_net if lpips_metric is not None else "",
        "output_dir": str(output_dir),
        "contact_sheet": str(contact_sheet_path.relative_to(output_dir)) if contact_images else "",
        "metrics": {
            "scene_psnr": _aggregate_mean(metric_totals["scene_psnr"]),
            "scene_ssim": _aggregate_mean(metric_totals["scene_ssim"]),
            "scene_lpips": _aggregate_mean(metric_totals["scene_lpips"]),
            "scene_mask_iou": _aggregate_mean(metric_totals["scene_mask_iou"]),
            "hidden_psnr": _aggregate_mean(metric_totals["hidden_psnr"]),
            "hidden_ssim": _aggregate_mean(metric_totals["hidden_ssim"]),
            "hidden_lpips": _aggregate_mean(metric_totals["hidden_lpips"]),
            "hidden_mask_iou": _aggregate_mean(metric_totals["hidden_mask_iou"]),
        },
        "per_scene": per_scene_records,
    }
    save_summary.parent.mkdir(parents=True, exist_ok=True)
    save_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
