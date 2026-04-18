from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from amodal_scene_diff.datasets import SingleViewPacketDataset, collate_single_view_packets
from amodal_scene_diff.models.diffusion import SingleViewReconstructionDiffusion
from amodal_scene_diff.models.geometry import GeometryVAE
from torch.utils.data import DataLoader


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
    parser = argparse.ArgumentParser(description="Export explicit single-view 3D scene representations as tri-plane scene codes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--config", default="configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml")
    parser.add_argument("--data-config", default="configs/data/pixarmesh_single_view_main.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--geometry-config", default="configs/geometry_vae/heavy.yaml")
    parser.add_argument("--geometry-checkpoint", default="outputs/real_data/geometry_vae_objects_v1_full/checkpoints_geometry_vae_heavy_fullresume/latest.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/debug/single_view_scene_repr")
    parser.add_argument("--save-summary", default="outputs/debug/single_view_scene_repr_summary.json")
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


def discover_packet_paths(packet_dir: str | Path, max_samples: int) -> list[Path]:
    root = Path(packet_dir)
    packet_paths = sorted(root.glob("*.pt"))
    if max_samples > 0:
        packet_paths = packet_paths[:max_samples]
    return packet_paths


def build_model(cfg: dict[str, Any]) -> SingleViewReconstructionDiffusion:
    return SingleViewReconstructionDiffusion.from_config(cfg)


def build_geometry_decoder(geometry_cfg: dict[str, Any], checkpoint_path: str | Path, device: str) -> GeometryVAE:
    model = GeometryVAE(
        input_dim=int(geometry_cfg["model"]["encoder"].get("input_dim", 6)),
        latent_dim=int(geometry_cfg["model"].get("latent_dim", 256)),
        encoder_hidden_dims=[int(v) for v in geometry_cfg["model"]["encoder"].get("hidden_dims", [64, 128, 256, 512])],
        triplane_feat_dim=int(geometry_cfg["model"]["decoder"].get("triplane_feat_dim", 16)),
        triplane_res_xy=int(geometry_cfg["model"]["decoder"].get("triplane_res_xy", 32)),
        query_hidden_dims=[int(v) for v in geometry_cfg["model"]["decoder"].get("query_hidden_dims", [256, 256, 128])],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    diffusion_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    runtime_cfg = load_yaml(args.runtime_config)
    cfg = deep_merge(deep_merge(diffusion_cfg, {"data": data_cfg}), {"runtime": runtime_cfg})
    geometry_cfg = load_yaml(args.geometry_config)

    device = str(cfg["runtime"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    packet_paths = discover_packet_paths(args.packet_dir, max_samples=args.max_samples)
    dataset = SingleViewPacketDataset(packet_paths, preload_packets=True, image_size=int(cfg["data"].get("image_size", 512)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_single_view_packets)

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    TRAIN_UTILS.load_model_state_compat(model, checkpoint["model_state"])
    model.eval()

    geometry_model = build_geometry_decoder(geometry_cfg, args.geometry_checkpoint, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            samples = model.sample_posterior(batch, num_sampling_steps=args.num_inference_steps)
            visible_pose = samples["visible"][..., :8]
            visible_z = samples["visible"][..., 8:]
            hidden_pose = samples["hidden"][..., :8]
            hidden_z = samples["hidden"][..., 8:]
            visible_triplanes = geometry_model.decode_triplanes(visible_z.reshape(-1, visible_z.shape[-1]))
            hidden_triplanes = geometry_model.decode_triplanes(hidden_z.reshape(-1, hidden_z.shape[-1]))
            visible_triplanes = visible_triplanes.view(visible_z.shape[0], visible_z.shape[1], *visible_triplanes.shape[1:])
            hidden_triplanes = hidden_triplanes.view(hidden_z.shape[0], hidden_z.shape[1], *hidden_triplanes.shape[1:])

            for row, sample_id in enumerate(batch.meta.sample_ids):
                payload = {
                    "sample_id": sample_id,
                    "scene_id": batch.meta.scene_ids[row],
                    "room_id": batch.meta.room_ids[row],
                    "camera_id": batch.meta.camera_ids[row],
                    "layout": samples["layout"][row].detach().cpu(),
                    "visible_pose": visible_pose[row].detach().cpu(),
                    "visible_z": visible_z[row].detach().cpu(),
                    "visible_exist_probs": samples["visible_exist_probs"][row].detach().cpu(),
                    "visible_cls_probs": samples["visible_cls_probs"][row].detach().cpu(),
                    "visible_triplanes": visible_triplanes[row].detach().cpu(),
                    "hidden_pose": hidden_pose[row].detach().cpu(),
                    "hidden_z": hidden_z[row].detach().cpu(),
                    "hidden_exist_probs": samples["hidden_exist_probs"][row].detach().cpu(),
                    "hidden_cls_probs": samples["hidden_cls_probs"][row].detach().cpu(),
                    "hidden_triplanes": hidden_triplanes[row].detach().cpu(),
                    "support_logits": samples["support_logits"][row].detach().cpu(),
                    "floor_logits": samples["floor_logits"][row].detach().cpu(),
                    "wall_logits": samples["wall_logits"][row].detach().cpu(),
                }
                target_path = output_dir / f"{sample_id}.pt"
                torch.save(payload, target_path)
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "path": str(target_path),
                        "visible_slots": int(visible_pose.shape[1]),
                        "hidden_slots": int(hidden_pose.shape[1]),
                    }
                )

    summary = {
        "checkpoint": args.checkpoint,
        "geometry_checkpoint": args.geometry_checkpoint,
        "packet_dir": args.packet_dir,
        "num_exported": len(manifest),
        "output_dir": str(output_dir),
        "manifest": manifest,
    }
    save_path = Path(args.save_summary)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
