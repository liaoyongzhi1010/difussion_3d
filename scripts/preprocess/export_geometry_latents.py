from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from amodal_scene_diff.geometry import GeometryVAE


class GeometryExportDataset(Dataset[dict[str, Any]]):
    def __init__(self, object_paths: list[Path]) -> None:
        self.object_paths = object_paths

    def __len__(self) -> int:
        return len(self.object_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.object_paths[index]
        payload = torch.load(path, map_location="cpu")
        return {
            "object_uid": str(payload["object_uid"]),
            "surface_points": payload["surface_points"].float(),
            "surface_normals": payload["surface_normals"].float(),
            "class_id": int(payload.get("class_id", -1)),
            "quality_flag": str(payload.get("quality_flag", "unknown")),
            "source_path": str(path),
        }


def collate_export(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object_uid": [item["object_uid"] for item in batch],
        "surface_points": torch.stack([item["surface_points"] for item in batch], dim=0),
        "surface_normals": torch.stack([item["surface_normals"] for item in batch], dim=0),
        "class_id": torch.tensor([item["class_id"] for item in batch], dtype=torch.long),
        "quality_flag": [item["quality_flag"] for item in batch],
        "source_path": [item["source_path"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned geometry latents z_mu/z_logvar from a trained GeometryVAE.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uid-source-dir", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-summary", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"yaml config at {path} must load as dict")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> GeometryVAE:
    cfg = checkpoint.get("cfg", {})
    model_cfg = cfg.get("model", {})
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    model = GeometryVAE(
        input_dim=int(encoder_cfg.get("input_dim", 6)),
        latent_dim=int(model_cfg.get("latent_dim", 256)),
        encoder_hidden_dims=[int(v) for v in encoder_cfg.get("hidden_dims", [64, 128, 256, 512])],
        triplane_feat_dim=int(decoder_cfg.get("triplane_feat_dim", 16)),
        triplane_res_xy=int(decoder_cfg.get("triplane_res_xy", 32)),
        query_hidden_dims=[int(v) for v in decoder_cfg.get("query_hidden_dims", [256, 256, 128])],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model.to(device)


def resolve_object_paths(object_root: Path, uid_source_dir: Path | None) -> list[Path]:
    all_paths = sorted((object_root / "objects").glob("*.pt"))
    if uid_source_dir is None:
        return all_paths
    wanted = {path.stem for path in sorted(uid_source_dir.glob("*.pt"))}
    return [path for path in all_paths if path.stem in wanted]


def main() -> None:
    args = parse_args()
    object_root = Path(args.object_root)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.save_summary)
    uid_source_dir = Path(args.uid_source_dir) if args.uid_source_dir else None

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model_from_checkpoint(checkpoint, device=device)
    object_paths = resolve_object_paths(object_root, uid_source_dir)
    if not object_paths:
        raise RuntimeError("no geometry objects selected for latent export")

    dataset = GeometryExportDataset(object_paths)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_export,
        drop_last=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped_existing = 0
    failures: list[str] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            surface_inputs = torch.cat([batch["surface_points"], batch["surface_normals"]], dim=-1).to(device)
            z_mu, z_logvar = model.encoder(surface_inputs)
            z_mu = z_mu.detach().cpu()
            z_logvar = z_logvar.detach().cpu()

            for row, object_uid in enumerate(batch["object_uid"]):
                target_path = output_dir / f"{object_uid}.pt"
                if target_path.exists() and not args.overwrite:
                    skipped_existing += 1
                    continue
                payload = {
                    "z_mu": z_mu[row],
                    "z_logvar": z_logvar[row],
                    "object_uid": object_uid,
                    "class_id": int(batch["class_id"][row].item()),
                    "quality_flag": batch["quality_flag"][row],
                    "source_checkpoint": str(args.checkpoint),
                    "source_object_path": batch["source_path"][row],
                }
                try:
                    torch.save(payload, target_path)
                    exported += 1
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{object_uid}: {exc}")

            if batch_index % 10 == 0 or batch_index == len(loader):
                summary = {
                    "checkpoint": str(args.checkpoint),
                    "object_root": str(object_root),
                    "uid_source_dir": str(uid_source_dir) if uid_source_dir is not None else "",
                    "output_dir": str(output_dir),
                    "requested_objects": len(object_paths),
                    "exported_objects": exported,
                    "skipped_existing": skipped_existing,
                    "available_objects": exported + skipped_existing,
                    "failed_objects": len(failures),
                    "last_batch_index": batch_index,
                    "num_batches": len(loader),
                    "completed": batch_index == len(loader),
                }
                save_json(summary_path, summary)
                print(json.dumps(summary), flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "object_root": str(object_root),
        "uid_source_dir": str(uid_source_dir) if uid_source_dir is not None else "",
        "output_dir": str(output_dir),
        "requested_objects": len(object_paths),
        "exported_objects": exported,
        "skipped_existing": skipped_existing,
        "available_objects": exported + skipped_existing,
        "failed_objects": len(failures),
        "failures": failures[:100],
        "completed": True,
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
