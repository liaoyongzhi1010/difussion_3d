from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from amodal_scene_diff.geometry import GeometryVAE


class GeometryObjectDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, object_paths: list[Path]) -> None:
        self.object_paths = object_paths

    def __len__(self) -> int:
        return len(self.object_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        payload = torch.load(self.object_paths[index], map_location="cpu")
        return {
            "surface_points": payload["surface_points"].float(),
            "surface_normals": payload["surface_normals"].float(),
            "query_points": payload["query_points"].float(),
            "query_sdf": payload["query_sdf"].float(),
            "query_occ": payload["query_occ"].float(),
        }


def collate_geometry(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train geometry VAE on exported real object geometry payloads.")
    parser.add_argument("--config", default="configs/geometry_vae/base.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/gpu_smoke.yaml")
    parser.add_argument("--object-root", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--save-summary", default="outputs/debug/geometry_vae_train_summary.json")
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-optimizer", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"yaml config at {path} must load as dict")
    return data


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"json at {path} must load as dict")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def stable_uid_bucket(path: Path) -> int:
    digest = hashlib.sha1(path.stem.encode("utf-8")).digest()
    return digest[0]


def split_object_paths(paths: list[Path], train_ratio: float) -> tuple[list[Path], list[Path]]:
    threshold = int(round(255 * train_ratio))
    train, val = [], []
    for path in sorted(paths):
        bucket = stable_uid_bucket(path)
        if bucket <= threshold:
            train.append(path)
        else:
            val.append(path)
    if not train or not val:
        pivot = max(1, int(len(paths) * train_ratio))
        train = sorted(paths)[:pivot]
        val = sorted(paths)[pivot:]
    return train, val


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=device.startswith("cuda")) for key, value in batch.items()}


def evaluate(model: GeometryVAE, loader: DataLoader, device: str, loss_cfg: dict[str, float]) -> dict[str, float]:
    model.eval()
    total = {"loss_total": 0.0, "loss_sdf": 0.0, "loss_occ": 0.0, "loss_kl": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            losses = model.compute_losses(
                surface_points=batch["surface_points"],
                surface_normals=batch["surface_normals"],
                query_points=batch["query_points"],
                query_sdf=batch["query_sdf"],
                query_occ=batch["query_occ"],
                lambda_kl=float(loss_cfg["lambda_kl"]),
                lambda_sdf=float(loss_cfg["lambda_sdf"]),
                lambda_occ=float(loss_cfg["lambda_occ"]),
            )
            batch_size = batch["surface_points"].shape[0]
            count += batch_size
            for key in total:
                total[key] += float(losses[key].detach().cpu().item()) * batch_size
    return {key: value / max(count, 1) for key, value in total.items()}


def save_checkpoint(
    path: Path,
    model: GeometryVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    best_val: float,
    best_checkpoint: str,
    object_root: str,
    num_train_objects: int,
    num_val_objects: int,
    resume_from: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "cfg": cfg,
            "metrics": metrics,
            "best_val": float(best_val),
            "best_checkpoint": str(best_checkpoint),
            "object_root": str(object_root),
            "num_train_objects": int(num_train_objects),
            "num_val_objects": int(num_val_objects),
            "resume_from": str(resume_from),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def build_summary(
    *,
    object_root: Path,
    num_train_objects: int,
    num_val_objects: int,
    epochs: int,
    batch_size: int,
    device: str,
    model: GeometryVAE,
    best_val: float,
    best_ckpt: str,
    train_history: list[dict[str, Any]],
    runtime_sec: float,
    start_epoch: int,
    current_epoch: int,
    resume_from: str,
) -> dict[str, Any]:
    return {
        "object_root": str(object_root),
        "num_train_objects": int(num_train_objects),
        "num_val_objects": int(num_val_objects),
        "epochs": int(epochs),
        "start_epoch": int(start_epoch),
        "last_epoch": int(current_epoch),
        "batch_size": int(batch_size),
        "device": device,
        "num_parameters": model.num_parameters,
        "best_val_loss": float(best_val),
        "best_checkpoint": str(best_ckpt),
        "resume_from": str(resume_from),
        "history": train_history,
        "runtime_sec": float(runtime_sec),
        "completed": bool(current_epoch >= epochs),
    }


def main() -> None:
    args = parse_args()
    cfg = deep_merge(load_yaml(args.config), {"runtime": load_yaml(args.runtime_config)})
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    object_root = Path(args.object_root or cfg["data"]["object_cache_root"])
    object_paths = sorted((object_root / "objects").glob("*.pt"))
    if args.max_objects > 0:
        object_paths = object_paths[: args.max_objects]
    if len(object_paths) < 8:
        raise RuntimeError(f"not enough geometry objects to train: found {len(object_paths)} in {object_root}")

    train_paths, val_paths = split_object_paths(object_paths, train_ratio=float(args.train_ratio))
    train_ds = GeometryObjectDataset(train_paths)
    val_ds = GeometryObjectDataset(val_paths)

    batch_size = int(args.batch_size or cfg["training"].get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_geometry, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_geometry, drop_last=False)

    device = str(cfg["runtime"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = GeometryVAE(
        input_dim=int(cfg["model"]["encoder"].get("input_dim", 6)),
        latent_dim=int(cfg["model"].get("latent_dim", 256)),
        encoder_hidden_dims=[int(v) for v in cfg["model"]["encoder"].get("hidden_dims", [64, 128, 256, 512])],
        triplane_feat_dim=int(cfg["model"]["decoder"].get("triplane_feat_dim", 16)),
        triplane_res_xy=int(cfg["model"]["decoder"].get("triplane_res_xy", 32)),
        query_hidden_dims=[int(v) for v in cfg["model"]["decoder"].get("query_hidden_dims", [256, 256, 128])],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1.0e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1.0e-4)),
    )
    loss_cfg = cfg["loss"]
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.save_summary).parent / "checkpoints_geometry_vae"
    latest_ckpt = checkpoint_dir / "latest.pt"
    save_every_epochs = max(1, int(args.save_every_epochs))
    summary_path = Path(args.save_summary)

    best_val = float("inf")
    best_ckpt = ""
    train_history: list[dict[str, Any]] = []
    start_epoch = 0
    resume_from = ""

    if summary_path.exists():
        summary_data = load_json(summary_path)
        if isinstance(summary_data.get("history"), list):
            train_history = [item for item in summary_data["history"] if isinstance(item, dict)]
        best_val = float(summary_data.get("best_val_loss", best_val))
        best_ckpt = str(summary_data.get("best_checkpoint", best_ckpt))

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if not args.reset_optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_from = str(resume_path)
        best_val = float(checkpoint.get("best_val", best_val))
        best_ckpt = str(checkpoint.get("best_checkpoint", best_ckpt or resume_path))
        if train_history:
            train_history = [item for item in train_history if int(item.get("epoch", 0)) <= start_epoch]
        print(
            json.dumps(
                {
                    "event": "resume_loaded",
                    "resume_from": resume_from,
                    "start_epoch": start_epoch,
                    "reset_optimizer": bool(args.reset_optimizer),
                }
            ),
            flush=True,
        )

    start_time = time.perf_counter()
    total_epochs = int(args.epochs)
    if start_epoch >= total_epochs:
        summary = build_summary(
            object_root=object_root,
            num_train_objects=len(train_paths),
            num_val_objects=len(val_paths),
            epochs=total_epochs,
            batch_size=batch_size,
            device=device,
            model=model,
            best_val=best_val,
            best_ckpt=best_ckpt,
            train_history=train_history,
            runtime_sec=time.perf_counter() - start_time,
            start_epoch=start_epoch,
            current_epoch=start_epoch,
            resume_from=resume_from,
        )
        save_json(summary_path, summary)
        print(
            json.dumps(
                {
                    "event": "already_complete",
                    "start_epoch": start_epoch,
                    "epochs": total_epochs,
                    "save_summary": str(summary_path),
                }
            ),
            flush=True,
        )
        return

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        accum = {"loss_total": 0.0, "loss_sdf": 0.0, "loss_occ": 0.0, "loss_kl": 0.0}
        seen = 0
        epoch_start = time.perf_counter()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_losses(
                surface_points=batch["surface_points"],
                surface_normals=batch["surface_normals"],
                query_points=batch["query_points"],
                query_sdf=batch["query_sdf"],
                query_occ=batch["query_occ"],
                lambda_kl=float(loss_cfg["lambda_kl"]),
                lambda_sdf=float(loss_cfg["lambda_sdf"]),
                lambda_occ=float(loss_cfg["lambda_occ"]),
            )
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("clip_grad_norm", 1.0)))
            optimizer.step()
            batch_size_now = batch["surface_points"].shape[0]
            seen += batch_size_now
            for key in accum:
                accum[key] += float(losses[key].detach().cpu().item()) * batch_size_now

        train_metrics = {key: value / max(seen, 1) for key, value in accum.items()}
        val_metrics = evaluate(model, val_loader, device=device, loss_cfg=loss_cfg)
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_sec": time.perf_counter() - epoch_start,
        }
        train_history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)

        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            best_ckpt = str(checkpoint_dir / "best.pt")
            save_checkpoint(
                Path(best_ckpt),
                model,
                optimizer,
                epoch,
                cfg,
                epoch_metrics,
                best_val=best_val,
                best_checkpoint=best_ckpt,
                object_root=str(object_root),
                num_train_objects=len(train_paths),
                num_val_objects=len(val_paths),
                resume_from=resume_from,
            )

        save_checkpoint(
            latest_ckpt,
            model,
            optimizer,
            epoch,
            cfg,
            epoch_metrics,
            best_val=best_val,
            best_checkpoint=best_ckpt,
            object_root=str(object_root),
            num_train_objects=len(train_paths),
            num_val_objects=len(val_paths),
            resume_from=resume_from,
        )

        if epoch % save_every_epochs == 0 or epoch == total_epochs:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                cfg,
                epoch_metrics,
                best_val=best_val,
                best_checkpoint=best_ckpt,
                object_root=str(object_root),
                num_train_objects=len(train_paths),
                num_val_objects=len(val_paths),
                resume_from=resume_from,
            )

        summary = build_summary(
            object_root=object_root,
            num_train_objects=len(train_paths),
            num_val_objects=len(val_paths),
            epochs=total_epochs,
            batch_size=batch_size,
            device=device,
            model=model,
            best_val=best_val,
            best_ckpt=best_ckpt,
            train_history=train_history,
            runtime_sec=time.perf_counter() - start_time,
            start_epoch=start_epoch,
            current_epoch=epoch,
            resume_from=resume_from,
        )
        save_json(summary_path, summary)

    summary = build_summary(
        object_root=object_root,
        num_train_objects=len(train_paths),
        num_val_objects=len(val_paths),
        epochs=total_epochs,
        batch_size=batch_size,
        device=device,
        model=model,
        best_val=best_val,
        best_ckpt=best_ckpt,
        train_history=train_history,
        runtime_sec=time.perf_counter() - start_time,
        start_epoch=start_epoch,
        current_epoch=total_epochs,
        resume_from=resume_from,
    )
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
