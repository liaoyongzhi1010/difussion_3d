from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep geometry VAE jobs running when the GPU goes idle.")
    parser.add_argument("--config", default="configs/geometry_vae/supervisor_full.yaml")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def query_gpu() -> dict[str, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = proc.stdout.strip().splitlines()[0]
    util_str, mem_used_str, mem_total_str = [item.strip() for item in line.split(",")]
    return {
        "utilization_gpu": int(util_str),
        "memory_used_mib": int(mem_used_str),
        "memory_total_mib": int(mem_total_str),
    }


def list_processes() -> list[tuple[int, str]]:
    proc = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True)
    rows: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, args = stripped.split(maxsplit=1)
        rows.append((int(pid_str), args))
    return rows


def summary_epoch(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    summary = load_json(summary_path)
    if "last_epoch" in summary:
        return int(summary["last_epoch"])
    if "epochs" in summary:
        return int(summary["epochs"])
    return 0


def experiment_complete(experiment: dict[str, Any]) -> bool:
    return summary_epoch(Path(experiment["save_summary"])) >= int(experiment["target_epochs"])


def current_resume_path(experiment: dict[str, Any]) -> str:
    latest_path = Path(experiment["checkpoint_dir"]) / "latest.pt"
    if latest_path.exists():
        return str(latest_path)
    bootstrap = str(experiment.get("bootstrap_checkpoint", "")).strip()
    return bootstrap if bootstrap and Path(bootstrap).exists() else ""


def find_running_pid(experiment: dict[str, Any], processes: list[tuple[int, str]]) -> int | None:
    marker = str(experiment["save_summary"])
    for pid, args in processes:
        if "train_geometry_vae.py" in args and marker in args:
            return pid
    return None


def build_launch_cmd(runtime_cfg: dict[str, Any], experiment: dict[str, Any]) -> list[str]:
    cmd = [
        str(runtime_cfg["python_bin"]),
        "scripts/train/train_geometry_vae.py",
        "--config",
        str(experiment["config"]),
        "--runtime-config",
        str(experiment["runtime_config"]),
        "--object-root",
        str(experiment["object_root"]),
        "--batch-size",
        str(experiment["batch_size"]),
        "--epochs",
        str(experiment["target_epochs"]),
        "--checkpoint-dir",
        str(experiment["checkpoint_dir"]),
        "--save-summary",
        str(experiment["save_summary"]),
        "--save-every-epochs",
        str(experiment.get("save_every_epochs", 10)),
        "--seed",
        str(experiment.get("seed", 42)),
    ]
    resume_path = current_resume_path(experiment)
    if resume_path:
        cmd.extend(["--resume", resume_path])
    if bool(experiment.get("reset_optimizer", False)):
        cmd.append("--reset-optimizer")
    return cmd


def launch_experiment(runtime_cfg: dict[str, Any], experiment: dict[str, Any], *, dry_run: bool, supervisor_log: Path) -> int | None:
    cmd = build_launch_cmd(runtime_cfg, experiment)
    append_log(supervisor_log, f"launching {experiment['name']} :: {' '.join(cmd)}")
    if dry_run:
        return None
    env = os.environ.copy()
    for key, value in dict(runtime_cfg.get("env", {})).items():
        env[str(key)] = str(value)
    log_path = Path(experiment["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(runtime_cfg["workdir"]),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    append_log(supervisor_log, f"started {experiment['name']} with pid={process.pid}")
    return process.pid


def supervisor_step(cfg: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    runtime_cfg = dict(cfg["runtime"])
    experiments = [dict(item) for item in cfg.get("experiments", []) if bool(item.get("enabled", True))]
    supervisor_log = Path(runtime_cfg["log_path"])
    state_path = Path(runtime_cfg["state_path"])
    gpu = query_gpu()
    processes = list_processes()

    statuses: list[dict[str, Any]] = []
    running_project_jobs = 0
    pending: list[dict[str, Any]] = []
    for experiment in experiments:
        pid = find_running_pid(experiment, processes)
        epoch = summary_epoch(Path(experiment["save_summary"]))
        complete = epoch >= int(experiment["target_epochs"])
        if pid is not None:
            running_project_jobs += 1
        if not complete and pid is None:
            pending.append(experiment)
        statuses.append(
            {
                "name": experiment["name"],
                "pid": pid,
                "current_epoch": epoch,
                "target_epochs": int(experiment["target_epochs"]),
                "complete": complete,
                "resume_path": current_resume_path(experiment),
                "save_summary": str(experiment["save_summary"]),
                "log_path": str(experiment["log_path"]),
            }
        )

    max_parallel = int(runtime_cfg.get("max_parallel_jobs", 1))
    idle_util_below = int(runtime_cfg.get("idle_gpu_util_below", 10))
    idle_mem_below = int(runtime_cfg.get("idle_memory_used_mib_below", 2048))
    gpu_idle = gpu["utilization_gpu"] <= idle_util_below and gpu["memory_used_mib"] <= idle_mem_below

    launches: list[str] = []
    available_slots = max(0, max_parallel - running_project_jobs)
    allow_launch = running_project_jobs > 0 or gpu_idle
    if allow_launch and available_slots > 0:
        for experiment in pending[:available_slots]:
            launch_experiment(runtime_cfg, experiment, dry_run=dry_run, supervisor_log=supervisor_log)
            launches.append(experiment["name"])

    state = {
        "timestamp_utc": timestamp(),
        "gpu": gpu,
        "gpu_idle": gpu_idle,
        "running_project_jobs": running_project_jobs,
        "max_parallel_jobs": max_parallel,
        "allow_launch": allow_launch,
        "launches": launches,
        "experiments": statuses,
    }
    save_json(state_path, state)
    append_log(
        supervisor_log,
        (
            f"gpu util={gpu['utilization_gpu']} mem={gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB "
            f"project_jobs={running_project_jobs} launches={','.join(launches) if launches else 'none'}"
        ),
    )
    return state


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    runtime_cfg = dict(cfg["runtime"])
    poll_seconds = int(runtime_cfg.get("poll_seconds", 120))
    while True:
        supervisor_step(cfg, dry_run=bool(args.dry_run))
        if args.once:
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
