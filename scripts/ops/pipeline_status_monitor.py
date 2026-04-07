from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_pipeline_supervisor import append_log, load_json, load_yaml, query_gpu, save_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Monitor watchdog freshness, GPU state, and outstanding project work.')
    parser.add_argument('--config', default='configs/pipeline/watchdog.yaml')
    parser.add_argument('--log-path', default='outputs/gpu_monitor.log')
    parser.add_argument('--state-path', default='outputs/gpu_monitor_state.json')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--stale-seconds', type=int, default=180)
    parser.add_argument('--hot-cpu-threshold', type=float, default=5.0)
    parser.add_argument('--once', action='store_true')
    return parser.parse_args()


def parse_timestamp_utc(raw: str) -> float | None:
    if not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def list_hot_jobs(min_cpu: float) -> list[dict[str, Any]]:
    proc = subprocess.run(['ps', '-eo', 'pid=,pcpu=,args='], capture_output=True, text=True, check=True)
    rows: list[dict[str, Any]] = []
    skip_markers = (
        'pipeline_watchdog.py',
        'pipeline_status_monitor.py',
        'research_pipeline_supervisor.py',
        'nvidia-smi',
        'node /usr/bin/codex',
        '/usr/bin/codex resume',
        'codex-linux-x64/vendor',
        'ps -eo',
        'bash monitor_gpu.sh',
    )
    include_markers = (
        '/root/3d/generation',
        '/root/project/LSSR',
        'render_partitions.py',
        'train_scene_diffusion.py',
        'eval_scene_posterior.py',
        'train_geometry_vae.py',
        'export_geometry_latents.py',
        'rewrite_packets_with_geometry_latents.py',
    )

    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_str, cpu_str, args = parts
        pid = int(pid_str)
        if pid == os.getpid():
            continue
        if any(marker in args for marker in skip_markers):
            continue
        cpu = float(cpu_str)
        if cpu < min_cpu and not any(marker in args for marker in include_markers):
            continue
        rows.append(
            {
                'pid': pid,
                'cpu_percent': cpu,
                'cmd': args,
            }
        )
    rows.sort(key=lambda item: item['cpu_percent'], reverse=True)
    return rows[:5]


def shorten_command(command: str, limit: int = 120) -> str:
    if len(command) <= limit:
        return command
    return command[: limit - 3] + '...'


def find_watchdog_pid() -> int | None:
    proc = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, args = stripped.split(maxsplit=1)
        if 'scripts/ops/pipeline_watchdog.py' in args:
            return int(pid_str)
    return None


def summarize_pipeline(entry: dict[str, Any]) -> dict[str, str]:
    name = str(entry.get('name', 'unknown'))
    active_stage = str(entry.get('active_stage', '')).strip()
    running_jobs = int(entry.get('running_project_jobs', 0))
    launches = [str(item) for item in entry.get('launches', [])]
    backfill_attempted = bool(entry.get('backfill_attempted', False))
    backfill_launched = bool(entry.get('backfill_launched', False))
    backfill_output = str(entry.get('backfill_output', '')).strip()

    if running_jobs > 0:
        status = f'running:{running_jobs}'
    elif launches:
        status = 'launching:' + ','.join(launches)
    elif active_stage:
        status = f'pending:{active_stage}'
    elif backfill_launched:
        status = 'backfill-launched'
    elif backfill_attempted and backfill_output == 'no_pending_backfill':
        status = 'complete'
    elif not active_stage:
        status = 'complete'
    else:
        status = 'idle'
    return {'name': name, 'status': status}


def monitor_step(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    runtime_cfg = dict(cfg['runtime'])
    watchdog_state_path = Path(runtime_cfg['state_path'])
    watchdog_poll_seconds = int(runtime_cfg.get('poll_seconds', 60))
    stale_seconds = max(int(args.stale_seconds), watchdog_poll_seconds * 2 + 30)

    monitor_log = Path(args.log_path)
    monitor_state_path = Path(args.state_path)

    watchdog_state = load_json(watchdog_state_path)
    gpu = query_gpu()
    watchdog_pid = find_watchdog_pid()

    now_ts = time.time()
    watchdog_state_ts = parse_timestamp_utc(str(watchdog_state.get('timestamp_utc', '')))
    watchdog_lag_seconds = None if watchdog_state_ts is None else max(0, int(now_ts - watchdog_state_ts))
    watchdog_fresh = bool(
        watchdog_pid is not None
        and watchdog_state_ts is not None
        and watchdog_lag_seconds is not None
        and watchdog_lag_seconds <= stale_seconds
    )

    pipelines = [summarize_pipeline(dict(entry)) for entry in watchdog_state.get('pipelines', [])]
    hot_jobs = list_hot_jobs(float(args.hot_cpu_threshold))

    state = {
        'timestamp_utc': timestamp(),
        'watchdog': {
            'pid': watchdog_pid,
            'state_path': str(watchdog_state_path),
            'fresh': watchdog_fresh,
            'lag_seconds': watchdog_lag_seconds,
            'stale_after_seconds': stale_seconds,
        },
        'gpu': gpu,
        'pipelines': pipelines,
        'hot_jobs': hot_jobs,
    }
    save_json(monitor_state_path, state)

    watchdog_text = 'fresh' if watchdog_fresh else 'stale'
    pid_text = str(watchdog_pid) if watchdog_pid is not None else 'none'
    lag_text = 'unknown' if watchdog_lag_seconds is None else f'{watchdog_lag_seconds}s'
    gpu_state_text = 'busy' if gpu['utilization_gpu'] > 0 or gpu['memory_used_mib'] > 2048 else 'idle'
    pipeline_text = ' | '.join(f"{item['name']}:{item['status']}" for item in pipelines) if pipelines else 'none'
    hot_jobs_text = (
        ' ; '.join(f"{item['pid']}:{item['cpu_percent']:.1f}% {shorten_command(str(item['cmd']))}" for item in hot_jobs)
        if hot_jobs
        else 'none'
    )
    append_log(
        monitor_log,
        (
            f'watchdog={watchdog_text}(pid={pid_text},lag={lag_text}) '
            f'gpu={gpu_state_text}({gpu["utilization_gpu"]}%,{gpu["memory_used_mib"]}/{gpu["memory_total_mib"]}MiB) '
            f'pipelines={pipeline_text} hot_jobs={hot_jobs_text}'
        ),
    )
    return state


def main() -> None:
    args = parse_args()
    while True:
        monitor_step(args)
        if args.once:
            break
        time.sleep(int(args.poll_seconds))


if __name__ == '__main__':
    main()
