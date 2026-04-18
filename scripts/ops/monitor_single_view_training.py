from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Monitor a single-view training run via process, log, GPU, and checkpoint freshness.')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--train-log', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--summary-path', default='')
    parser.add_argument('--state-path', required=True)
    parser.add_argument('--monitor-log', required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--stale-seconds', type=int, default=600)
    parser.add_argument('--once', action='store_true')
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def append_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(f'[{utc_timestamp()}] {message}\n')


def load_last_json_metric(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'step' in payload and 'metrics' in payload:
            return payload
    return {}


def list_checkpoints(checkpoint_dir: Path) -> list[str]:
    if not checkpoint_dir.exists():
        return []
    return sorted(path.name for path in checkpoint_dir.glob('step_*.pt'))


def query_gpu() -> dict[str, Any]:
    command = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits',
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    line = proc.stdout.strip().splitlines()[0]
    util, mem_used, mem_total, temp_c, power_w = [item.strip() for item in line.split(',')]
    return {
        'utilization_gpu': float(util),
        'memory_used_mib': float(mem_used),
        'memory_total_mib': float(mem_total),
        'temperature_c': float(temp_c),
        'power_w': float(power_w),
    }


def process_info(pid: int) -> dict[str, Any]:
    proc = subprocess.run(['ps', '-p', str(pid), '-o', 'pid=,etime=,%cpu=,%mem=,args='], capture_output=True, text=True)
    line = proc.stdout.strip()
    if not line:
        return {'alive': False}
    parts = line.split(maxsplit=4)
    if len(parts) < 5:
        return {'alive': False}
    pid_str, etime, cpu, mem, cmd = parts
    return {
        'alive': True,
        'pid': int(pid_str),
        'elapsed': etime,
        'cpu_percent': float(cpu),
        'mem_percent': float(mem),
        'cmd': cmd,
    }


def summarize_status(proc: dict[str, Any], gpu: dict[str, Any], log_age_sec: float | None, metric_payload: dict[str, Any]) -> str:
    if not proc.get('alive', False):
        return 'dead'
    if log_age_sec is not None and log_age_sec > 600 and gpu['utilization_gpu'] < 10:
        return 'stalled'
    if metric_payload:
        return 'training'
    return 'starting'


def monitor_step(args: argparse.Namespace) -> dict[str, Any]:
    log_path = Path(args.train_log)
    checkpoint_dir = Path(args.checkpoint_dir)
    state_path = Path(args.state_path)
    monitor_log = Path(args.monitor_log)
    summary_path = Path(args.summary_path) if args.summary_path else None

    proc = process_info(args.pid)
    gpu = query_gpu()
    metric_payload = load_last_json_metric(log_path)
    checkpoints = list_checkpoints(checkpoint_dir)

    log_mtime = log_path.stat().st_mtime if log_path.exists() else None
    now = time.time()
    log_age_sec = None if log_mtime is None else max(0.0, now - log_mtime)
    summary_exists = bool(summary_path is not None and summary_path.exists())

    latest_step = int(metric_payload.get('step', 0)) if metric_payload else 0
    metrics = dict(metric_payload.get('metrics', {})) if metric_payload else {}
    status = summarize_status(proc, gpu, log_age_sec, metric_payload)

    state = {
        'timestamp_utc': utc_timestamp(),
        'run_name': args.run_name,
        'status': status,
        'process': proc,
        'gpu': gpu,
        'train_log': {
            'path': str(log_path),
            'exists': log_path.exists(),
            'age_seconds': log_age_sec,
            'latest_step': latest_step,
            'latest_metrics': metrics,
        },
        'checkpoints': {
            'dir': str(checkpoint_dir),
            'count': len(checkpoints),
            'latest': checkpoints[-1] if checkpoints else '',
        },
        'summary': {
            'path': str(summary_path) if summary_path is not None else '',
            'exists': summary_exists,
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')

    loss_total = metrics.get('loss_total', None)
    hidden_diff = metrics.get('loss_hidden_diff', None)
    visible_direct = metrics.get('loss_visible_direct', None)
    append_line(
        monitor_log,
        (
            f"status={status} pid_alive={proc.get('alive', False)} step={latest_step} "
            f"loss_total={loss_total} visible_direct={visible_direct} hidden_diff={hidden_diff} "
            f"gpu={gpu['utilization_gpu']}% mem={gpu['memory_used_mib']}/{gpu['memory_total_mib']}MiB "
            f"checkpoints={len(checkpoints)} latest_ckpt={checkpoints[-1] if checkpoints else 'none'} "
            f"log_age={log_age_sec}"
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
