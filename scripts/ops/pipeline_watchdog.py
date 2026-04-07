from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from research_pipeline_supervisor import append_log, load_yaml, query_gpu, save_json, supervisor_step, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tick multiple pipeline supervisors from a single watchdog loop.')
    parser.add_argument('--config', default='configs/pipeline/watchdog.yaml')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def watchdog_step(cfg: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    runtime_cfg = dict(cfg['runtime'])
    watchdog_log = Path(runtime_cfg['log_path'])
    state_path = Path(runtime_cfg['state_path'])
    pipeline_states: list[dict[str, Any]] = []
    errors: list[str] = []

    gpu_before = query_gpu()
    gpu_before_util = gpu_before['utilization_gpu']
    gpu_before_mem_used = gpu_before['memory_used_mib']
    gpu_before_mem_total = gpu_before['memory_total_mib']
    append_log(
        watchdog_log,
        f'tick start :: gpu util={gpu_before_util} mem={gpu_before_mem_used}/{gpu_before_mem_total} MiB',
    )

    for entry in cfg.get('pipelines', []):
        name = str(entry['name'])
        config_path = str(entry['config'])
        try:
            pipeline_cfg = load_yaml(config_path)
            state = supervisor_step(pipeline_cfg, dry_run=dry_run)
            pipeline_states.append(
                {
                    'name': name,
                    'config': config_path,
                    'active_stage': state.get('active_stage', ''),
                    'running_project_jobs': int(state.get('running_project_jobs', 0)),
                    'gpu_idle': bool(state.get('gpu_idle', False)),
                    'launches': list(state.get('launches', [])),
                    'backfill_attempted': bool(state.get('backfill_attempted', False)),
                    'backfill_launched': bool(state.get('backfill_launched', False)),
                    'backfill_output': str(state.get('backfill_output', '')),
                    'timestamp_utc': str(state.get('timestamp_utc', '')),
                }
            )
        except Exception as exc:
            message = f'{name} :: {config_path} :: {exc}'
            errors.append(message)
            append_log(watchdog_log, f'tick error :: {message}')

    gpu_after = query_gpu()
    state = {
        'timestamp_utc': timestamp(),
        'gpu_before': gpu_before,
        'gpu_after': gpu_after,
        'pipelines': pipeline_states,
        'errors': errors,
    }
    save_json(state_path, state)
    gpu_after_util = gpu_after['utilization_gpu']
    gpu_after_mem_used = gpu_after['memory_used_mib']
    gpu_after_mem_total = gpu_after['memory_total_mib']
    append_log(
        watchdog_log,
        f'tick end :: gpu util={gpu_after_util} mem={gpu_after_mem_used}/{gpu_after_mem_total} MiB errors={len(errors)}',
    )
    return state


def main() -> None:
    args = parse_args()
    while True:
        cfg = load_yaml(args.config)
        poll_seconds = int(dict(cfg['runtime']).get('poll_seconds', 60))
        watchdog_step(cfg, dry_run=bool(args.dry_run))
        if args.once:
            break
        time.sleep(poll_seconds)


if __name__ == '__main__':
    main()
