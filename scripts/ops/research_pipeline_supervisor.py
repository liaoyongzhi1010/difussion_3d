from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

STATE_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run staged research tasks continuously and keep the GPU pipeline moving.')
    parser.add_argument('--config', default='configs/pipeline/single_view_scene.yaml')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f'yaml config at {path} must load as dict')
    return data


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f'json at {path} must load as dict')
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def timestamp() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f'[{timestamp()}] {message}'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def query_gpu() -> dict[str, int]:
    proc = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    util_str, mem_used_str, mem_total_str = [item.strip() for item in proc.stdout.strip().splitlines()[0].split(',')]
    return {
        'utilization_gpu': int(util_str),
        'memory_used_mib': int(mem_used_str),
        'memory_total_mib': int(mem_total_str),
    }


def list_processes() -> list[tuple[int, str]]:
    proc = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True, check=True)
    rows: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, args = stripped.split(maxsplit=1)
        rows.append((int(pid_str), args))
    return rows


def get_nested_field(payload: dict[str, Any], field: str) -> Any:
    value: Any = payload
    for part in field.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def is_task_complete(task: dict[str, Any]) -> bool:
    completion = dict(task.get('completion', {}))
    ctype = str(completion.get('type', ''))
    if ctype == 'summary_field_gte':
        fallback_dir = str(completion.get('dir', '')).strip()
        if fallback_dir:
            fallback_count = sum(1 for _ in Path(fallback_dir).glob(str(completion.get('glob', '*'))))
            if fallback_count >= int(completion['threshold']):
                return True
        payload = load_json(Path(completion['path']))
        value = get_nested_field(payload, str(completion['field']))
        return value is not None and float(value) >= float(completion['threshold'])
    if ctype == 'file_count_gte':
        count = sum(1 for _ in Path(completion['dir']).glob(str(completion.get('glob', '*'))))
        return count >= int(completion['threshold'])
    if ctype == 'file_exists':
        return Path(completion['path']).exists()
    raise ValueError(f'unsupported completion type: {ctype}')


def find_running_pid(task: dict[str, Any], processes: list[tuple[int, str]]) -> int | None:
    marker = str(task.get('match_substring', '')).strip()
    if not marker:
        return None
    for pid, args in processes:
        if marker in args:
            return pid
    return None


def launch_task(task: dict[str, Any], runtime_cfg: dict[str, Any], *, dry_run: bool, supervisor_log: Path) -> int | None:
    command = [str(part) for part in task['launch_command']]
    task_name = str(task['name'])
    joined_command = ' '.join(command)
    append_log(supervisor_log, f'launching {task_name} :: {joined_command}')
    if dry_run:
        return None
    env = os.environ.copy()
    for key, value in dict(runtime_cfg.get('env', {})).items():
        env[str(key)] = str(value)
    log_path = Path(task['log_path'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('a', encoding='utf-8')
    process = subprocess.Popen(
        command,
        cwd=str(runtime_cfg['workdir']),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    append_log(supervisor_log, f'started {task_name} with pid={process.pid}')
    return process.pid


def supervisor_step(cfg: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    runtime_cfg = dict(cfg['runtime'])
    supervisor_log = Path(runtime_cfg['log_path'])
    state_path = Path(runtime_cfg['state_path'])
    gpu = query_gpu()
    processes = list_processes()

    idle_util_below = int(runtime_cfg.get('idle_gpu_util_below', 10))
    idle_mem_below = int(runtime_cfg.get('idle_memory_used_mib_below', 2048))
    gpu_idle = gpu['utilization_gpu'] <= idle_util_below and gpu['memory_used_mib'] <= idle_mem_below

    stages_state: list[dict[str, Any]] = []
    launches: list[str] = []
    running_project_jobs = 0
    active_stage_name = ''

    for stage in cfg.get('stages', []):
        stage_name = str(stage['name'])
        max_parallel_jobs = int(stage.get('max_parallel_jobs', 1))
        task_states: list[dict[str, Any]] = []
        pending_tasks: list[dict[str, Any]] = []
        running_in_stage = 0
        complete_count = 0

        for task in stage.get('tasks', []):
            if not bool(task.get('enabled', True)):
                continue
            pid = find_running_pid(task, processes)
            complete = is_task_complete(task)
            if pid is not None:
                running_project_jobs += 1
                running_in_stage += 1
            if complete:
                complete_count += 1
            if not complete and pid is None:
                pending_tasks.append(task)
            task_states.append(
                {
                    'name': task['name'],
                    'pid': pid,
                    'complete': complete,
                    'log_path': str(task.get('log_path', '')),
                    'match_substring': str(task.get('match_substring', '')),
                }
            )

        stage_complete = complete_count == len(task_states) if task_states else True
        stages_state.append(
            {
                'name': stage_name,
                'complete': stage_complete,
                'running_jobs': running_in_stage,
                'tasks': task_states,
            }
        )

        if stage_complete:
            continue

        active_stage_name = stage_name
        allow_launch = running_project_jobs > 0 or gpu_idle
        available_slots = max(0, max_parallel_jobs - running_in_stage)
        if allow_launch and available_slots > 0:
            for task in pending_tasks[:available_slots]:
                launch_task(task, runtime_cfg, dry_run=dry_run, supervisor_log=supervisor_log)
                launches.append(str(task['name']))
        break

    backfill_attempted = False
    backfill_launched = False
    backfill_returncode: int | None = None
    backfill_output = ''
    backfill_command = runtime_cfg.get('backfill_command')
    if not active_stage_name and running_project_jobs == 0 and gpu_idle and backfill_command:
        joined_backfill = ' '.join(str(part) for part in backfill_command)
        append_log(supervisor_log, f'backfill :: {joined_backfill}')
        backfill_attempted = True
        if not dry_run:
            env = os.environ.copy()
            for key, value in dict(runtime_cfg.get('env', {})).items():
                env[str(key)] = str(value)
            result = subprocess.run(
                [str(part) for part in backfill_command],
                cwd=str(runtime_cfg['workdir']),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            backfill_returncode = int(result.returncode)
            backfill_output = (result.stdout or '').strip()
            if backfill_output:
                append_log(supervisor_log, f'backfill output :: {backfill_output}')
            stderr_output = (result.stderr or '').strip()
            if stderr_output:
                append_log(supervisor_log, f'backfill stderr :: {stderr_output}')
            backfill_launched = 'launched ' in backfill_output
        else:
            backfill_launched = True

    state = {
        'state_version': STATE_VERSION,
        'timestamp_utc': timestamp(),
        'gpu': gpu,
        'gpu_idle': gpu_idle,
        'running_project_jobs': running_project_jobs,
        'active_stage': active_stage_name,
        'launches': launches,
        'backfill_attempted': backfill_attempted,
        'backfill_launched': backfill_launched,
        'backfill_returncode': backfill_returncode,
        'backfill_output': backfill_output,
        'stages': stages_state,
    }
    save_json(state_path, state)
    launch_text = ','.join(launches) if launches else 'none'
    stage_text = active_stage_name or 'none'
    backfill_text = 'yes' if backfill_attempted else 'no'
    gpu_util = gpu['utilization_gpu']
    gpu_mem_used = gpu['memory_used_mib']
    gpu_mem_total = gpu['memory_total_mib']
    append_log(
        supervisor_log,
        f'gpu util={gpu_util} mem={gpu_mem_used}/{gpu_mem_total} MiB active_stage={stage_text} project_jobs={running_project_jobs} launches={launch_text} backfill={backfill_text}',
    )
    return state


def main() -> None:
    args = parse_args()
    while True:
        cfg = load_yaml(args.config)
        poll_seconds = int(dict(cfg['runtime']).get('poll_seconds', 120))
        supervisor_step(cfg, dry_run=bool(args.dry_run))
        if args.once:
            break
        time.sleep(poll_seconds)


if __name__ == '__main__':
    main()
