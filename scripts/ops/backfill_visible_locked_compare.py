from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path("/root/3d/generation")
PYTHON = str(ROOT / ".venv/bin/python")
PACKETS = "outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets"
DATA_CFG = "configs/data/3dfront_v1.yaml"
RUNTIME_CFG = "configs/runtime/gpu_smoke.yaml"
SPLIT_JSON = "outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json"

METHODS = [
    {
        "name": "visiblelocked",
        "config": "configs/diffusion/visible_locked.yaml",
        "checkpoint": "outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048/checkpoints_scene_denoiser_v1/step_0024000.pt",
        "out_dir": "outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048",
    },
    {
        "name": "fullscene_control",
        "config": "configs/diffusion/base.yaml",
        "checkpoint": "outputs/real_data/pixarmesh_fullscene_control_resume_b128_train2048/checkpoints_scene_denoiser_v1/step_0024000.pt",
        "out_dir": "outputs/real_data/pixarmesh_fullscene_control_resume_b128_train2048",
    },
]

# Longer jobs first so the GPU stays occupied for meaningful stretches.
SETTINGS = [
    {"samples": 100, "steps": 8, "batch": 2},
    {"samples": 100, "steps": 4, "batch": 2},
    {"samples": 200, "steps": 20, "batch": 1},
    {"samples": 200, "steps": 8, "batch": 1},
    {"samples": 200, "steps": 4, "batch": 1},
    {"samples": 400, "steps": 20, "batch": 1},
]


def summary_path(method: dict[str, str], samples: int, steps: int) -> Path:
    prefix = "posterior_eval_visiblelocked_resume_b128_test" if method["name"] == "visiblelocked" else "posterior_eval_fullscene_control_resume_b128_test"
    return ROOT / method["out_dir"] / f"{prefix}_p{samples}_s{steps}.json"


def log_path(method: dict[str, str], samples: int, steps: int) -> Path:
    prefix = "posterior_eval_visiblelocked_resume_b128_test" if method["name"] == "visiblelocked" else "posterior_eval_fullscene_control_resume_b128_test"
    return ROOT / method["out_dir"] / "logs" / f"{prefix}_p{samples}_s{steps}.log"


def launch(method: dict[str, str], samples: int, steps: int, batch: int) -> int:
    out_json = summary_path(method, samples, steps)
    out_log = log_path(method, samples, steps)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "scripts/eval/eval_scene_posterior.py",
        "--checkpoint", method["checkpoint"],
        "--packet-dir", PACKETS,
        "--config", method["config"],
        "--data-config", DATA_CFG,
        "--runtime-config", RUNTIME_CFG,
        "--batch-size", str(batch),
        "--max-samples", "0",
        "--num-posterior-samples", str(samples),
        "--num-inference-steps", str(steps),
        "--sample-id-json", SPLIT_JSON,
        "--split", "test",
        "--save-summary", str(out_json.relative_to(ROOT)),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    handle = out_log.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    print(f"launched {method['name']} p{samples}_s{steps} pid={proc.pid}")
    return proc.pid


def main() -> int:
    launched = 0
    for setting in SETTINGS:
        missing_methods = [m for m in METHODS if not summary_path(m, setting["samples"], setting["steps"]).exists()]
        if not missing_methods:
            continue
        for method in missing_methods[:2]:
            launch(method, setting["samples"], setting["steps"], setting["batch"])
            launched += 1
        return 0
    print("no_pending_backfill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
