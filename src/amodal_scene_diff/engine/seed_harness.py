"""Multi-seed runner + 95% CI aggregator.

For top-venue tables we run a config N times with different seeds, then
report mean and 95% CI (t-distribution) over val metrics. The harness calls
`engine.train_loop` and `engine.eval_loop` via subprocess so runs stay in
fresh processes and bad seeds cannot contaminate each other.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

# Student-t two-sided 95% critical values for small N (df = N - 1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n <= 1:
        return (float(values[0]) if n == 1 else 0.0, 0.0)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(var / n)
    t = _T95.get(n - 1, 1.96)  # fall back to normal for large N
    return (mean, t * stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=-1)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    per_seed_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_dir = args.output_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            args.python, "-m", "amodal_scene_diff.engine.train_loop",
            "--config", str(args.config),
            "--seed", str(seed),
            "--output-dir", str(seed_dir),
        ]
        if args.train_steps > 0:
            train_cmd += ["--train-steps", str(args.train_steps)]
        _run(train_cmd)

        if args.skip_eval:
            continue

        eval_dir = seed_dir / "eval"
        eval_cmd = [
            args.python, "-m", "amodal_scene_diff.engine.eval_loop",
            "--config", str(args.config),
            "--checkpoint", str(seed_dir / "latest.pt"),
            "--output-dir", str(eval_dir),
            "--num-sampling-steps", str(args.num_sampling_steps),
        ]
        _run(eval_cmd)

        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        summary["seed"] = seed
        per_seed_summaries.append(summary)

    if not per_seed_summaries:
        print(json.dumps({"seeds": args.seeds, "status": "train_only"}))
        return

    # aggregate
    keys = sorted({k for s in per_seed_summaries for k, v in s.items() if isinstance(v, (int, float))})
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(s[key]) for s in per_seed_summaries if key in s]
        mean, ci = _mean_ci95(values)
        aggregated[key] = {"mean": mean, "ci95": ci, "n": len(values)}

    fscore_keys = sorted({k for s in per_seed_summaries if "fscore" in s for k in s["fscore"]})
    aggregated_fscore: dict[str, dict[str, float]] = {}
    for k in fscore_keys:
        values = [float(s["fscore"][k]) for s in per_seed_summaries if "fscore" in s and k in s["fscore"]]
        mean, ci = _mean_ci95(values)
        aggregated_fscore[str(k)] = {"mean": mean, "ci95": ci, "n": len(values)}

    summary = {
        "config": str(args.config),
        "seeds": args.seeds,
        "per_seed": per_seed_summaries,
        "aggregated": aggregated,
        "aggregated_fscore": aggregated_fscore,
    }
    out = args.output_root / "seeds_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
