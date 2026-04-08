from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalSpec:
    label: str
    path: str
    family: str
    notes: str = ""


@dataclass(frozen=True)
class SelectorSpec:
    label: str
    path: str
    notes: str = ""


MAIN_SPECS = [
    EvalSpec(
        label="Full-scene diffusion",
        path="outputs/real_data/pixarmesh_fullscene_control_resume_b128_train2048/posterior_eval_fullscene_control_resume_b128_test_p5_s20.json",
        family="main",
        notes="No visible-locking; control baseline.",
    ),
    EvalSpec(
        label="Visible-locked resume",
        path="outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048/posterior_eval_visiblelocked_resume_b128_test_p5_s20.json",
        family="main",
        notes="Deterministic visible baseline before attention / loss tuning.",
    ),
    EvalSpec(
        label="Tradeoff v0.25",
        path="outputs/real_data/pixarmesh_bootstrap_visiblelocked_tradeoff_v025_ft_b128_train2048/posterior_eval_visiblelocked_tradeoff_v025_ft_b128_test_p5_s20.json",
        family="main",
        notes="Improves hidden completion but damages visible reconstruction.",
    ),
    EvalSpec(
        label="OccBias v0.50",
        path="outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v050_ft_b128_test_p5_s20.json",
        family="main",
        notes="Previous best-balanced visible / hidden tradeoff.",
    ),
    EvalSpec(
        label="OccBias v0.625",
        path="outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v0625_ft_b128_test_p5_s20.json",
        family="main",
        notes="Current canonical mainline.",
    ),
]

ABLATION_SPECS = [
    EvalSpec("Hidden-focus", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_hiddenfocus_b128_train2048/posterior_eval_visiblelocked_hiddenfocus_b128_test_p5_s20.json", "ablation"),
    EvalSpec("Hidden-only", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_hiddenonly_ft_b128_train2048/posterior_eval_visiblelocked_hiddenonly_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("OccBias v0.25", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v025_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v025_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("OccBias v0.375", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0375_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v0375_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("OccBias v0.50", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v050_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("OccBias v0.625", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v0625_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("Tradeoff v0.50", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_tradeoff_v050_ft_b128_train2048/posterior_eval_visiblelocked_tradeoff_v050_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("Tradeoff v0.25", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_tradeoff_v025_ft_b128_train2048/posterior_eval_visiblelocked_tradeoff_v025_ft_b128_test_p5_s20.json", "ablation"),
    EvalSpec("Full-scene diffusion", "outputs/real_data/pixarmesh_fullscene_control_resume_b128_train2048/posterior_eval_fullscene_control_resume_b128_test_p5_s20.json", "ablation"),
]

POSTERIOR_SWEEP_SPECS = [
    EvalSpec("Visible-locked resume p5", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048/posterior_eval_visiblelocked_resume_b128_test_p5_s20.json", "posterior_sweep"),
    EvalSpec("Visible-locked resume p20", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048/posterior_eval_visiblelocked_resume_b128_test_p20_s20.json", "posterior_sweep"),
    EvalSpec("Visible-locked resume p50", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048/posterior_eval_visiblelocked_resume_b128_test_p50_s20.json", "posterior_sweep"),
    EvalSpec("Visible-locked resume p100", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_resume_b128_train2048/posterior_eval_visiblelocked_resume_b128_test_p100_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.50 p5", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v050_ft_b128_test_p5_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.50 p20", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v050_ft_b128_test_p20_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.50 p50", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v050_ft_b128_test_p50_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.50 p100", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v050_ft_b128_test_p100_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.625 p5", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v0625_ft_b128_test_p5_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.625 p20", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v0625_ft_b128_test_p20_s20.json", "posterior_sweep"),
    EvalSpec("OccBias v0.625 p50", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/posterior_eval_visiblelocked_occbias_v0625_ft_b128_test_p50_s20.json", "posterior_sweep"),
]

SELECTOR_SPECS = [
    SelectorSpec("Selector small b4/p8", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_selector_b4_p8_train/selector_test_best_summary.json", "Best practical reranker on top of v0.50 generator."),
    SelectorSpec("Selector heavy b8/p16", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_selector_b8_p16_long_train/selector_test_heavy_best_summary.json", "Higher-capacity selector; underperforms despite more parameters."),
    SelectorSpec("Selector wide h256/d4", "outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_selector_wide_b8_p16_h256_d4_t15_train/selector_test_wide_summary.json", "Capacity sweep confirms generator quality is the bottleneck."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing tables and markdown summary from experiment JSON files.")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--summary-md", default="docs/paper_results.md")
    parser.add_argument("--manifest-path", default="outputs/tables/paper_report_manifest.json")
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any] | None:
    file_path = ROOT / Path(path)
    if not file_path.exists():
        return None
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict at {file_path}, got {type(payload)!r}")
    return payload


def _safe_round(value: Any) -> float | str:
    if value is None:
        return ""
    return round(float(value), 4)


def _build_eval_row(spec: EvalSpec) -> dict[str, Any] | None:
    payload = _load_json(spec.path)
    if payload is None:
        return None
    metrics = dict(payload.get("metrics", {}))
    joint = dict(dict(payload.get("selector_metrics", {})).get("joint_confidence", {}))
    return {
        "label": spec.label,
        "family": spec.family,
        "path": spec.path,
        "notes": spec.notes,
        "checkpoint_step": int(payload.get("checkpoint_step", 0)),
        "posterior_samples": int(payload.get("num_posterior_samples", 0)),
        "inference_steps": int(payload.get("num_inference_steps", 0)),
        "num_eval_scenes": int(payload.get("num_eval_scenes", 0)),
        "visible_mse": _safe_round(metrics.get("visible_mse")),
        "hidden_mse": _safe_round(metrics.get("hidden_mse")),
        "best_hidden_mse": _safe_round(metrics.get("best_hidden_mse")),
        "hidden_diversity": _safe_round(metrics.get("hidden_diversity")),
        "visible_diversity": _safe_round(metrics.get("visible_diversity")),
        "hidden_exist_acc": _safe_round(metrics.get("hidden_exist_acc")),
        "hidden_cls_acc": _safe_round(metrics.get("hidden_cls_acc")),
        "hidden_exist_brier": _safe_round(metrics.get("hidden_exist_brier")),
        "joint_hidden_mse": _safe_round(joint.get("hidden_mse")),
        "joint_gap_closed": _safe_round(joint.get("oracle_hidden_gap_closed")),
    }


def _build_selector_row(spec: SelectorSpec) -> dict[str, Any] | None:
    payload = _load_json(spec.path)
    if payload is None:
        return None
    metrics = dict(payload.get("test_metrics", {}))
    return {
        "label": spec.label,
        "path": spec.path,
        "notes": spec.notes,
        "generator_step": int(payload.get("generator_step", 0)),
        "selector_num_parameters": int(payload.get("selector_num_parameters", 0)),
        "eval_num_posterior_samples": int(payload.get("eval_num_posterior_samples", 0)),
        "selected_hidden_mse": _safe_round(metrics.get("selected_hidden_mse")),
        "mean_hidden_mse": _safe_round(metrics.get("mean_hidden_mse")),
        "best_hidden_mse": _safe_round(metrics.get("best_hidden_mse")),
        "selected_visible_mse": _safe_round(metrics.get("selected_visible_mse")),
        "selected_layout_mse": _safe_round(metrics.get("selected_layout_mse")),
        "oracle_hidden_gap_closed": _safe_round(metrics.get("oracle_hidden_gap_closed")),
        "oracle_top1_acc": _safe_round(metrics.get("oracle_top1_acc")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: str) -> str:
    return str(Path(path))


def _format_delta(current: float | str, previous: float | str) -> str:
    if current == "" or previous == "":
        return "n/a"
    delta = float(current) - float(previous)
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"


def _build_summary_markdown(
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    main_map = {row["label"]: row for row in main_rows}
    current = main_map.get("OccBias v0.625")
    previous = main_map.get("OccBias v0.50")
    selector_best = selector_rows[0] if selector_rows else None

    lines: list[str] = []
    lines.append("# Paper Results")
    lines.append("")
    lines.append(f"Last updated: {today} (UTC)")
    lines.append("")
    lines.append("## Canonical Mainline")
    lines.append("")
    lines.append("- config: `configs/diffusion/visible_locked_occbias_v0625.yaml`")
    lines.append("- checkpoint: `outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0034000.pt`")
    lines.append("- principle: visible reconstruction stays deterministic while hidden structure remains diffusion-sampled")
    lines.append("")
    if current is not None:
        lines.append("## Current Main Result")
        lines.append("")
        lines.append(f"- visible_mse: `{current['visible_mse']}`")
        lines.append(f"- hidden_mse: `{current['hidden_mse']}`")
        lines.append(f"- best_hidden_mse: `{current['best_hidden_mse']}`")
        lines.append(f"- hidden_diversity: `{current['hidden_diversity']}`")
        lines.append(f"- joint_confidence_hidden_mse: `{current['joint_hidden_mse']}`")
        lines.append("")
    if current is not None and previous is not None:
        lines.append("## Delta Vs Previous Mainline")
        lines.append("")
        lines.append(f"- visible_mse delta vs v0.50: `{_format_delta(current['visible_mse'], previous['visible_mse'])}`")
        lines.append(f"- hidden_mse delta vs v0.50: `{_format_delta(current['hidden_mse'], previous['hidden_mse'])}`")
        lines.append(f"- best_hidden_mse delta vs v0.50: `{_format_delta(current['best_hidden_mse'], previous['best_hidden_mse'])}`")
        lines.append("")
    lines.append("## Generator Ablations")
    lines.append("")
    for row in ablation_rows:
        lines.append(
            f"- {row['label']}: visible_mse `{row['visible_mse']}`, hidden_mse `{row['hidden_mse']}`, best_hidden_mse `{row['best_hidden_mse']}`"
        )
    lines.append("")
    if sweep_rows:
        lines.append("## Posterior Sweep")
        lines.append("")
        for row in sweep_rows:
            lines.append(
                f"- {row['label']}: p=`{row['posterior_samples']}`, hidden_mse `{row['hidden_mse']}`, best_hidden_mse `{row['best_hidden_mse']}`, joint_hidden_mse `{row['joint_hidden_mse']}`"
            )
        lines.append("")
    if selector_best is not None:
        lines.append("## Selector Status")
        lines.append("")
        lines.append(
            f"- best lightweight selector remains `{selector_best['label']}` with selected_hidden_mse `{selector_best['selected_hidden_mse']}` on top of v0.50"
        )
        lines.append("- conclusion: selector capacity is not the limiting factor; generator posterior quality is")
        lines.append("")
    lines.append("## Generated Tables")
    lines.append("")
    lines.append("- `outputs/tables/table_c_posterior.csv`")
    lines.append("- `outputs/tables/table_d_ablation.csv`")
    lines.append("- `outputs/tables/table_f_baseline_tracker.csv`")
    lines.append("- `outputs/tables/table_g_selector.csv`")
    lines.append("")
    lines.append("## Example Figures")
    lines.append("")
    lines.append("- `examples/figures/visible_locked_occbias_v0625_main/contact_sheet.png`")
    lines.append("- `examples/figures/visible_locked_occbias_v0625_main/*.png`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    output_dir = ROOT / Path(args.output_dir)
    summary_md_path = ROOT / Path(args.summary_md)
    manifest_path = ROOT / Path(args.manifest_path)

    main_rows = [row for row in (_build_eval_row(spec) for spec in MAIN_SPECS) if row is not None]
    ablation_rows = [row for row in (_build_eval_row(spec) for spec in ABLATION_SPECS) if row is not None]
    sweep_rows = [row for row in (_build_eval_row(spec) for spec in POSTERIOR_SWEEP_SPECS) if row is not None]
    selector_rows = [row for row in (_build_selector_row(spec) for spec in SELECTOR_SPECS) if row is not None]

    _write_csv(output_dir / "table_c_posterior.csv", sweep_rows)
    _write_csv(output_dir / "table_d_ablation.csv", ablation_rows)
    _write_csv(output_dir / "table_f_baseline_tracker.csv", main_rows)
    _write_csv(output_dir / "table_g_selector.csv", selector_rows)

    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text(
        _build_summary_markdown(main_rows, ablation_rows, sweep_rows, selector_rows),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "summary_md": _relative(args.summary_md),
        "tables": {
            "posterior": _relative(str(Path(args.output_dir) / "table_c_posterior.csv")),
            "ablation": _relative(str(Path(args.output_dir) / "table_d_ablation.csv")),
            "main": _relative(str(Path(args.output_dir) / "table_f_baseline_tracker.csv")),
            "selector": _relative(str(Path(args.output_dir) / "table_g_selector.csv")),
        },
        "main_rows": main_rows,
        "ablation_rows": ablation_rows,
        "sweep_rows": sweep_rows,
        "selector_rows": selector_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
