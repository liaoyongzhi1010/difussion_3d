# Paper Results

Paper-facing results table. Cells marked `—` are pending runs scheduled in
`TASKS.md`.

## Main comparison

Dataset-by-dataset comparison of the v5 headline against baselines. All
metrics reported as mean ± 95% CI over 3 seeds (Student-t, 2-tailed) via
`engine.seed_harness`. Chamfer Distance (CD) is symmetric, in meters;
F-score@τ follows the InstPIFu protocol; hidden-recall (H-Rec) uses
3D-IoU ≥ 0.25 matching.

### pixarmesh (1555 / 179 / 215 room-disjoint split)

| Model                               | CD ↓ (scene) | CD ↓ (hidden) | F@1cm ↑ | F@2cm ↑ | F@5cm ↑ | H-Rec ↑ |
|-------------------------------------|--------------|---------------|---------|---------|---------|---------|
| visible-only baseline               | —            | —             | —       | —       | —       | —       |
| fullscene diffusion baseline        | —            | —             | —       | —       | —       | —       |
| v5 (DiT + DETR + DINOv2-large)      | —            | —             | —       | —       | —       | —       |

### 3D-FRONT (public, EULA-gated)

| Model                               | CD ↓ (scene) | CD ↓ (hidden) | F@1cm ↑ | F@2cm ↑ | F@5cm ↑ | H-Rec ↑ |
|-------------------------------------|--------------|---------------|---------|---------|---------|---------|
| Total3D (re-run)                    | —            | —             | —       | —       | —       | —       |
| InstPIFu (re-run)                   | —            | —             | —       | —       | —       | —       |
| v5 (DiT + DETR + DINOv2-large)      | —            | —             | —       | —       | —       | —       |

### ScanNet (public, token-gated)

| Model                               | CD ↓ (scene) | CD ↓ (hidden) | F@1cm ↑ | F@2cm ↑ | F@5cm ↑ | H-Rec ↑ |
|-------------------------------------|--------------|---------------|---------|---------|---------|---------|
| v5 (DiT + DETR + DINOv2-large)      | —            | —             | —       | —       | —       | —       |

## Ablations (pixarmesh)

Same metric suite, same split, same budget (2k–10k steps depending on
whether we're running smoke or production).

| Ablation                  | CD ↓ (scene) | CD ↓ (hidden) | F@2cm ↑ | H-Rec ↑ |
|---------------------------|--------------|---------------|---------|---------|
| v5 (all on)               | —            | —             | —       | —       |
| − DiT (baseline decoder)  | —            | —             | —       | —       |
| − DETR (fixed K=12)       | —            | —             | —       | —       |
| − hidden diffusion        | —            | —             | —       | —       |
| − occlusion bias          | —            | —             | —       | —       |

## Posterior-sweep analysis

| Method               | steps | posteriors (p) | samples (s) | best hidden MSE ↓ |
|----------------------|-------|----------------|-------------|-------------------|
| v3 (reference)       | 20k   | 5              | 20          | 20.74             |
| v5                   | —     | —              | —           | —                 |

## Physical-plausibility proxies (pixarmesh)

| Model                | Collision ↓ | Support violation ↓ |
|----------------------|-------------|---------------------|
| v5                   | —           | —                   |

## Compute

Runs are executed on a single A6000 (48 GB). The headline v5 config
(`configs/experiments/main/v5_dit_detr_dinov2_large.yaml`) targets bf16 with
`batch_size=4` and gradient accumulation where needed; see each run's
`config.resolved.yaml` for the exact resolved config.

## Reproducibility

Tables above are produced by
`python scripts/benchmark/make_paper_tables.py` (pending) consuming
`outputs/v5_seeds/**/eval/summary.json` plus each baseline's run directory.
Every number should remain reproducible from `{config.resolved.yaml,
latest.pt}`.
