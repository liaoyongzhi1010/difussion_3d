# Tasks

Goal-oriented tracker for the top-venue submission. Historical daily logs
from earlier in the project have been archived out of version control; this
file is the forward-looking plan.

## In flight

- [ ] Wire `heads/dit_hidden.DiTHiddenDenoiser` into
  `diffusion/scene_model.SingleViewSceneDiffusion` behind
  `cfg.model.hidden_head.type == "dit"`.
- [ ] Wire `heads/detr_visible.DetrVisibleHead` into the scene model behind
  `cfg.model.visible_head.type == "detr"`; plumb the set loss through
  `compute_losses` for the DETR path.
- [ ] Implement `docs/superpowers/specs/datasets/3dfront.extract.py`'s
  renderer (pyrender/EGL) and produce an initial 3D-FRONT packet set.
- [ ] Implement `docs/superpowers/specs/datasets/scannet.extract.py` and
  produce an initial ScanNet packet set.
- [ ] Launch the v5 smoke training run on pixarmesh (2k steps) and record
  baseline metric numbers via `engine.eval_loop`.

## Queued

- [ ] Three-seed v5 headline run on pixarmesh + 3D-FRONT; produce
  `seeds_summary.json` per dataset.
- [ ] Ablation sweep: `no_dit`, `no_detr`, `no_hidden_diffusion`,
  `no_occlusion_bias` at a shared step budget.
- [ ] Baseline comparisons: `fullscene_diffusion`, `visible_only` at the same
  budget.
- [ ] `scripts/benchmark/make_paper_tables.py` to compile
  `outputs/tables/{main_table,ablation,posterior_sweep}.csv`.
- [ ] InstantMesh geometry-decoder adapter (`geometry/instantmesh_adapter.py`).
- [ ] Add a render-metrics module (`metrics/render_metrics.py`) wrapping the
  existing `scripts/eval/eval_single_view_render_metrics.py` logic.

## Deferred (out of scope for this session)

- Full training to convergence (multi-day on A6000).
- Replacing the geometry VAE with a full LRM.
- Multi-view branches and 3D Gaussian Splatting extensions.

## Done during the April 2026 restructure

- Design spec pinned to
  `docs/superpowers/specs/2026-04-18-top-venue-restructure-design.md`.
- Destructive cleanup: `legacy/`, dead configs, stale `outputs/*` (~500 GB).
- Repo reorganized into `backbones/ heads/ diffusion/ engine/ metrics/
  datasets/ geometry/` with aliases for backward compatibility.
- DiT hidden denoiser and DETR visible head implemented as standalone modules.
- 3D metric suite (Chamfer, F-score, box IoU, hidden recall, collision)
  added and smoke-tested.
- `engine.{train_loop, eval_loop, seed_harness}` entry points added.
- `configs/experiments/{main,ablations,baselines}/` tree in place.
