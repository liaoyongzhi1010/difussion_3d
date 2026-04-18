# Top-Venue Restructure — Design Spec

- Date: 2026-04-18
- Target: CVPR / ICCV / NeurIPS-grade submission of the single-view amodal scene reconstruction line
- Scope owner: @lyz1010

## 1. Motivation

The repo currently supports one working research mainline (`single_view_visible_direct_hidden_diffusion` on DINOv2-base). Internal ablations look clean (visible MSE 1.43, best_hidden_mse 20.74 on p=5, s=20 at `v0625`), but the project does not yet meet the submission bar of a top vision/learning venue:

1. Only one private synthetic dataset (`pixarmesh`, 1949 scenes, 1555/179/215 split). No public benchmark.
2. No comparison to published baselines (Total3D, InstPIFu, MIDI, BlockFusion, etc.).
3. Non-standard metrics: only latent-space MSE + top-down semantic PSNR/SSIM/LPIPS. No Scene Chamfer Distance, F-score@τ, Hidden-Recall, box-IoU.
4. Hidden denoiser is a plain `nn.TransformerDecoder`, not a DiT-class AdaLN-Zero backbone.
5. Fixed 20 slots (`K_VIS=12`, `K_HID=8`), no variable-length DETR-style decoding.
6. Geometry decoder is a compact PointNet + tri-plane, not LRM / InstantMesh class.
7. Single-seed runs, no confidence intervals.
8. `legacy/`, old supervisors, dead configs, stale `TASKS.md` history, 504 GB of obsolete `outputs/` are cluttering the repo.

This spec pins down exactly what this restructure delivers, what it explicitly does not, and what gets pushed to `main`.

## 2. Non-goals

- Downloading license-gated datasets automatically (3D-FRONT EULA, ScanNet license). Provide scripts; user signs and runs.
- Achieving final paper numbers in-session. Training a real mainline run takes days on a single A6000.
- Replacing the geometry VAE decoder with a full LRM (out-of-scope for a single-GPU session; we integrate InstantMesh pretrained weights instead).
- Introducing multi-view / 3D Gaussian Splatting branches.

## 3. What this session delivers

### 3.1 Destructive cleanup (irreversible, user-authorized)

Deleted:
- `legacy/` (entire directory, 184 KB)
- `configs/data/3dfront_v1.yaml`
- `configs/diffusion/visible_locked*.yaml` (9 files: `visible_locked.yaml`, `visible_locked_hiddenfocus.yaml`, `visible_locked_hiddenonly.yaml`, `visible_locked_occbias_v025/0375/050/0625.yaml`, `visible_locked_tradeoff_v025/050.yaml`)
- `configs/diffusion/single_view_visible_direct_hidden_diffusion_v1.yaml`, `..._v2_heavy.yaml` (superseded by v3/v4; v3 is current, v4 is the upgrade target)
- `outputs/real_data/pixarmesh_bootstrap_visiblelocked_*`, `outputs/real_data/pixarmesh_fullscene_control_*` (~500 GB of obsolete checkpoints)
- `outputs/pipeline_*`, `outputs/gpu_monitor*`, `outputs/visible_locked_compare_supervisor*` (stale logs / state files)
- `scripts/eval/run_4shard_val_scan_loop.sh` (old 4-shard harness)
- `scripts/ops/backfill_visible_locked_compare.py`, `scripts/ops/single_view_pipeline_board.py` if unused after refactor; keep `paper_mainline_harness.py` and `monitor_single_view_training.py`.
- Rewrite `TASKS.md` from scratch (the multi-thousand-line dated history goes; new content is goal-oriented for the paper).
- Rewrite `docs/paper_results.md` into a paper-results table template (Benchmark × Metric × Model).

Kept:
- `outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm` (real packet data and current best checkpoint family)
- `docs/dataset_layout.md`, `docs/reproducibility.md`, `docs/quickstart.md`
- `docs/formal_paper_mainline.md`, `docs/repo_audit.md` (reference background; may be refreshed)

### 3.2 New directory layout (`src/amodal_scene_diff/`)

```
src/amodal_scene_diff/
├── backbones/
│   ├── __init__.py
│   ├── patch_vit.py              # moved from models/diffusion/
│   ├── dinov2.py                 # moved from models/diffusion/observation_backbones.py
│   └── dinov2_hybrid.py          # moved from models/diffusion/observation_backbones.py
├── heads/
│   ├── __init__.py
│   ├── detr_visible.py           # NEW: variable-slot DETR-style visible head with Hungarian matcher
│   ├── dit_hidden.py             # NEW: DiT-class AdaLN-Zero hidden denoiser
│   ├── layout.py                 # layout regression head
│   └── relation.py               # floor/wall/support heads
├── diffusion/
│   ├── __init__.py
│   ├── scheduler.py              # beta schedule + q_sample + prediction_type handling
│   ├── sampler.py                # DDIM / DPM-Solver-2 sampling loop
│   └── model.py                  # SingleViewReconstructionDiffusion assembler
├── geometry/
│   ├── geometry_vae.py           # kept as-is for now
│   └── instantmesh_adapter.py    # NEW: optional pretrained geometry decoder adapter
├── datasets/
│   ├── __init__.py
│   ├── pixarmesh.py              # renamed from single_view.py; packet-format loader
│   ├── threedfront.py            # NEW: 3D-FRONT loader (requires user download)
│   └── scannet.py                # NEW: ScanNet loader stub
├── metrics/
│   ├── __init__.py
│   ├── chamfer.py                # NEW: symmetric & one-sided Chamfer at mesh/point-cloud level
│   ├── fscore.py                 # NEW: F-score@τ with τ ∈ {1cm, 2cm, 5cm}
│   ├── box_iou.py                # NEW: oriented 3D IoU for box-level metrics
│   ├── hidden_recall.py          # NEW: hidden-region recall / precision / F1
│   ├── collision.py              # NEW: collision / support violation rates
│   └── render_metrics.py         # existing PSNR/SSIM/LPIPS on top-down semantic (kept)
├── engine/
│   ├── __init__.py
│   ├── train_loop.py             # extracted from scripts/train/train_single_view_scene.py
│   ├── eval_loop.py              # shared state-space + render evaluator
│   └── seed_harness.py           # NEW: 3-seed runner + 95% CI reporter
└── structures/
    ├── single_view_batch.py
    └── scene_state.py            # explicit scene state dataclass (shared with exporter)
```

### 3.3 Configs reorganized

```
configs/
├── data/
│   ├── pixarmesh.yaml            # current real packet main
│   ├── threedfront.yaml          # NEW
│   └── scannet.yaml              # NEW (stub)
├── backbones/
│   ├── dinov2_base_frozen.yaml
│   ├── dinov2_large_hybrid.yaml
│   └── patch_vit.yaml
├── experiments/
│   ├── main/
│   │   └── v5_dit_detr_dinov2_large.yaml    # NEW headline config
│   ├── ablations/
│   │   ├── no_detr.yaml
│   │   ├── no_dit.yaml
│   │   ├── no_occlusion_bias.yaml
│   │   └── no_hidden_diffusion.yaml
│   └── baselines/
│       ├── fullscene_diffusion.yaml
│       └── visible_only.yaml
├── geometry_vae/                 # existing
└── runtime/                      # existing
```

### 3.4 Scripts

```
scripts/
├── data/
│   ├── download_3dfront.py       # NEW: prints EULA URL + fetches with user token
│   ├── download_scannet.py       # NEW: same pattern
│   └── restore_packet_rgb_paths_from_parquet.py  # existing
├── preprocess/
│   └── (existing)
├── train/
│   └── train_single_view_scene.py          # thinned; delegates to engine/
├── eval/
│   ├── eval_single_view_scene.py           # existing state eval
│   ├── eval_single_view_render_metrics.py  # existing render eval
│   ├── eval_3d_metrics.py                  # NEW: CD / F-score / IoU / hidden-recall
│   └── export_single_view_scene_repr.py    # existing
└── benchmark/
    ├── run_seeds.py              # NEW: multi-seed driver
    └── make_paper_tables.py      # NEW: compiles CSV tables from eval outputs
```

### 3.5 Dataset download staging

Per user instruction, dataset download artifacts live under `docs/superpowers/specs/`:

```
docs/superpowers/specs/
├── 2026-04-18-top-venue-restructure-design.md   # this doc
├── datasets/
│   ├── README.md                  # NEW: walks through 3D-FRONT / ScanNet EULA flow
│   ├── 3dfront.download.sh        # NEW
│   ├── 3dfront.extract.py         # NEW
│   ├── scannet.download.sh        # NEW (requires ScanNet token)
│   └── manifest.yaml              # NEW: expected folder layout and checksums
```

The downloaded raw archives land under `data/external/{3d_front, scannet}/` (symlinked from the download scripts, not inside `docs/`). Rationale: keeping 100s of GB of binary assets out of `docs/` preserves repo hygiene; only the *scripts and manifests* live in `docs/superpowers/specs/datasets/`.

### 3.6 Top-venue code additions (details)

- **`heads/dit_hidden.py`**
  - AdaLN-Zero conditioning (time + observation tokens)
  - Sinusoidal positional embedding + learned slot embedding
  - RMSNorm, GELU-Tanh, zero-initialized output projection
  - 12 blocks / 1024 d-model at `large`, 8 blocks / 768 d-model at `base`

- **`heads/detr_visible.py`**
  - 50 learnable object queries (vs. current fixed 12)
  - Hungarian matcher over (class, box center IoU, latent-code cosine)
  - Set-loss (classification + L1 box + GIoU + latent MSE)

- **`metrics/chamfer.py`**
  - Per-object sampled point clouds from predicted latent → mesh decode
  - Symmetric Chamfer Distance in room frame
  - Separate reporting: scene-level, visible-only, hidden-only

- **`metrics/fscore.py`**
  - F-score@{0.01, 0.02, 0.05} m thresholds
  - Matches InstPIFu / Total3D evaluation protocol

- **`metrics/hidden_recall.py`**
  - Object-level recall of hidden instances (pred ↔ GT matched by 3D IoU > 0.25)
  - Hidden-region F1

- **`engine/seed_harness.py`**
  - Runs N seeds in sequence on same config
  - Aggregates mean ± 95% CI
  - Emits `seeds_summary.json`

- **`scripts/benchmark/make_paper_tables.py`**
  - Reads eval outputs from `outputs/<exp>/<seed>/`
  - Writes `outputs/tables/main_table.csv`, `ablation.csv`, `posterior_sweep.csv`

## 4. Push strategy

Direct commits to `main`, authorized by the user in this session. Commits are semantic so the diff history remains bisectable:

1. `chore: remove legacy/, dead configs, stale outputs`
2. `refactor: reorganize src/amodal_scene_diff into backbones/heads/diffusion/engine/metrics`
3. `feat(metrics): add 3d chamfer, f-score, hidden-recall, collision metrics`
4. `feat(heads): add DiT-class hidden denoiser and DETR-style visible head`
5. `feat(data): add 3D-FRONT and ScanNet loaders with download scripts`
6. `feat(engine): add seed harness and paper-table builder`
7. `feat(configs): reorganize experiments into main/ablations/baselines`
8. `docs: rewrite README, TASKS, paper_results for top-venue layout`
9. `chore: launch v5 mainline smoke training on pixarmesh`

All commits signed `Co-Authored-By: Claude Opus 4.7 (1M context)`.

## 5. Execution order in this session

1. Pre-flight snapshot: list exact deletions with sizes, pause only if the user has said "wait"; otherwise proceed (user authorized).
2. Delete obsolete `outputs/` first (frees disk for downloads and new runs).
3. Delete `legacy/`, obsolete configs, obsolete scripts.
4. Reorganize `src/` via moves + import rewrites; run `pytest`-level smoke (if tests exist) or a 2-step dummy training to confirm imports.
5. Add new modules (`metrics/`, `heads/dit_hidden.py`, `heads/detr_visible.py`, `engine/seed_harness.py`).
6. Add dataset scaffolding (`datasets/threedfront.py`, `datasets/scannet.py`, download scripts under `docs/superpowers/specs/datasets/`).
7. Reorganize `configs/`.
8. Rewrite `README.md`, `TASKS.md`, `docs/paper_results.md`.
9. Commit in the order above, push after each group.
10. Launch `v5_dit_detr_dinov2_large` smoke training (2000 steps) on pixarmesh in the background. Longer training is the user's call.

## 6. Acceptance criteria (what "done" means for this session)

- `main` on `origin` contains the new tree.
- `python -m amodal_scene_diff.engine.train_loop --config configs/experiments/main/v5_dit_detr_dinov2_large.yaml --train-steps 2` runs end-to-end without import or shape error.
- `scripts/eval/eval_3d_metrics.py --help` prints usage.
- `python scripts/benchmark/run_seeds.py --help` prints usage.
- `docs/superpowers/specs/datasets/README.md` documents EULA + download steps.
- One background smoke training job is running and logging to `outputs/real_data/<v5_root>/logs/`.

## 7. Explicit out-of-session work

- User must sign 3D-FRONT EULA at <https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future> and drop the token into `scripts/data/download_3dfront.py` config.
- User must obtain a ScanNet access token from Stanford.
- Full v5 training to convergence (estimated 2–3 days on A6000).
- Final benchmark table production once both public datasets are in place.

## 8. Risks

- Mass `outputs/` deletion cannot be undone. User authorized.
- DiT / DETR additions may surface subtle distribution shift vs. current pretrained checkpoints; v3 and v4 checkpoints remain valid but will require a compatibility shim if loaded into the new v5 architecture. Decision: do not attempt cross-arch resume; v5 is a new training line.
- InstantMesh pretrained-weight integration depends on license availability at runtime; if blocked, v5 still trains with the existing geometry VAE decoder.

## 9. Self-review notes

- All section requirements are concrete (file paths, module names, commit subjects).
- No TBD / TODO placeholders.
- Scope is one coherent refactor + one smoke run; larger training is explicitly deferred.
- No contradictions between sections after re-read.
