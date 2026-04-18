# Amodal Scene Diffusion

This repository is now organized around one paper mainline only:

- input: single real RGB view plus depth-derived observation channels
- visible content: direct 3D reconstruction
- hidden content: conditional diffusion completion
- output: explicit scene-state codes that can be exported to 3D tri-plane representations

Old `visible_locked / fullscene_control / selector` branches are no longer first-class entry points. They have been moved under [`legacy/`](/root/3d/generation/legacy) so the root repository reflects the actual paper path.

## Mainline Status

Current mainline data root:
- `outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm`

Current room-level split:
- train: `1555`
- val: `179`
- test: `215`
- room overlap across splits: `0`

Current strongest working checkpoint family:
- config: [`configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml`](/root/3d/generation/configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml)
- backbone: official `facebook/dinov2-base`
- method: `visible direct reconstruction + hidden diffusion`

## Repository Layout

```text
.
├── configs/
│   ├── data/                # dataset roots and split metadata
│   ├── diffusion/           # single-view paper configs
│   ├── geometry_vae/
│   └── runtime/
├── docs/                    # quickstart, reproducibility, audit notes
├── examples/                # canonical train / eval / export commands
├── legacy/                  # archived older baselines and utilities
├── scripts/
│   ├── preprocess/
│   ├── train/
│   ├── eval/
│   └── ops/
├── src/
│   └── amodal_scene_diff/
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## What Is Paper-Grade Already

- canonical `paper_mainline_harness` now guards the route before train / eval / export

- `src/` package layout with installable code
- room-level train/val/test split generation with disjoint rooms
- real RGB restored into the packet pipeline
- official DINOv2 observation backbone in the current best mainline
- state-space evaluation for single-view reconstruction
- render-space evaluation script for `PSNR / SSIM / LPIPS` on top-down semantic renders
- explicit 3D export from predicted scene states to tri-plane scene representations

## What Still Needs Upgrading

These are the current weak points relative to top-conference-strength open codebases:

- visible reconstruction head is still a custom transformer-set decoder, not yet a stronger DETR-style or geometry-native decoder stack
- hidden diffusion branch is still an in-repo transformer denoiser, not yet a DiT-class latent diffusion backbone
- geometry VAE is still a compact PointNet plus tri-plane decoder, not yet an LRM / InstantMesh-class geometry decoder
- single-view training still needs a cleaner built-in validation hook instead of relying only on standalone eval scripts

The detailed audit is recorded in [`docs/repo_audit.md`](/root/3d/generation/docs/repo_audit.md).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Canonical Commands

Run the route harness before large experiments:

```bash
make harness-single-view
```

Train the mainline:

```bash
make train-single-view
```

Evaluate state metrics on the test split:

```bash
make eval-single-view-state
```

Evaluate render metrics on the test split:

```bash
make eval-single-view-render
```

Run the unified stage board:

```bash
make pipeline-board
```

Export explicit 3D scene representations:

```bash
make export-single-view-repr
```

Restore packet RGB paths if packet metadata is incomplete:

```bash
make restore-rgb-paths
```

## Examples

Canonical runnable examples live in:

- [`examples/train_single_view_main.sh`](/root/3d/generation/examples/train_single_view_main.sh)
- [`examples/eval_single_view_state.sh`](/root/3d/generation/examples/eval_single_view_state.sh)
- [`examples/eval_single_view_render.sh`](/root/3d/generation/examples/eval_single_view_render.sh)
- [`examples/export_single_view_repr.sh`](/root/3d/generation/examples/export_single_view_repr.sh)

## Documentation

- [`docs/quickstart.md`](/root/3d/generation/docs/quickstart.md)
- [`docs/dataset_layout.md`](/root/3d/generation/docs/dataset_layout.md)
- [`docs/reproducibility.md`](/root/3d/generation/docs/reproducibility.md)
- [`docs/formal_paper_mainline.md`](/root/3d/generation/docs/formal_paper_mainline.md)
- [`docs/repo_audit.md`](/root/3d/generation/docs/repo_audit.md)
