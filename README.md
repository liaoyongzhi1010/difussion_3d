# Amodal Scene Diffusion

Single-view 3D reconstruction with amodal completion: **direct reconstruction
of visible content + conditional diffusion over hidden content**. The repo
targets a CVPR/ICCV/NeurIPS-grade submission and is organized accordingly.

## What's here

- A single paper mainline (`v5_dit_detr_dinov2_large`) plus named ablations
  and two reference baselines.
- A public-benchmark path (3D-FRONT + ScanNet) alongside the private
  pixarmesh dataset.
- A top-venue 3D metric suite (Chamfer Distance, F-score@τ, hidden recall,
  collision / support violation) as a small, tested library.
- A clean training/evaluation engine and a multi-seed harness that reports
  mean ± 95% CI.

## Directory layout

```
configs/
  data/            # pixarmesh.yaml, threedfront.yaml, scannet.yaml
  backbones/       # dinov2_base_frozen, dinov2_large_hybrid, patch_vit
  experiments/
    main/v5_dit_detr_dinov2_large.yaml
    ablations/{no_dit,no_detr,no_hidden_diffusion,no_occlusion_bias}.yaml
    baselines/{fullscene_diffusion,visible_only}.yaml
  geometry_vae/
  runtime/
docs/
  superpowers/specs/   # design specs + dataset acquisition scripts
scripts/
  data/            # dataset download shims
  preprocess/
  train/           # legacy v3/v4 driver (kept)
  eval/
src/amodal_scene_diff/
  backbones/       # patch-ViT, DINOv2, DINOv2-hybrid
  datasets/        # pixarmesh, threedfront, scannet
  diffusion/       # scene_model, scheduler, sampler
  engine/          # train_loop, eval_loop, seed_harness
  geometry/        # GeometryVAE
  heads/           # dit_hidden, detr_visible, layout, relation
  metrics/         # chamfer, fscore, box_iou, hidden_recall, collision
  structures/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Smoke the v5 config (2 steps, exercises the scene model end-to-end)
python -m amodal_scene_diff.engine.train_loop \
  --config configs/experiments/main/v5_dit_detr_dinov2_large.yaml \
  --train-steps 2

# 2. 2k-step smoke training run
python -m amodal_scene_diff.engine.train_loop \
  --config configs/experiments/main/v5_dit_detr_dinov2_large.yaml

# 3. Evaluate a checkpoint with the 3D metric suite
python -m amodal_scene_diff.engine.eval_loop \
  --config configs/experiments/main/v5_dit_detr_dinov2_large.yaml \
  --checkpoint outputs/v5_dit_detr_dinov2_large_smoke/latest.pt \
  --output-dir outputs/v5_dit_detr_dinov2_large_smoke/eval

# 4. Three-seed run with 95% CI aggregation
python -m amodal_scene_diff.engine.seed_harness \
  --config configs/experiments/main/v5_dit_detr_dinov2_large.yaml \
  --seeds 0 1 2 \
  --output-root outputs/v5_seeds
```

## Public datasets

The pixarmesh packets used throughout development live under
`outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/`. For
paper-grade comparisons you additionally need:

- **3D-FRONT** — EULA-gated. See `docs/superpowers/specs/datasets/README.md`
  for the download + render workflow.
- **ScanNet** — token-gated. Same folder has the wrapper script.

Both datasets extract into `data/external/{3d_front,scannet}/rendered/` as
`.pt` packets matching `PixarMeshPacketDataset`'s schema.

## Design doc

`docs/superpowers/specs/2026-04-18-top-venue-restructure-design.md` is the
source of truth for this restructure: scope, deletions, new tree, commit
plan, acceptance criteria, and what's deliberately out of scope.

## Status

| Component                       | State                             |
|---------------------------------|-----------------------------------|
| Pixarmesh packets + split       | done                              |
| DiT hidden denoiser             | implemented, config wiring TODO   |
| DETR visible head               | implemented, config wiring TODO   |
| 3D metric suite                 | done, smoke-tested                |
| Engine (train/eval/seeds)       | done, `--help` verified           |
| 3D-FRONT loader + render        | loader done; renderer TODO        |
| ScanNet loader + render         | loader done; renderer TODO        |
| v5 smoke training (2k steps)    | run via engine.train_loop         |

## License

See upstream model licenses for DINOv2 (Meta) and any weights pulled from
HuggingFace. Dataset terms (3D-FRONT Alibaba, ScanNet Stanford) bind each
downstream user individually.
