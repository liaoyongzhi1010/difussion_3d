# Amodal Scene Diffusion

Research code for single-view conditioned 3D scene posterior modeling with two explicit factors:

- visible region: deterministic / strongly constrained reconstruction
- occluded region: posterior diffusion generation

The current public code path also includes a learned posterior selector trained on top of a frozen generator, so the system can rank multiple hidden hypotheses instead of only averaging them.

## Repository Layout

```text
.
├── configs/                  # training / eval / pipeline configs
│   ├── data/
│   ├── diffusion/
│   ├── geometry_vae/
│   ├── pipeline/
│   ├── preprocess/
│   └── runtime/
├── scripts/
│   ├── preprocess/           # dataset indexing / packet materialization
│   ├── train/                # generator / selector / geometry VAE training
│   ├── eval/                 # posterior evaluation and reranking metrics
│   ├── analysis/             # analysis utilities
│   └── ops/                  # watchdog / pipeline automation
├── src/
│   └── amodal_scene_diff/    # installable source package
└── README.md
```

Large local artifacts are intentionally excluded from git:

- `outputs/`
- `data/`
- `.venv/`
- internal notes / task logs

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset Convention

This repository expects pre-materialized scene packet files (`*.pt`) and a split file mapping `train / val / test` to sample ids.

Example local paths used by the experiments:

- packet root: `outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets`
- split file: `outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json`

The dataset itself is not bundled into the repository.

## Main Training Entrypoints

### 1. Train the scene diffusion generator

```bash
PYTHONPATH=src python scripts/train/train_scene_diffusion.py \
  --config configs/diffusion/visible_locked_occbias_v050.yaml \
  --data-config configs/data/3dfront_v1.yaml \
  --runtime-config configs/runtime/gpu_smoke.yaml \
  --packet-dir outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets
```

### 2. Train the posterior selector on top of a frozen generator

```bash
PYTHONPATH=src python scripts/train/train_posterior_selector.py \
  --generator-checkpoint outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0030000.pt \
  --packet-dir outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets \
  --config configs/diffusion/visible_locked_occbias_v050.yaml \
  --data-config configs/data/3dfront_v1.yaml \
  --runtime-config configs/runtime/gpu_smoke.yaml \
  --sample-id-json outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json \
  --train-split train \
  --val-split val \
  --test-split test
```

### 3. Evaluate posterior samples and reranking statistics

```bash
PYTHONPATH=src python scripts/eval/eval_scene_posterior.py \
  --checkpoint outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0030000.pt \
  --packet-dir outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets \
  --config configs/diffusion/visible_locked_occbias_v050.yaml \
  --data-config configs/data/3dfront_v1.yaml \
  --runtime-config configs/runtime/gpu_smoke.yaml \
  --sample-id-json outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json \
  --split test
```

## Key Release Files

- generator model: `src/amodal_scene_diff/models/diffusion/scene_denoising_transformer.py`
- generator training: `scripts/train/train_scene_diffusion.py`
- selector training: `scripts/train/train_posterior_selector.py`
- posterior evaluation: `scripts/eval/eval_scene_posterior.py`
- staged pipeline: `configs/pipeline/visible_locked_compare.yaml`

## Reproducibility Notes

- source package layout follows the `src/` convention
- experiment configs are separated from Python code
- local artifacts are excluded from git
- all main experiments are runnable from explicit script entrypoints

