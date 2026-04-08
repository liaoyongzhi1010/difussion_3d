# Amodal Scene Diffusion

A research codebase for single-view conditioned 3D scene posterior modeling with an explicit factorization:

- visible region: deterministic / strongly constrained reconstruction
- occluded region: posterior diffusion generation
- posterior selection: a learned selector trained on top of a frozen generator

This repository is intended to be the long-term project root for training, evaluation, qualitative examples, and future fixes. The public layout follows a cleaner release-style structure instead of a raw experiment workspace.

## Highlights

- `src/` package layout for installable research code
- explicit `configs/` for diffusion, selector, geometry, and pipeline runs
- standalone `scripts/train/`, `scripts/eval/`, and `scripts/preprocess/` entrypoints
- `examples/` shell scripts for common train/eval workflows
- `docs/` notes for quickstart, dataset layout, reproducibility, and current paper results

## Repository Layout

```text
.
├── configs/                  # experiment configs
│   ├── data/
│   ├── diffusion/
│   ├── geometry_vae/
│   ├── pipeline/
│   ├── preprocess/
│   └── runtime/
├── docs/                     # public-facing documentation
├── examples/                 # runnable example commands and figure placeholders
├── scripts/
│   ├── preprocess/
│   ├── train/
│   ├── eval/
│   ├── analysis/
│   └── ops/
├── src/
│   └── amodal_scene_diff/
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## What Is Included vs Excluded

This repo tracks code, configs, and reproducibility instructions.

It intentionally does **not** track large local artifacts such as:

- `outputs/`
- `data/`
- `.venv/`
- internal notes and task scratch files

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You can also use:

```bash
make install
```

## Quick Start

See the following docs first:

- [`docs/quickstart.md`](/root/3d/generation/docs/quickstart.md)
- [`docs/dataset_layout.md`](/root/3d/generation/docs/dataset_layout.md)
- [`docs/reproducibility.md`](/root/3d/generation/docs/reproducibility.md)

## Main Entry Points

### Train the scene diffusion generator

```bash
make train-generator \
  PACKET_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets
```

### Train the posterior selector

```bash
make train-selector \
  PACKET_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets \
  SPLIT_JSON=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json \
  GENERATOR_CKPT=outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0034000.pt
```

### Evaluate posterior samples and reranking

```bash
make eval-posterior \
  PACKET_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets \
  SPLIT_JSON=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json \
  GENERATOR_CKPT=outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0034000.pt
```

### Build paper tables and summary

```bash
make paper-report
```

### Export paper examples

```bash
make paper-examples
```

## Canonical Release Files

- generator model: `src/amodal_scene_diff/models/diffusion/scene_denoising_transformer.py`
- generator trainer: `scripts/train/train_scene_diffusion.py`
- selector trainer: `scripts/train/train_posterior_selector.py`
- posterior evaluation: `scripts/eval/eval_scene_posterior.py`
- staged pipeline config: `configs/pipeline/visible_locked_compare.yaml`

## Examples

Runnable command examples live in:

- [`examples/train_generator_full.sh`](/root/3d/generation/examples/train_generator_full.sh)
- [`examples/train_selector_full.sh`](/root/3d/generation/examples/train_selector_full.sh)
- [`examples/eval_posterior_full.sh`](/root/3d/generation/examples/eval_posterior_full.sh)
- [`examples/build_paper_report.sh`](/root/3d/generation/examples/build_paper_report.sh)
- [`examples/export_paper_examples.sh`](/root/3d/generation/examples/export_paper_examples.sh)
- [`examples/smoke_selector.sh`](/root/3d/generation/examples/smoke_selector.sh)

Qualitative figure placeholders live under:

- [`examples/figures/README.md`](/root/3d/generation/examples/figures/README.md)
- [`docs/figures/README.md`](/root/3d/generation/docs/figures/README.md)

## Repository Policy

The GitHub repository is the single mainline project for all future work.

- training, evaluation, visualization, and example export should all be driven from this repository
- bug fixes and method iterations should be implemented here, validated here, and pushed back here
- representative example figures and lightweight result summaries may be committed when useful

See [`CONTRIBUTING.md`](/root/3d/generation/CONTRIBUTING.md) for the explicit project policy.

## Reproducibility Conventions

- `src/` layout for clean imports
- configs separated from code
- public examples are command-first, not notebook-first
- large local artifacts are ignored from git
- repository can be patched and re-pushed as experiments evolve
