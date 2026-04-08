# Reproducibility

## Canonical configuration family

Current mainline release path:

- generator config: `configs/diffusion/visible_locked_occbias_v0625.yaml`
- selector trainer: `scripts/train/train_posterior_selector.py`
- posterior eval: `scripts/eval/eval_scene_posterior.py`

## Recommended release workflow

1. Train the generator.
2. Freeze the best checkpoint.
3. Train the selector on `train`, monitor on `val`.
4. Evaluate on `test` with both `p5` and stronger `p20/p50` posterior sweeps.
5. Build the paper summary via `make paper-report`.
6. Export qualitative figures into `examples/figures/<experiment_name>/` via `make paper-examples`.

## Standard commands

### Generator

```bash
make train-generator PACKET_DIR=<packet_dir>
```

### Selector

```bash
make train-selector \
  PACKET_DIR=<packet_dir> \
  SPLIT_JSON=<split_json> \
  GENERATOR_CKPT=<generator_ckpt>
```

### Posterior evaluation

```bash
make eval-posterior \
  PACKET_DIR=<packet_dir> \
  SPLIT_JSON=<split_json> \
  GENERATOR_CKPT=<generator_ckpt>
```

### Paper summary and figures

```bash
make paper-report
make paper-examples
```

## Repository validation

Run this before pushing:

```bash
make check
```
