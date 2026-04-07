# Quickstart

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Prepare local data paths

This repository expects local materialized scene packets and split metadata.

Minimal required paths:

- packet directory
- split json
- generator checkpoint for selector training or posterior evaluation

See [`dataset_layout.md`](/root/3d/generation/docs/dataset_layout.md).

## 3. Run a smoke test

```bash
bash examples/smoke_selector.sh
```

## 4. Run full training

```bash
bash examples/train_selector_full.sh
```

## 5. Run evaluation

```bash
bash examples/eval_posterior_full.sh
```

## 6. Validate the repository entrypoints

```bash
make check
```
