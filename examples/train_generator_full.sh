#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/3dfront_v1.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/visible_locked_occbias_v0625.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/real_data/release_generator/checkpoints_scene_denoiser_v1}"
SUMMARY_PATH="${SUMMARY_PATH:-outputs/real_data/release_generator/denoiser_train_summary.json}"

PYTHONPATH=src "$PYTHON_BIN" scripts/train/train_scene_diffusion.py \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --packet-dir "$PACKET_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --save-summary "$SUMMARY_PATH"
