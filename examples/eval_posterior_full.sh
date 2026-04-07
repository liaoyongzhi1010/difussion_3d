#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATOR_CKPT="${GENERATOR_CKPT:-outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0030000.pt}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
SPLIT_JSON="${SPLIT_JSON:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/3dfront_v1.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/visible_locked_occbias_v050.yaml}"
SUMMARY_PATH="${SUMMARY_PATH:-outputs/real_data/release_eval/posterior_eval_test.json}"

PYTHONPATH=src "$PYTHON_BIN" scripts/eval/eval_scene_posterior.py \
  --checkpoint "$GENERATOR_CKPT" \
  --packet-dir "$PACKET_DIR" \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --sample-id-json "$SPLIT_JSON" \
  --split test \
  --save-summary "$SUMMARY_PATH"
