#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATOR_CKPT="${GENERATOR_CKPT:-outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0030000.pt}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
SPLIT_JSON="${SPLIT_JSON:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/3dfront_v1.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/visible_locked_occbias_v050.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/real_data/release_selector}"

PYTHONPATH=src "$PYTHON_BIN" scripts/train/train_posterior_selector.py \
  --generator-checkpoint "$GENERATOR_CKPT" \
  --packet-dir "$PACKET_DIR" \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --sample-id-json "$SPLIT_JSON" \
  --train-split train \
  --val-split val \
  --test-split test \
  --batch-size 4 \
  --num-posterior-samples 8 \
  --eval-num-posterior-samples 16 \
  --num-inference-steps 20 \
  --train-steps 1000 \
  --eval-every-steps 100 \
  --save-every-steps 100 \
  --checkpoint-dir "$OUTPUT_ROOT/checkpoints_posterior_selector" \
  --save-summary "$OUTPUT_ROOT/selector_train_summary.json"
