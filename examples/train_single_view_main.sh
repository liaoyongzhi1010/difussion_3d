#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/pixarmesh_single_view_main.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml}"
SPLIT_JSON="${SPLIT_JSON:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-1949}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
VAL_SPLIT="${VAL_SPLIT:-val}"
VAL_NUM_INFERENCE_STEPS="${VAL_NUM_INFERENCE_STEPS:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048}"

PYTHONPATH=src "$PYTHON_BIN" scripts/ops/paper_mainline_harness.py \
  --mode train \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --packet-dir "$PACKET_DIR" \
  --split-json "$SPLIT_JSON" \
  --output-root "$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" scripts/train/train_single_view_scene.py \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --packet-dir "$PACKET_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --train-steps "$TRAIN_STEPS" \
  --checkpoint-dir "$OUTPUT_ROOT/checkpoints_single_view_scene" \
  --save-summary "$OUTPUT_ROOT/single_view_scene_train_summary.json" \
  --val-sample-id-json "$SPLIT_JSON" \
  --val-split "$VAL_SPLIT" \
  --val-every-steps "$VAL_EVERY_STEPS" \
  --val-num-inference-steps "$VAL_NUM_INFERENCE_STEPS" \
  --save-every-steps 1000
