#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/pixarmesh_single_view_main.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml}"
MODEL_CKPT="${MODEL_CKPT:-outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048/checkpoints_single_view_scene/latest.pt}"
SPLIT_JSON="${SPLIT_JSON:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json}"
SPLIT="${SPLIT:-test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048}"
NUM_POSTERIOR_SAMPLES="${NUM_POSTERIOR_SAMPLES:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"

PYTHONPATH=src "$PYTHON_BIN" scripts/ops/paper_mainline_harness.py \
  --mode eval_state \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --packet-dir "$PACKET_DIR" \
  --split-json "$SPLIT_JSON" \
  --output-root "$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" scripts/eval/eval_single_view_scene.py \
  --checkpoint "$MODEL_CKPT" \
  --packet-dir "$PACKET_DIR" \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --batch-size "$BATCH_SIZE" \
  --num-posterior-samples "$NUM_POSTERIOR_SAMPLES" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --sample-id-json "$SPLIT_JSON" \
  --split "$SPLIT" \
  --save-summary "$OUTPUT_ROOT/single_view_eval_${SPLIT}_p${NUM_POSTERIOR_SAMPLES}_s${NUM_INFERENCE_STEPS}.json"
