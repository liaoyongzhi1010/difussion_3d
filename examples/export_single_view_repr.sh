#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/pixarmesh_single_view_main.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml}"
MODEL_CKPT="${MODEL_CKPT:-outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048/checkpoints_single_view_scene/latest.pt}"
GEOMETRY_CONFIG="${GEOMETRY_CONFIG:-configs/geometry_vae/heavy.yaml}"
GEOMETRY_CKPT="${GEOMETRY_CKPT:-outputs/real_data/geometry_vae_objects_v1_full/checkpoints_geometry_vae_heavy_fullresume/latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/scene_repr}"
MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_ROOT/scene_repr_summary.json}"
SPLIT_JSON="${SPLIT_JSON:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json}"

PYTHONPATH=src "$PYTHON_BIN" scripts/ops/paper_mainline_harness.py \
  --mode export \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --packet-dir "$PACKET_DIR" \
  --split-json "$SPLIT_JSON" \
  --output-root "$OUTPUT_ROOT"

PYTHONPATH=src "$PYTHON_BIN" scripts/eval/export_single_view_scene_repr.py \
  --checkpoint "$MODEL_CKPT" \
  --packet-dir "$PACKET_DIR" \
  --config "$DIFFUSION_CONFIG" \
  --data-config "$DATA_CONFIG" \
  --runtime-config "$RUNTIME_CONFIG" \
  --geometry-config "$GEOMETRY_CONFIG" \
  --geometry-checkpoint "$GEOMETRY_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --save-summary "$MANIFEST_PATH"
