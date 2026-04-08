
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATOR_CKPT="${GENERATOR_CKPT:-outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v0625_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0034000.pt}"
PACKET_DIR="${PACKET_DIR:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets}"
SPLIT_JSON="${SPLIT_JSON:-outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json}"
DATA_CONFIG="${DATA_CONFIG:-configs/data/3dfront_v1.yaml}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime/gpu_smoke.yaml}"
DIFFUSION_CONFIG="${DIFFUSION_CONFIG:-configs/diffusion/visible_locked_occbias_v0625.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-examples/figures/visible_locked_occbias_v0625_main}"
MANIFEST_PATH="${MANIFEST_PATH:-outputs/tables/paper_examples_manifest.json}"

PYTHONPATH=src "$PYTHON_BIN" scripts/analysis/export_paper_examples.py   --checkpoint "$GENERATOR_CKPT"   --packet-dir "$PACKET_DIR"   --config "$DIFFUSION_CONFIG"   --data-config "$DATA_CONFIG"   --runtime-config "$RUNTIME_CONFIG"   --sample-id-json "$SPLIT_JSON"   --split test   --output-dir "$OUTPUT_DIR"   --manifest-path "$MANIFEST_PATH"
