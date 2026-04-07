#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/checkpoints_scene_denoiser_v1
PACKET_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets
SPLIT_JSON=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json
OUT_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/val_ckpt_scan
LOG_DIR=outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

for STEP in 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000; do
  CKPT=$(printf "%s/step_%07d.pt" "$CHECKPOINT_DIR" "$STEP")
  OUT=$(printf "%s/step_%07d_val_p5_s20.json" "$OUT_DIR" "$STEP")
  LOG=$(printf "%s/val_step%05d_p5_s20.log" "$LOG_DIR" "$STEP")
  while [ ! -f "$CKPT" ]; do
    sleep 30
  done
  if [ ! -f "$OUT" ]; then
    env PYTHONPATH=src .venv/bin/python scripts/eval/eval_scene_posterior.py       --checkpoint "$CKPT"       --packet-dir "$PACKET_DIR"       --sample-id-json "$SPLIT_JSON"       --split val       --batch-size 16       --max-samples 0       --num-posterior-samples 5       --num-inference-steps 20       --save-summary "$OUT" > "$LOG" 2>&1
  fi
done
