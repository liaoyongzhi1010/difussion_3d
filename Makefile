PYTHON ?= python3
DATA_CONFIG ?= configs/data/pixarmesh_single_view_main.yaml
RUNTIME_CONFIG ?= configs/runtime/gpu_smoke.yaml
SINGLE_VIEW_CONFIG ?= configs/diffusion/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen.yaml
PACKET_DIR ?= outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets
SPLIT_JSON ?= outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json
SPLIT ?= test
SINGLE_VIEW_CKPT ?= outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048/checkpoints_single_view_scene/latest.pt
GEOMETRY_CONFIG ?= configs/geometry_vae/heavy.yaml
GEOMETRY_CKPT ?= outputs/real_data/geometry_vae_objects_v1_full/checkpoints_geometry_vae_heavy_fullresume/latest.pt
OUTPUT_ROOT ?= outputs/real_data/single_view_visible_direct_hidden_diffusion_v3_dinov2_frozen_train2048
VAL_EVERY_STEPS ?= 1000

.PHONY: install check harness-single-view pipeline-board train-single-view eval-single-view-state eval-single-view-render export-single-view-repr restore-rgb-paths monitor-single-view

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m py_compile \
		scripts/train/train_single_view_scene.py \
		scripts/train/train_geometry_vae.py \
		scripts/eval/eval_single_view_scene.py \
		scripts/eval/eval_single_view_render_metrics.py \
		scripts/eval/export_single_view_scene_repr.py \
		scripts/preprocess/restore_packet_rgb_paths_from_parquet.py \
		scripts/ops/monitor_single_view_training.py


harness-single-view:
	PYTHONPATH=src $(PYTHON) scripts/ops/paper_mainline_harness.py \
		--mode train \
		--config $(SINGLE_VIEW_CONFIG) \
		--data-config $(DATA_CONFIG) \
		--runtime-config $(RUNTIME_CONFIG) \
		--packet-dir $(PACKET_DIR) \
		--split-json $(SPLIT_JSON) \
		--output-root $(OUTPUT_ROOT)


pipeline-board:
	PYTHONPATH=src $(PYTHON) scripts/ops/single_view_pipeline_board.py \
		--data-config $(DATA_CONFIG) \
		--packet-dir $(PACKET_DIR) \
		--split-json $(SPLIT_JSON) \
		--output-root $(OUTPUT_ROOT)

train-single-view:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(SINGLE_VIEW_CONFIG) OUTPUT_ROOT=$(OUTPUT_ROOT) SPLIT_JSON=$(SPLIT_JSON) VAL_EVERY_STEPS=$(VAL_EVERY_STEPS) bash examples/train_single_view_main.sh

eval-single-view-state:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(SINGLE_VIEW_CONFIG) MODEL_CKPT=$(SINGLE_VIEW_CKPT) SPLIT_JSON=$(SPLIT_JSON) SPLIT=$(SPLIT) OUTPUT_ROOT=$(OUTPUT_ROOT) bash examples/eval_single_view_state.sh

eval-single-view-render:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(SINGLE_VIEW_CONFIG) MODEL_CKPT=$(SINGLE_VIEW_CKPT) SPLIT_JSON=$(SPLIT_JSON) SPLIT=$(SPLIT) OUTPUT_ROOT=$(OUTPUT_ROOT) bash examples/eval_single_view_render.sh

export-single-view-repr:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(SINGLE_VIEW_CONFIG) MODEL_CKPT=$(SINGLE_VIEW_CKPT) GEOMETRY_CONFIG=$(GEOMETRY_CONFIG) GEOMETRY_CKPT=$(GEOMETRY_CKPT) OUTPUT_ROOT=$(OUTPUT_ROOT) bash examples/export_single_view_repr.sh

restore-rgb-paths:
	PYTHONPATH=src $(PYTHON) scripts/preprocess/restore_packet_rgb_paths_from_parquet.py \
		--packet-root outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm \
		--parquet-glob 'outputs/real_data/hf_cache/data/train-*.parquet' \
		--save-summary outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/restore_packet_rgb_summary.json

monitor-single-view:
	PYTHONPATH=src $(PYTHON) scripts/ops/monitor_single_view_training.py \
		--output-root $(OUTPUT_ROOT) \
		--state-path $(OUTPUT_ROOT)/monitor/state.json \
		--log-path $(OUTPUT_ROOT)/monitor/monitor.log
