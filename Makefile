PYTHON ?= python
PACKET_DIR ?= outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/packets
SPLIT_JSON ?= outputs/real_data/pixarmesh_depr_bootstrap_train2048_norm/index/split_to_sample_ids.json
DATA_CONFIG ?= configs/data/3dfront_v1.yaml
RUNTIME_CONFIG ?= configs/runtime/gpu_smoke.yaml
DIFFUSION_CONFIG ?= configs/diffusion/visible_locked_occbias_v050.yaml
GENERATOR_CKPT ?= outputs/real_data/pixarmesh_bootstrap_visiblelocked_occbias_v050_ft_b128_train2048/checkpoints_scene_denoiser_v1/step_0030000.pt

.PHONY: install check smoke-selector train-generator train-selector eval-posterior

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) -m py_compile scripts/train/train_scene_diffusion.py scripts/train/train_posterior_selector.py scripts/eval/eval_scene_posterior.py

smoke-selector:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) SPLIT_JSON=$(SPLIT_JSON) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(DIFFUSION_CONFIG) GENERATOR_CKPT=$(GENERATOR_CKPT) bash examples/smoke_selector.sh

train-generator:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(DIFFUSION_CONFIG) bash examples/train_generator_full.sh

train-selector:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) SPLIT_JSON=$(SPLIT_JSON) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(DIFFUSION_CONFIG) GENERATOR_CKPT=$(GENERATOR_CKPT) bash examples/train_selector_full.sh

eval-posterior:
	PYTHON_BIN=$(PYTHON) PACKET_DIR=$(PACKET_DIR) SPLIT_JSON=$(SPLIT_JSON) DATA_CONFIG=$(DATA_CONFIG) RUNTIME_CONFIG=$(RUNTIME_CONFIG) DIFFUSION_CONFIG=$(DIFFUSION_CONFIG) GENERATOR_CKPT=$(GENERATOR_CKPT) bash examples/eval_posterior_full.sh
