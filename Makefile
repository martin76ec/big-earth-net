PY ?= python
TAR_PATH ?= /home/elarreaa/big-earth-data/BigEarthNet-S2.tar.zst

DATA_SMOKE ?= data/processed/bigearthnet_smoke
DATA_FULL ?= data/processed/bigearthnet_25k_singlelabel

CKPT_DIR ?= checkpoints/ijepa
CKPT ?= $(CKPT_DIR)/ijepa-best.ckpt

EMB_IJEPA_SMOKE ?= data/processed/ijepa_embeddings_smoke.npz
EMB_RESNET_SMOKE ?= data/processed/resnet50_embeddings_smoke.npz
EMB_IJEPA_FULL ?= data/processed/ijepa_embeddings.npz
EMB_RESNET_FULL ?= data/processed/resnet50_embeddings.npz

HPO_DIR ?= reports/hpo

SMOKE_PATCHES ?= 200
SMOKE_EPOCHS ?= 1
SMOKE_BS ?= 16
SMOKE_TRIALS ?= 5
SMOKE_FOLDS ?= 3
SMOKE_REPEATS ?= 1

.PHONY: help install smoke_data_prep smoke_smoke_tests smoke_train smoke_feats smoke_hpo smoke_cv smoke_all \
	full_data_prep full_train full_feats full_hpo full_cv full_all

help:
	@echo "Targets:"
	@echo "  make install"
	@echo "  make smoke_all   (tiny end-to-end validation)"
	@echo "  make full_all    (25k run)"
	@echo "  make smoke_data_prep smoke_train smoke_feats smoke_hpo smoke_cv"
	@echo "  make full_data_prep full_train full_feats full_hpo full_cv"

install:
	$(PY) -m pip install -r requirements.txt

smoke_data_prep:
	$(PY) src/data/extract_and_downsample.py \
		--tar_path $(TAR_PATH) \
		--output_dir $(DATA_SMOKE) \
		--max_patches $(SMOKE_PATCHES) \
		--single_label_strategy first

smoke_smoke_tests:
	$(PY) scripts/smoke_cv.py
	$(PY) scripts/smoke_ijepa.py --config configs/ijepa_small.yaml

smoke_train:
	$(PY) src/train/ijepa_train.py \
		--config configs/ijepa_small.yaml \
		--data_dir $(DATA_SMOKE) \
		--output_dir $(CKPT_DIR) \
		--batch_size $(SMOKE_BS) \
		--epochs $(SMOKE_EPOCHS)

smoke_feats:
	$(PY) src/train/extract_features.py \
		--model_type ijepa \
		--checkpoint $(CKPT) \
		--data_dir $(DATA_SMOKE) \
		--output $(EMB_IJEPA_SMOKE)
	$(PY) src/train/extract_features.py \
		--model_type resnet50 \
		--data_dir $(DATA_SMOKE) \
		--output $(EMB_RESNET_SMOKE)

smoke_hpo:
	$(PY) notebooks/04_hpo_and_cv.py \
		--mode hpo \
		--embeddings_a $(EMB_IJEPA_SMOKE) \
		--embeddings_b $(EMB_RESNET_SMOKE) \
		--n_trials $(SMOKE_TRIALS) \
		--output_dir $(HPO_DIR)

smoke_cv:
	$(PY) notebooks/04_hpo_and_cv.py \
		--mode cv \
		--embeddings_a $(EMB_IJEPA_SMOKE) \
		--embeddings_b $(EMB_RESNET_SMOKE) \
		--n_folds $(SMOKE_FOLDS) \
		--n_repeats $(SMOKE_REPEATS) \
		--output_dir $(HPO_DIR)

smoke_all: smoke_data_prep smoke_smoke_tests smoke_train smoke_feats smoke_hpo smoke_cv

full_data_prep:
	$(PY) src/data/extract_and_downsample.py \
		--tar_path $(TAR_PATH) \
		--output_dir $(DATA_FULL) \
		--max_patches 25000 \
		--single_label_strategy first

full_train:
	$(PY) src/train/ijepa_train.py \
		--config configs/ijepa_small.yaml \
		--data_dir $(DATA_FULL) \
		--output_dir $(CKPT_DIR)

full_feats:
	$(PY) src/train/extract_features.py \
		--model_type ijepa \
		--checkpoint $(CKPT) \
		--data_dir $(DATA_FULL) \
		--output $(EMB_IJEPA_FULL)
	$(PY) src/train/extract_features.py \
		--model_type resnet50 \
		--data_dir $(DATA_FULL) \
		--output $(EMB_RESNET_FULL)

full_hpo:
	$(PY) notebooks/04_hpo_and_cv.py \
		--mode hpo \
		--embeddings_a $(EMB_IJEPA_FULL) \
		--embeddings_b $(EMB_RESNET_FULL) \
		--n_trials 50 \
		--output_dir $(HPO_DIR)

full_cv:
	$(PY) notebooks/04_hpo_and_cv.py \
		--mode cv \
		--embeddings_a $(EMB_IJEPA_FULL) \
		--embeddings_b $(EMB_RESNET_FULL) \
		--n_folds 10 \
		--n_repeats 10 \
		--output_dir $(HPO_DIR)

full_all: full_data_prep full_train full_feats full_hpo full_cv
