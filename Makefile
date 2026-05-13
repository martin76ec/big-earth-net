PY ?= python

.PHONY: help install smokes phase0 phase1 phase2 phase3 phase4 phase5 all \
	smoke_phase0 smoke_phase1 smoke_phase2 smoke_phase3 smoke_phase4 smoke_phase5 smoke_all

help:
	@echo "Targets:"
	@echo "  make install"
	@echo "  make smokes         (fast sanity checks)"
	@echo "  make phase0..phase5 (full pipeline phases)"
	@echo "  make all            (phase0..phase5)"
	@echo "  make smoke_all      (smoke_phase0..smoke_phase5 + smokes)"

install:
	uv sync

smokes:
	uv run $(PY) scripts/smoke_data.py
	uv run $(PY) scripts/smoke_ijepa.py
	uv run $(PY) scripts/smoke_cv.py

phase0:
	uv run $(PY) scripts/phase0_data_prep.py

phase1:
	uv run $(PY) scripts/phase1_train.py

phase2:
	uv run $(PY) scripts/phase2_extract_features.py --model ijepa

phase3:
	uv run $(PY) scripts/phase2_extract_features.py --model resnet50

phase4:
	uv run $(PY) scripts/phase4_hpo.py

phase5:
	uv run $(PY) scripts/phase5_cv.py

all: phase0 phase1 phase2 phase3 phase4 phase5

smoke_phase0:
	uv run $(PY) scripts/phase0_data_prep.py --smoke

smoke_phase1:
	uv run $(PY) scripts/phase1_train.py --smoke

smoke_phase2:
	uv run $(PY) scripts/phase2_extract_features.py --model ijepa --smoke

smoke_phase3:
	uv run $(PY) scripts/phase2_extract_features.py --model resnet50 --smoke

smoke_phase4:
	uv run $(PY) scripts/phase4_hpo.py --smoke

smoke_phase5:
	uv run $(PY) scripts/phase5_cv.py --smoke

smoke_all: smokes smoke_phase0 smoke_phase1 smoke_phase2 smoke_phase3 smoke_phase4 smoke_phase5
