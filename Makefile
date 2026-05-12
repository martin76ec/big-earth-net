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
	$(PY) -m pip install -r requirements.txt

smokes:
	$(PY) scripts/smoke_data.py
	$(PY) scripts/smoke_ijepa.py
	$(PY) scripts/smoke_cv.py

phase0:
	$(PY) scripts/phase0_data_prep.py

phase1:
	$(PY) scripts/phase1_train.py

phase2:
	$(PY) scripts/phase2_extract_features.py --model ijepa

phase3:
	$(PY) scripts/phase2_extract_features.py --model resnet50

phase4:
	$(PY) scripts/phase4_hpo.py

phase5:
	$(PY) scripts/phase5_cv.py

all: phase0 phase1 phase2 phase3 phase4 phase5

smoke_phase0:
	$(PY) scripts/phase0_data_prep.py --smoke

smoke_phase1:
	$(PY) scripts/phase1_train.py --smoke

smoke_phase2:
	$(PY) scripts/phase2_extract_features.py --model ijepa --smoke

smoke_phase3:
	$(PY) scripts/phase2_extract_features.py --model resnet50 --smoke

smoke_phase4:
	$(PY) scripts/phase4_hpo.py --smoke

smoke_phase5:
	$(PY) scripts/phase5_cv.py --smoke

smoke_all: smokes smoke_phase0 smoke_phase1 smoke_phase2 smoke_phase3 smoke_phase4 smoke_phase5
