# BigEarthNet-S2 I-JEPA Project

## Remote Execution Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make runbook executable
chmod +x runbook.sh

# 3. Phase 0 — Extract & downsample data (detached tmux session)
./runbook.sh phase_0

# 4. Phase 1 — I-JEPA pretraining (longest job, ~18-24h)
./runbook.sh phase_1

# 5. Phase 2 & 3 — Feature extraction
./runbook.sh phase_2
./runbook.sh phase_3

# 6. Phase 4 — HPO with Optuna
./runbook.sh phase_4

# 7. Phase 5 — Rigorous 10x10 CV + Wilcoxon test
./runbook.sh phase_5

# Reattach to any session:
tmux attach -t ijepa_pretrain
```

## Smoke Tests (run before long jobs)

```bash
./runbook.sh smoke_data   # requires Phase 0 done
./runbook.sh smoke_ijepa  # no data needed
./runbook.sh smoke_cv     # no data needed
```

## Directory Layout

```
.
├── configs/               # YAML configs for I-JEPA and HPO
├── data/
│   ├── raw/               # Extracted BigEarthNet-S2 (gitignored)
│   └── processed/         # Downsampled metadata + embeddings
├── notebooks/
│   ├── 01_eda.ipynb       # EDA report
│   └── 05_report.ipynb    # Final report (to fill after experiments)
├── src/
│   ├── data/              # Extraction, datamodule, transforms
│   ├── models/            # I-JEPA adapter, ResNet-50 baseline
│   └── train/             # Training & feature extraction scripts
├── scripts/               # Smoke tests
├── runbook.sh             # Tmux-driven phase launcher
└── requirements.txt
```
