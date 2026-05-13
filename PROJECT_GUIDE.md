# BigEarthNet-S2 I-JEPA Project

## Remote Execution Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Put the dataset archive under data/
#    Example: data/BigEarthNet-S2.tar.zst

# 3. Smoke tests (fast sanity checks)
make smokes

# 4. Full pipeline by phases
make phase0
make phase1
make phase2
make phase3
make phase4
make phase5

# Or run everything end-to-end
make all
```

## Smoke Tests (run before long jobs)

```bash
make smokes

# Optional: smoke versions of each phase (tiny jobs + separate outputs)
make smoke_all
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
├── scripts/               # Phase wrappers + smoke tests
└── requirements.txt
```
