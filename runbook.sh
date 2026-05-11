#!/usr/bin/env bash
# runbook.sh — One-liners for each project phase (tmux-driven execution)
# Usage on remote container:
#   chmod +x runbook.sh
#   ./runbook.sh phase_0   # launches tmux session for Phase 0
#
# To detach from tmux:  Ctrl-b d
# To reattach:         tmux attach -t <session_name>
# To list sessions:    tmux ls

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

timestamp() { date +%Y%m%d_%H%M%S; }

phase_0() {
    local sess="data_setup"
    tmux new-session -d -s "$sess" -n "extract"
    tmux send-keys -t "$sess" "cd $PROJECT_ROOT && python src/data/extract_and_downsample.py \
        --tar_path /home/elarreaa/big-earth-data/BigEarthNet-S2.tar.zst \
        --output_dir data/processed/bigearthnet_25k_singlelabel \
        --max_patches 25000 \
        --single_label_strategy first \
        > $LOG_DIR/phase0_$(timestamp).log 2>&1" C-m
    echo "[Phase 0] Session '$sess' started. Detach with Ctrl-b d."
    echo "[Phase 0] Log: $LOG_DIR/phase0_*.log"
}

phase_1() {
    local sess="ijepa_pretrain"
    tmux new-session -d -s "$sess" -n "train"
    tmux send-keys -t "$sess" "cd $PROJECT_ROOT && python src/train/ijepa_train.py \
        --config configs/ijepa_small.yaml \
        --data_dir data/processed/bigearthnet_25k_singlelabel \
        --output_dir checkpoints/ijepa \
        > $LOG_DIR/phase1_$(timestamp).log 2>&1" C-m
    echo "[Phase 1] Session '$sess' started. Detach with Ctrl-b d."
    echo "[Phase 1] Log: $LOG_DIR/phase1_*.log"
}

phase_2() {
    local sess="feat_ijepa"
    tmux new-session -d -s "$sess" -n "extract"
    tmux send-keys -t "$sess" "cd $PROJECT_ROOT && python src/train/extract_features.py \
        --model_type ijepa \
        --checkpoint checkpoints/ijepa/ijepa_best.ckpt \
        --data_dir data/processed/bigearthnet_25k_singlelabel \
        --output data/processed/ijepa_embeddings.npy \
        > $LOG_DIR/phase2_$(timestamp).log 2>&1" C-m
    echo "[Phase 2] Session '$sess' started."
}

phase_3() {
    local sess="feat_resnet"
    tmux new-session -d -s "$sess" -n "extract"
    tmux send-keys -t "$sess" "cd $PROJECT_ROOT && python src/train/extract_features.py \
        --model_type resnet50 \
        --data_dir data/processed/bigearthnet_25k_singlelabel \
        --output data/processed/resnet50_embeddings.npy \
        > $LOG_DIR/phase3_$(timestamp).log 2>&1" C-m
    echo "[Phase 3] Session '$sess' started."
}

phase_4() {
    local sess="hpo_optuna"
    tmux new-session -d -s "$sess" -n "hpo"
    tmux send-keys -t "$sess" "cd $PROJECT_ROOT && python notebooks/04_hpo_and_cv.py \
        --mode hpo \
        --n_trials 50 \
        > $LOG_DIR/phase4_$(timestamp).log 2>&1" C-m
    echo "[Phase 4] Session '$sess' started."
}

phase_5() {
    local sess="cv_rigor"
    tmux new-session -d -s "$sess" -n "cv"
    tmux send-keys -t "$sess" "cd $PROJECT_ROOT && python notebooks/04_hpo_and_cv.py \
        --mode cv \
        --n_repeats 10 \
        --n_folds 10 \
        > $LOG_DIR/phase5_$(timestamp).log 2>&1" C-m
    echo "[Phase 5] Session '$sess' started."
}

smoke_data() {
    python "$PROJECT_ROOT/scripts/smoke_data.py" \
        --data_dir data/processed/bigearthnet_25k_singlelabel
}

smoke_ijepa() {
    python "$PROJECT_ROOT/scripts/smoke_ijepa.py" \
        --config configs/ijepa_small.yaml
}

smoke_cv() {
    python "$PROJECT_ROOT/scripts/smoke_cv.py"
}

case "${1:-}" in
    phase_0) phase_0 ;;
    phase_1) phase_1 ;;
    phase_2) phase_2 ;;
    phase_3) phase_3 ;;
    phase_4) phase_4 ;;
    phase_5) phase_5 ;;
    smoke_data) smoke_data ;;
    smoke_ijepa) smoke_ijepa ;;
    smoke_cv) smoke_cv ;;
    *)
        echo "Usage: $0 {phase_0|phase_1|phase_2|phase_3|phase_4|phase_5|smoke_data|smoke_ijepa|smoke_cv}"
        exit 1
        ;;
esac
