#!/usr/bin/env bash
set -euo pipefail

cd /home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
mkdir -p "$UV_CACHE_DIR"

FINAL_DIR="runs/final_aggregate_seed123"
mkdir -p "$FINAL_DIR"
rm -f "$FINAL_DIR"/optuna_*.json "$FINAL_DIR"/optuna_*.log "$FINAL_DIR"/live.log

echo "[final] waiting for stage56 to finish at $(date -Is)"
while ps -u dev -o cmd= | rg -q "optimize_hyperparameters.py pure_coder .*--seed (80|90|91)"; do
  sleep 20
done

echo "[final] start aggregate sweep at $(date -Is)"
PYTHONUNBUFFERED=1 stdbuf -oL -eL timeout -k 30s 14400s \
  uv run optimize_hyperparameters.py pure_coder \
    --space full \
    --trials 500 \
    --cv-folds 5 \
    --n-jobs 4 \
    --cache-only \
    --candidates 500 \
    --seed 123 \
    --log-dir "$FINAL_DIR" \
    --warm-start-log-dir runs/stage3_core_refine_seed60 \
    --enqueue-params-file runs/final_enqueue_candidates.json \
  2>&1 | tee "$FINAL_DIR/live.log"

echo "[final] done at $(date -Is)"
