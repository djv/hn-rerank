#!/usr/bin/env bash
set -euo pipefail

cd /home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
mkdir -p "$UV_CACHE_DIR"

BASE_WARM="runs/stage4_full_expand_seed70"
RUN_ROOT="runs"
CAT1="$RUN_ROOT/cat1_relevance_seed130"
CAT2="$RUN_ROOT/cat2_freshness_seed131"
CAT3="$RUN_ROOT/cat3_semantic_seed132"
CAT4="$RUN_ROOT/cat4_hn_seed133"
FINAL_DIR="$RUN_ROOT/cat_final_full_seed134"

mkdir -p "$CAT1" "$CAT2" "$CAT3" "$CAT4" "$FINAL_DIR"
for d in "$CAT1" "$CAT2" "$CAT3" "$CAT4" "$FINAL_DIR"; do
  rm -f "$d"/optuna_*.json "$d"/optuna_*.log "$d"/live.log
done

run_one() {
  local space="$1"
  local trials="$2"
  local seed="$3"
  local log_dir="$4"
  local warm_dir="$5"
  echo "[run] space=$space seed=$seed trials=$trials warm=$warm_dir log_dir=$log_dir at $(date -Is)" | tee -a runs/category_pipeline.log
  PYTHONUNBUFFERED=1 stdbuf -oL -eL timeout -k 30s 10800s \
    uv run optimize_hyperparameters.py pure_coder \
      --space "$space" \
      --trials "$trials" \
      --cv-folds 5 \
      --n-jobs 4 \
      --cache-only \
      --candidates 500 \
      --seed "$seed" \
      --log-dir "$log_dir" \
      --warm-start-log-dir "$warm_dir" \
    2>&1 | tee "$log_dir/live.log" | tee -a runs/category_pipeline.log
}

: > runs/category_pipeline.log

echo "[pipeline] category sweeps start $(date -Is)" | tee -a runs/category_pipeline.log
run_one cat_relevance 250 130 "$CAT1" "$BASE_WARM"
run_one cat_freshness 200 131 "$CAT2" "$BASE_WARM"
run_one cat_semantic 200 132 "$CAT3" "$BASE_WARM"
run_one cat_hn 200 133 "$CAT4" "$BASE_WARM"

python3 scripts/build_final_enqueue.py --runs-root runs --out runs/category_enqueue_candidates.json --top-k 40 --cv-folds 5 | tee -a runs/category_pipeline.log

echo "[pipeline] final full refine start $(date -Is)" | tee -a runs/category_pipeline.log
PYTHONUNBUFFERED=1 stdbuf -oL -eL timeout -k 30s 14400s \
  uv run optimize_hyperparameters.py pure_coder \
    --space full \
    --trials 400 \
    --cv-folds 5 \
    --n-jobs 4 \
    --cache-only \
    --candidates 500 \
    --seed 134 \
    --log-dir "$FINAL_DIR" \
    --warm-start-log-dir "$BASE_WARM" \
    --enqueue-params-file runs/category_enqueue_candidates.json \
  2>&1 | tee "$FINAL_DIR/live.log" | tee -a runs/category_pipeline.log

echo "[pipeline] done $(date -Is)" | tee -a runs/category_pipeline.log
