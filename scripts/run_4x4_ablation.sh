#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="tests/snapshots/baseline_full.json"
FEEDBACK=".cache/user_feedback/dashboard_feedback.json"
OUTDIR="results"
RUN_D_ARGS="--no-drop-one --no-single-features --cv 0"

mkdir -p "$OUTDIR"

# 22 features (current TOML)
F22=(
    centroid,pos_knn,neg_knn,pos_neg_ratio
    log_points,log_comments,comment_ratio
    title_len,text_len,is_github,is_pdf,comments_count
    closest_pos,closest_neg,closest_margin
    is_hn
    source_trust
    local_density
    story_age
    cluster_size
    domain_recency
    embedding_magnitude
)
F22_STR=$(IFS=,; echo "${F22[*]}")

# 16 features (remove log_points, log_comments, comment_ratio, comments_count, story_age, domain_recency)
F16=(
    centroid,pos_knn,neg_knn,pos_neg_ratio
    title_len,text_len,is_github,is_pdf
    closest_pos,closest_neg,closest_margin
    is_hn
    source_trust
    local_density
    cluster_size
    embedding_magnitude
)
F16_STR=$(IFS=,; echo "${F16[*]}")

run_eval() {
    local label="$1"
    local features="$2"
    local raw="$3"
    local model_type="$4"
    local overrides="${5:-}"
    local outfile="$OUTDIR/${label}.json"

    echo ""
    echo "========================================================================"
    echo "  [$label] features=$features model=$model_type raw=$raw overrides=[${overrides}]"
    echo "========================================================================"

    CMD="uv run python scripts/feature_ablation.py '$SNAPSHOT' \
        --feedback '$FEEDBACK' \
        $RUN_D_ARGS \
        --output '$outfile' \
        --features '${label}=${features}'"

    if [ "$model_type" != "default" ]; then
        CMD="$CMD --model-type '$model_type'"
    fi

    if [ "$raw" = "true" ]; then
        CMD="$CMD --raw-embedding-features"
    fi

    if [ -n "$overrides" ]; then
        # shellcheck disable=SC2089
        CMD="$CMD --override $overrides"
    fi

    echo "Running: $CMD"
    # shellcheck disable=SC2090
    eval "$CMD"
}

# ── 16 combos: 4 feature configs × 4 model configs ────────────────────

# Model: mlp-relu (current default)
run_eval "22f-mlp-relu"     "$F22_STR" "false" "mlp" ""
run_eval "22f+raw-mlp-relu" "$F22_STR" "true"  "mlp" ""
run_eval "16f-mlp-relu"     "$F16_STR" "false" "mlp" ""
run_eval "16f+raw-mlp-relu" "$F16_STR" "true"  "mlp" ""

# Model: mlp-sigmoid
run_eval "22f-mlp-sigmoid"     "$F22_STR" "false" "mlp" "mlp_activation=logistic"
run_eval "22f+raw-mlp-sigmoid" "$F22_STR" "true"  "mlp" "mlp_activation=logistic"
run_eval "16f-mlp-sigmoid"     "$F16_STR" "false" "mlp" "mlp_activation=logistic"
run_eval "16f+raw-mlp-sigmoid" "$F16_STR" "true"  "mlp" "mlp_activation=logistic"

# Model: mlp-alpha
run_eval "22f-mlp-alpha"     "$F22_STR" "false" "mlp" "mlp_alpha=0.01"
run_eval "22f+raw-mlp-alpha" "$F22_STR" "true"  "mlp" "mlp_alpha=0.01"
run_eval "16f-mlp-alpha"     "$F16_STR" "false" "mlp" "mlp_alpha=0.01"
run_eval "16f+raw-mlp-alpha" "$F16_STR" "true"  "mlp" "mlp_alpha=0.01"

# Model: svm-rbf
run_eval "22f-svm-rbf"     "$F22_STR" "false" "svm" ""
run_eval "22f+raw-svm-rbf" "$F22_STR" "true"  "svm" ""
run_eval "16f-svm-rbf"     "$F16_STR" "false" "svm" ""
run_eval "16f+raw-svm-rbf" "$F16_STR" "true"  "svm" ""

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  All 16 evals complete. Results in $OUTDIR/"
echo "═══════════════════════════════════════════════════════════════════════"
