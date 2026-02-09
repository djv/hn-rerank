#!/usr/bin/env bash
set -euo pipefail

cd /home/dev/hn_rerank
export UV_CACHE_DIR=/home/dev/hn_rerank/.cache/uv
mkdir -p "$UV_CACHE_DIR"

RUN_ROOT="runs"
STAGE3_A="$RUN_ROOT/stage3_core_refine_seed60"
STAGE3_B="$RUN_ROOT/stage3_core_refine_seed61"
STAGE4_A="$RUN_ROOT/stage4_full_expand_seed70"
STAGE4_B="$RUN_ROOT/stage4_full_expand_seed71"
STAGE5="$RUN_ROOT/stage5_full_exploit_seed80"
STAGE6_A="$RUN_ROOT/stage6_robustness_seed90"
STAGE6_B="$RUN_ROOT/stage6_robustness_seed91"
SUMMARY_JSON="$RUN_ROOT/multistage_summary.json"

mkdir -p "$STAGE3_A" "$STAGE3_B" "$STAGE4_A" "$STAGE4_B" "$STAGE5" "$STAGE6_A" "$STAGE6_B"

# Clean stale artifacts that can affect range narrowing and visibility
for d in "$STAGE3_A" "$STAGE3_B" "$STAGE4_A" "$STAGE4_B" "$STAGE5" "$STAGE6_A" "$STAGE6_B"; do
  rm -f "$d"/optuna_*.json "$d"/optuna_*.log "$d"/live.log
done

run_sweep() {
  local space="$1"
  local trials="$2"
  local cv="$3"
  local seed="$4"
  local log_dir="$5"
  local timeout_s="$6"
  local warm_start_dir="${7:-}"
  local live_log="$log_dir/live.log"

  echo "[run] start space=$space seed=$seed trials=$trials cv=$cv n_jobs=4 timeout=${timeout_s}s log_dir=$log_dir warm_start='${warm_start_dir:-none}' at $(date -Is)"
  set +e
  local cmd=(
    uv run optimize_hyperparameters.py pure_coder
    --space "$space"
    --trials "$trials"
    --cv-folds "$cv"
    --n-jobs 4
    --cache-only
    --candidates 500
    --seed "$seed"
    --log-dir "$log_dir"
  )
  if [[ -n "$warm_start_dir" ]]; then
    cmd+=(--warm-start-log-dir "$warm_start_dir")
  fi
  PYTHONUNBUFFERED=1 stdbuf -oL -eL timeout -k 30s "${timeout_s}s" \
    "${cmd[@]}" \
    2>&1 | tee "$live_log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[run] FAILED space=$space seed=$seed rc=$rc"
    return $rc
  fi
  echo "[run] done space=$space seed=$seed at $(date -Is)"
}

latest_json_score() {
  local dir="$1"
  python3 - <<PY
import glob, json, math
p = sorted(glob.glob("$dir/optuna_*.json"))
if not p:
    print("nan")
    raise SystemExit
best = float("nan")
for fp in p:
    try:
        with open(fp) as f:
            d = json.load(f)
    except Exception:
        continue
    v = d.get("best_score")
    if isinstance(v, (int, float)):
        best = float(v)
print(best)
PY
}

# Stage 3: core refinement (serial)
echo "[stage3] seed 60 then 61 (serial)"
run_sweep core 300 5 60 "$STAGE3_A" 7200
run_sweep core 300 5 61 "$STAGE3_B" 7200
echo "[stage3] complete"

s3a_score=$(latest_json_score "$STAGE3_A")
s3b_score=$(latest_json_score "$STAGE3_B")
if python3 - <<PY
a=float("$s3a_score")
b=float("$s3b_score")
raise SystemExit(0 if a >= b else 1)
PY
then
  stage3_warm_dir="$STAGE3_A"
else
  stage3_warm_dir="$STAGE3_B"
fi
echo "[stage3] warm-start source for stage4: $stage3_warm_dir"

# Stage 4: full expansion (serial)
echo "[stage4] seed 70 then 71 (serial)"
run_sweep full 250 5 70 "$STAGE4_A" 7200 "$stage3_warm_dir"
run_sweep full 250 5 71 "$STAGE4_B" 7200 "$stage3_warm_dir"
echo "[stage4] complete"

s4a_score=$(latest_json_score "$STAGE4_A")
s4b_score=$(latest_json_score "$STAGE4_B")
if python3 - <<PY
a=float("$s4a_score")
b=float("$s4b_score")
raise SystemExit(0 if a >= b else 1)
PY
then
  stage4_warm_dir="$STAGE4_A"
else
  stage4_warm_dir="$STAGE4_B"
fi
echo "[stage4] warm-start source for full-space follow-up: $stage4_warm_dir"

read -r core_best full_best delta winner stage5_needed <<EOF
$(python3 - <<PY
s3a=float("$s3a_score")
s3b=float("$s3b_score")
s4a=float("$s4a_score")
s4b=float("$s4b_score")
core=max(s3a,s3b)
full=max(s4a,s4b)
delta=full-core
winner="full" if delta>=0.01 else "core"
stage5="1" if delta>=0.01 else "0"
print(f"{core} {full} {delta} {winner} {stage5}")
PY)
EOF

echo "[gate] core_best=$core_best full_best=$full_best delta=$delta winner=$winner stage5_needed=$stage5_needed"

winner_ref=""
stage5_score="nan"
if [[ "$stage5_needed" == "1" ]]; then
  echo "[stage5] full exploit seed 80"
  run_sweep full 400 5 80 "$STAGE5" 10800 "$stage4_warm_dir"
  stage5_score=$(latest_json_score "$STAGE5")
  winner_ref="$stage5_score"
else
  winner_ref="$core_best"
fi

if [[ "$winner" == "full" ]]; then
  winner_space="full"
else
  winner_space="core"
fi

# Stage 6: robustness confirmation (serial)
echo "[stage6] winner_space=$winner_space seed 90 then 91"
if [[ "$winner_space" == "full" ]]; then
  if [[ "$stage5_needed" == "1" ]]; then
    stage6_warm_dir="$STAGE5"
  else
    stage6_warm_dir="$stage4_warm_dir"
  fi
else
  stage6_warm_dir="$stage3_warm_dir"
fi
run_sweep "$winner_space" 150 5 90 "$STAGE6_A" 3600 "$stage6_warm_dir"
run_sweep "$winner_space" 150 5 91 "$STAGE6_B" 3600 "$stage6_warm_dir"
echo "[stage6] complete"

s6a_score=$(latest_json_score "$STAGE6_A")
s6b_score=$(latest_json_score "$STAGE6_B")

read -r robust_ok s6a_diff s6b_diff <<EOF
$(python3 - <<PY
ref=float("$winner_ref")
a=float("$s6a_score")
b=float("$s6b_score")
da=abs(a-ref)
db=abs(b-ref)
ok = 1 if (da<=0.02 and db<=0.02) else 0
print(f"{ok} {da} {db}")
PY)
EOF

echo "[result] winner_space=$winner_space winner_ref=$winner_ref stage5_score=$stage5_score"
echo "[result] stage3: seed60=$s3a_score seed61=$s3b_score"
echo "[result] stage4: seed70=$s4a_score seed71=$s4b_score"
echo "[result] stage6: seed90=$s6a_score seed91=$s6b_score robust_ok=$robust_ok diffs=($s6a_diff,$s6b_diff)"

python3 - <<PY
import json
payload = {
  "stage3": {"seed60": float("$s3a_score"), "seed61": float("$s3b_score")},
  "stage4": {"seed70": float("$s4a_score"), "seed71": float("$s4b_score")},
  "decision": {
    "core_best": float("$core_best"),
    "full_best": float("$full_best"),
    "delta_full_minus_core": float("$delta"),
    "winner_space": "$winner_space",
    "stage5_needed": bool(int("$stage5_needed")),
  },
  "stage5": {"seed80": float("$stage5_score") if "$stage5_score" != "nan" else None},
  "stage6": {
    "seed90": float("$s6a_score"),
    "seed91": float("$s6b_score"),
    "robust_ok": bool(int("$robust_ok")),
    "diff_from_ref": {
      "seed90": float("$s6a_diff"),
      "seed91": float("$s6b_diff"),
    },
  },
}
with open("$SUMMARY_JSON", "w") as f:
    json.dump(payload, f, indent=2)
print("wrote", "$SUMMARY_JSON")
PY

echo "[runner] completed $(date -Is)"
