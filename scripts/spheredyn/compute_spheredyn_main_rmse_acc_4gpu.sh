#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

GOAL_ROOT="$ROOT_DIR/artifacts/spheredyn_spherettc_open_goal_20260801"
RUN_ROOT="$GOAL_ROOT/main_prediction_v2_seed44"
METRIC_ROOT="$RUN_ROOT/paper_metrics"
CONFIG="$ROOT_DIR/configs/unified_11model_protocol.yaml"
CLIMATOLOGY="$ROOT_DIR/artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz"
EXISTING_2018="$GOAL_ROOT/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/caches/spheredyn_v9_multiscale_seed44_2018.zarr"
mkdir -p "$METRIC_ROOT"/{fragments,logs,profiles}
mapfile -t COMMON_VARIABLES < <(/usr/bin/python -c 'from src.experiments.protocol import COMMON_EVALUATION_VARIABLES; print("\n".join(COMMON_EVALUATION_VARIABLES))')

run_metric() {
  local gpu="$1" tag="$2" cache="$3"
  CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/python scripts/run_ttc.py \
    --config "$CONFIG" --model spheredyn_v9_seed44 --method raw \
    --cache "$cache" --variables "${COMMON_VARIABLES[@]}" \
    --no-cache-output --acc-climatology "$CLIMATOLOGY" \
    --metrics-output "$METRIC_ROOT/fragments/${tag}.npz" \
    --profile-output "$METRIC_ROOT/profiles/${tag}.json" \
    >"$METRIC_ROOT/logs/${tag}.log" 2>&1
}

pids=()
run_metric 0 raw_2018 "$EXISTING_2018" & pids+=("$!")
run_metric 1 raw_2019_h1 "$RUN_ROOT/raw_fragments/v9_seed44_2019_h1.zarr" & pids+=("$!")
run_metric 2 raw_2019_h2 "$RUN_ROOT/raw_fragments/v9_seed44_2019_h2.zarr" & pids+=("$!")
run_metric 3 final_2018 "$RUN_ROOT/final/final_spheredyn_v9_spherettc_v22_2018.zarr" & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one SphereDyn RMSE/ACC metric fragment failed." >&2
  exit 8
fi

run_metric 0 final_2019 "$RUN_ROOT/final/final_spheredyn_v9_spherettc_v22_2019.zarr"

/usr/bin/python scripts/spheredyn/merge_metric_npz.py \
  --input \
    "$METRIC_ROOT/fragments/raw_2018.npz" \
    "$METRIC_ROOT/fragments/raw_2019_h1.npz" \
    "$METRIC_ROOT/fragments/raw_2019_h2.npz" \
  --output "$METRIC_ROOT/spheredyn_v9_raw_2018_2019.npz"
/usr/bin/python scripts/spheredyn/merge_metric_npz.py \
  --input \
    "$METRIC_ROOT/fragments/final_2018.npz" \
    "$METRIC_ROOT/fragments/final_2019.npz" \
  --output "$METRIC_ROOT/spheredyn_v9_spherettc_v22_2018_2019.npz"
/usr/bin/python scripts/spheredyn/build_spheredyn_main_rmse_acc_table.py
date -u +%FT%TZ >"$METRIC_ROOT/METRICS_COMPLETED"
echo "SPHEREDYN_MAIN_RMSE_ACC_COMPLETE $METRIC_ROOT"
