#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GOAL_ROOT="$ROOT_DIR/artifacts/spheredyn_spherettc_open_goal_20260801"
RUN_ROOT="$GOAL_ROOT/main_prediction_v2_seed44"
QUEUE_ROOT="$GOAL_ROOT/main_prediction_v2_4gpu_queue"
CHECKPOINT="$GOAL_ROOT/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt"
STATS="$RUN_ROOT/frozen_inputs/s2s_daily_54var_stats.json"
REFERENCE_ROOT="$ROOT_DIR/artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache"
DATA_ROOT="$ROOT_DIR/data/S2S"
DAILY_CACHE="/tmp/spheredyn_development_daily_cache_1979_2022"
EXISTING_2018="$GOAL_ROOT/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/caches/spheredyn_v9_multiscale_seed44_2018.zarr"
mkdir -p "$RUN_ROOT"/{raw_fragments,profiles,logs,final} "$QUEUE_ROOT"
DAILY_CACHE_ARGS=()
if [[ -f "$DAILY_CACHE/MANIFEST.json" ]]; then
  DAILY_CACHE_ARGS=(--daily-cache "$DAILY_CACHE")
fi

exec 9>"$QUEUE_ROOT/QUEUE.lock"
if ! flock -n 9; then echo "Another v2 main-prediction queue holds the lease." >&2; exit 9; fi
if [[ $(/usr/bin/python -c 'import torch; print(torch.cuda.device_count())') -lt 4 ]]; then
  echo "Four visible CUDA GPUs are required for this launcher." >&2; exit 5
fi
if [[ ! -f "$CHECKPOINT" || ! -f "$EXISTING_2018/.zmetadata" ]]; then
  echo "Missing frozen v9 checkpoint or 2018 cache." >&2; exit 6
fi
if [[ $(sha256sum "$CHECKPOINT" | awk '{print $1}') != "121f9b900db554f175f121e629ac64a10ce2697e7cf4bfd224f09984c6bbf453" ]]; then
  echo "Frozen checkpoint hash mismatch." >&2; exit 7
fi

date -u +%FT%TZ >"$QUEUE_ROOT/QUEUE_STARTED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >"$QUEUE_ROOT/GPU_INVENTORY.txt"

infer_fragment() {
  local gpu="$1" tag="$2" start="$3" end="$4"
  local output="$RUN_ROOT/raw_fragments/${tag}.zarr"
  if [[ -f "$output/.zmetadata" ]]; then
    echo "resume: $tag already complete"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/python scripts/spheredyn/infer_spatiotemporal_checkpoint.py \
    --checkpoint "$CHECKPOINT" --data "$DATA_ROOT" --stats "$STATS" "${DAILY_CACHE_ARGS[@]}" \
    --init-start "$start" --init-end-exclusive "$end" --batch-size 1 --io-workers 8 \
    --output "$output" --profile-output "$RUN_ROOT/profiles/${tag}.json" \
    >"$RUN_ROOT/logs/${tag}.log" 2>&1
}

pids=()
infer_fragment 0 v9_seed44_2017_h1 2017-01-01 2017-07-02 & pids+=("$!")
infer_fragment 1 v9_seed44_2017_h2 2017-07-02 2018-01-01 & pids+=("$!")
infer_fragment 2 v9_seed44_2019_h1 2019-01-01 2019-07-02 & pids+=("$!")
infer_fragment 3 v9_seed44_2019_h2 2019-07-02 2020-01-01 & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then echo "At least one main inference fragment failed." >&2; exit 8; fi
date -u +%FT%TZ >"$QUEUE_ROOT/RAW_INFERENCE_COMPLETED"

CUDA_VISIBLE_DEVICES=0 /usr/bin/python scripts/spheredyn/run_main_spherettc_v2.py \
  --fit-backbone-fragment "$RUN_ROOT/raw_fragments/v9_seed44_2017_h1.zarr" \
  --fit-backbone-fragment "$RUN_ROOT/raw_fragments/v9_seed44_2017_h2.zarr" \
  --eval-backbone-fragment "2018=$EXISTING_2018" \
  --eval-backbone-fragment "2019=$RUN_ROOT/raw_fragments/v9_seed44_2019_h1.zarr" \
  --eval-backbone-fragment "2019=$RUN_ROOT/raw_fragments/v9_seed44_2019_h2.zarr" \
  --reference-root "$REFERENCE_ROOT" --stats "$STATS" \
  --minimum-backbone-weight 0.5 --batch-size 1 --output-root "$RUN_ROOT/final" \
  >"$RUN_ROOT/logs/main_spherettc.log" 2>&1

/usr/bin/python scripts/spheredyn/assess_main_prediction_goal.py \
  --raw "$RUN_ROOT/final/RAW_METRICS.npz" \
  --final "$RUN_ROOT/final/FINAL_METRICS.npz" \
  --graphcast "$RUN_ROOT/final/GRAPHCAST_METRICS.npz" \
  --stats "$STATS" --seed 44 --minimum-gain-percent 5 \
  --output "$RUN_ROOT/MAIN_GOAL_GATE.json" \
  >"$RUN_ROOT/logs/main_goal_gate.log" 2>&1
date -u +%FT%TZ >"$RUN_ROOT/MAIN_EXPERIMENT_COMPLETED"
date -u +%FT%TZ >"$QUEUE_ROOT/QUEUE_COMPLETED"
echo "MAIN_PREDICTION_V2_COMPLETE $RUN_ROOT"
