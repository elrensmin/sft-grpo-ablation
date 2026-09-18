#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Export .env so HF_HOME/HF_TOKEN reach both the eval entrypoint and the `lm_eval`
# subprocess it spawns (neither loads .env by itself).
if [[ -f .env ]]; then set -a; source .env; set +a; fi

EVAL_DIR="results/evals"
mkdir -p "$EVAL_DIR"

# For every trained adapter in results/raw, merge and evaluate
for RUN_DIR in results/raw/*/; do
  RUN_NAME=$(basename "$RUN_DIR")
  MERGED="$EVAL_DIR/${RUN_NAME}_merged"

  # Only evaluate ablation runs; skip smoke/debug dirs so they don't consume
  # GPU hours (aggregate_results.py filters them out of the table anyway).
  case "$RUN_NAME" in
    P0_*|P1_*|P2_*|P3_*) ;;
    *) echo "[skip] $RUN_NAME (not an ablation run)"; continue ;;
  esac

  # Skip if already merged & evaluated
  if [[ -f "$EVAL_DIR/$RUN_NAME/summary.json" ]]; then
    echo "[skip] $RUN_NAME already evaluated"
    continue
  fi

  if [[ -f "$RUN_DIR/adapter_config.json" ]]; then
    echo "[merge] $RUN_NAME"
    python src/merge_adapter.py \
      --adapter_path "$RUN_DIR" \
      --output_path "$MERGED"
    MODEL_PATH="$MERGED"
  else
    # Full FT / no adapter
    MODEL_PATH="$RUN_DIR"
  fi

  echo "[eval] $RUN_NAME"
  python src/eval_model.py \
    --model_path "$MODEL_PATH" \
    --run_name "$RUN_NAME" \
    --eval_dir "$EVAL_DIR"
done

echo "All evaluations complete."
