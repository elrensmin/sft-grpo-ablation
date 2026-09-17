#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

EVAL_DIR="results/evals"
mkdir -p "$EVAL_DIR"

# For every trained adapter in results/raw, merge and evaluate
for RUN_DIR in results/raw/*/; do
  RUN_NAME=$(basename "$RUN_DIR")
  MERGED="$EVAL_DIR/${RUN_NAME}_merged"

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
