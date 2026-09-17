#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

MODEL="Qwen/Qwen2.5-3B-Instruct"
GPUS=4

# P2: SFT (small dose) then GRPO (continuing adapter)
for DOSE in 500 5000 50000; do
  for R in 4 16 64; do
    SFT_RUN="P2_sft_dose${DOSE}_r${R}_attn"
    SFT_DIR="results/raw/$SFT_RUN"
    GRPO_RUN="P2_grpo_dose${DOSE}_r${R}_attn"

    if [[ ! -d "$SFT_DIR" ]]; then
      echo "=== SFT stage: $SFT_RUN ==="
      accelerate launch \
        --config_file configs/accelerate/4gpu_fsdp.yaml \
        --num_processes $GPUS \
        -m src.train_sft \
        --run_name "$SFT_RUN" \
        --pipeline P2 --stage sft \
        --model_name "$MODEL" \
        --lora_r $R --lora_alpha $((R*2)) \
        --lora_target_preset attn \
        -- \
        --sft_dose $DOSE --learning_rate 2e-4 --num_epochs 1
    fi

    echo "=== GRPO stage: $GRPO_RUN ==="
    accelerate launch \
      --config_file configs/accelerate/4gpu_fsdp.yaml \
      --num_processes $GPUS \
      -m src.train_grpo \
      --run_name "$GRPO_RUN" \
      --pipeline P2 --stage grpo \
      --model_name "$MODEL" \
      --lora_r $R --lora_alpha $((R*2)) \
      --lora_target_preset attn \
      --init_adapter_path "$SFT_DIR" \
      -- \
      --learning_rate 1e-6 --num_epochs 1
  done
done
