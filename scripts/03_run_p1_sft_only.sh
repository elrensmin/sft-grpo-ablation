#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Export .env before Python starts: huggingface_hub resolves HF_HOME at import
# time, so a Python-side load_dotenv() is too late for the model/dataset cache.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

MODEL="Qwen/Qwen2.5-3B-Instruct"
GPUS=4

# Dose ladder x (rank, target)
for DOSE in 500 5000 50000; do
  for R in 4 16 64; do
    for TARGET in attn mlp all; do
      RUN="P1_sft_dose${DOSE}_r${R}_${TARGET}"
      echo "=== $RUN ==="
      accelerate launch \
        --config_file configs/accelerate/4gpu_fsdp.yaml \
        --num_processes $GPUS \
        -m src.train_sft \
        --run_name "$RUN" \
        --pipeline P1 --stage sft \
        --model_name "$MODEL" \
        --lora_r $R --lora_alpha $((R*2)) \
        --lora_target_preset $TARGET \
        --sft_dose $DOSE \
        --learning_rate 2e-4 \
        --num_epochs 1
    done
  done
done
