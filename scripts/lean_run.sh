#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Export .env before Python starts: huggingface_hub resolves HF_HOME at import
# time, so a Python-side load_dotenv() is too late for the model/dataset cache.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
ACCEL="configs/accelerate/1gpu.yaml"
GRPO_DATA="grpo_train_small.jsonl"   # 1500 prompts
GRPO_STEPS=300                       # cap GRPO cost

# Treat a run as complete only if it actually produced an adapter. A bare
# output directory is created before the config is built, so a run that died on
# a bad flag would otherwise look "done" and be silently skipped.
have_run() { [[ -f "results/raw/$1/adapter_config.json" ]]; }

for DOSE in 500 5000 50000; do
  # --- P1: SFT only ---
  P1_RUN="P1_sft_dose${DOSE}_r16_attn"
  if ! have_run "$P1_RUN"; then
    echo "=== P1 SFT: $P1_RUN ==="
    accelerate launch --config_file "$ACCEL" -m src.train_sft \
      --run_name "$P1_RUN" --pipeline P1 --stage sft \
      --model_name "$MODEL" --lora_r 16 --lora_alpha 32 \
      --lora_target_preset attn \
      --sft_dose "$DOSE" --learning_rate 2e-4 --num_epochs 1
  fi

  # --- P2: GRPO continuing the SFT adapter ---
  P2_RUN="P2_grpo_dose${DOSE}_r16_attn"
  if ! have_run "$P2_RUN"; then
    echo "=== P2 GRPO: $P2_RUN ==="
    accelerate launch --config_file "$ACCEL" -m src.train_grpo \
      --run_name "$P2_RUN" --pipeline P2 --stage grpo \
      --model_name "$MODEL" --lora_r 16 --lora_alpha 32 \
      --lora_target_preset attn \
      --init_adapter_path "results/raw/$P1_RUN" \
      --learning_rate 1e-6 --num_epochs 1 \
      --grpo_data_file "$GRPO_DATA" --max_steps "$GRPO_STEPS"
  fi
done

# --- P0: GRPO baseline, no SFT ---
P0_RUN="P0_grpo_r16_attn"
if ! have_run "$P0_RUN"; then
  echo "=== P0 GRPO: $P0_RUN ==="
  accelerate launch --config_file "$ACCEL" -m src.train_grpo \
    --run_name "$P0_RUN" --pipeline P0 --stage grpo \
    --model_name "$MODEL" --lora_r 16 --lora_alpha 32 \
    --lora_target_preset attn \
    --learning_rate 1e-6 --num_epochs 1 \
    --grpo_data_file "$GRPO_DATA" --max_steps "$GRPO_STEPS"
fi

echo "Training complete. Run scripts/05_eval_all.sh (or eval the GRPO runs directly)."
