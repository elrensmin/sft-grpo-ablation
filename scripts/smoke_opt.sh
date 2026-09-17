#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

export WANDB_MODE=disabled

MODEL="facebook/opt-125m"
ACCEL="configs/accelerate/1gpu.yaml"

# --- SFT ---
accelerate launch --config_file "$ACCEL" -m src.train_sft \
  --run_name smoke_sft --pipeline P2 --stage sft \
  --model_name "$MODEL" \
  --lora_r 4 --lora_alpha 8 --lora_target_preset attn -- \
  --sft_dose 500 --learning_rate 2e-4 --num_epochs 1 \
  --per_device_batch_size 1 --grad_accum 1

# --- GRPO (continues the SFT adapter) ---
accelerate launch --config_file "$ACCEL" -m src.train_grpo \
  --run_name smoke_grpo --pipeline P2 --stage grpo \
  --model_name "$MODEL" \
  --lora_r 4 --lora_alpha 8 --lora_target_preset attn \
  --init_adapter_path results/raw/smoke_sft -- \
  --learning_rate 1e-6 --num_epochs 1 \
  --per_device_batch_size 1 --grad_accum 1 \
  --num_generations 4 --max_prompt_length 256 --max_completion_length 128 \
  --grpo_data_file grpo_train_small.jsonl --max_steps 20 \
  --attn_implementation sdpa

# --- Merge ---
python src/merge_adapter.py \
  --adapter_path results/raw/smoke_grpo \
  --output_path results/evals/smoke_grpo_merged

# --- Eval (HF greedy, small sample) ---
python src/eval_model.py \
  --model_path results/evals/smoke_grpo_merged \
  --run_name smoke_grpo \
  --use_vllm False --gsm8k_samples 20

echo "Smoke test complete."
