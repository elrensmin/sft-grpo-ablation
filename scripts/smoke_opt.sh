#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Export .env first so the explicit WANDB_MODE=disabled below still wins, and so
# HF_HOME is set before huggingface_hub resolves it at import time.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

export WANDB_MODE=disabled

MODEL="facebook/opt-125m"
ACCEL="configs/accelerate/1gpu.yaml"

# --- SFT ---
accelerate launch --config_file "$ACCEL" -m src.train_sft \
  --run_name smoke_sft --pipeline P2 --stage sft \
  --model_name "$MODEL" \
  --lora_r 4 --lora_alpha 8 --lora_target_preset attn \
  --sft_dose 500 --learning_rate 2e-4 --num_epochs 1 \
  --per_device_batch_size 1 --grad_accum 1

# --- GRPO (continues the SFT adapter) ---
# grad_accum is 4 so that generation_batch_size (1 * 4 * 1 gpu) is divisible by
# num_generations=4, which trl enforces in GRPOConfig.__post_init__.
accelerate launch --config_file "$ACCEL" -m src.train_grpo \
  --run_name smoke_grpo --pipeline P2 --stage grpo \
  --model_name "$MODEL" \
  --lora_r 4 --lora_alpha 8 --lora_target_preset attn \
  --init_adapter_path results/raw/smoke_sft \
  --learning_rate 1e-6 --num_epochs 1 \
  --per_device_batch_size 1 --grad_accum 4 \
  --num_generations 4 --max_completion_length 128 \
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
  --no-use_vllm --gsm8k_samples 20

echo "Smoke test complete."
