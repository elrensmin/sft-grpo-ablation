#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

MODEL="Qwen/Qwen2.5-3B-Instruct"
GPUS=4

# P0a: low rank, attn-only
accelerate launch \
  --config_file configs/accelerate/4gpu_fsdp.yaml \
  --num_processes $GPUS \
  -m src.train_grpo \
  --run_name P0_grpo_r16_attn_lr1e-6 \
  --pipeline P0 --stage grpo \
  --model_name $MODEL \
  --lora_r 16 --lora_alpha 32 \
  --lora_target_preset attn \
  --wandb_tags '["P0","baseline"]' \
  --learning_rate 1e-6

# P0b: higher LR (RL typically needs lower LR than SFT)
accelerate launch \
  --config_file configs/accelerate/4gpu_fsdp.yaml \
  --num_processes $GPUS \
  -m src.train_grpo \
  --run_name P0_grpo_r16_attn_lr5e-6 \
  --pipeline P0 --stage grpo \
  --model_name $MODEL \
  --lora_r 16 --lora_alpha 32 \
  --lora_target_preset attn \
  --learning_rate 5e-6
