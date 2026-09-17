#!/usr/bin/env bash
set -euo pipefail

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip wheel
pip install -r requirements.txt

# Verify multi-GPU is visible
accelerate env

# Login to W&B (interactive)
wandb login

# Set env vars from template
cp .env.example .env
echo "Edit .env with your WANDB_PROJECT and HF_TOKEN, then re-run."
