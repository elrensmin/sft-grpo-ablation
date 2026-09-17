#!/usr/bin/env bash
set -euo pipefail

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip wheel
pip install -r requirements.txt

# Verify GPU is visible
accelerate env

# Local runs: W&B optional. Set WANDB_MODE=disabled to run offline.
echo "If you don't have W&B set up, run:"
echo "  export WANDB_MODE=disabled"
echo "and create .env if you need WANDB_PROJECT / HF_TOKEN."
