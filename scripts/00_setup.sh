#!/usr/bin/env bash
set -euo pipefail

# Create venv and install deps via uv. --allow-existing keeps this idempotent:
# re-running setup reuses the venv instead of erroring out.
uv venv .venv --python 3.12 --allow-existing
source .venv/bin/activate

# Install the frozen, verified set by default so a remote box reproduces local
# results. Set SETUP_UNPINNED=1 to resolve the newest compatible versions from
# requirements.txt instead (then re-freeze with: uv pip freeze > requirements.lock).
if [[ "${SETUP_UNPINNED:-0}" == "1" || ! -f requirements.lock ]]; then
  echo "[setup] installing unpinned requirements.txt (newest compatible versions)"
  uv pip install -r requirements.txt
else
  echo "[setup] installing frozen requirements.lock (verified set)"
  uv pip install -r requirements.lock
fi

# Verify GPU is visible
accelerate env

# Local runs: W&B optional. Set WANDB_MODE=disabled to run offline.
echo "If you don't have W&B set up, run:"
echo "  export WANDB_MODE=disabled"
echo "and create .env if you need WANDB_PROJECT / HF_TOKEN."
