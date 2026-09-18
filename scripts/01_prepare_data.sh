#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Export .env into the environment: prepare_gsm8k.py does not load it itself, and
# huggingface_hub reads HF_HOME at import time. Without this the dataset lands in
# ~/.cache instead of the HF_HOME configured in .env.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

python data/prepare_gsm8k.py
ls -lh data/processed/
