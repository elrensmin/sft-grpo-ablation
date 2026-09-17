#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python data/prepare_gsm8k.py
ls -lh data/processed/
