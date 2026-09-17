#!/usr/bin/env bash
set -euo pipefail
bash scripts/00_setup.sh
bash scripts/01_prepare_data.sh
bash scripts/02_run_p0_grpo_only.sh
bash scripts/03_run_p1_sft_only.sh
bash scripts/04_run_p2_sft_grpo.sh
bash scripts/05_eval_all.sh
bash scripts/06_analyze.sh
