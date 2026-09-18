#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

python analysis/aggregate_results.py
python analysis/task_vectors.py
python analysis/format_vs_reasoning.py
python analysis/make_figures.py
