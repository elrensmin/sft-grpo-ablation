# AGENTS.md

## Project

SFT-GRPO ablation study on Qwen2.5 models. Investigates whether an SFT stage before GRPO improves GSM8K target accuracy while retaining general capability (MMLU, IFEval, BBH), and how SFT config (LoRA rank, target modules, data quantity) affects that net effect.

Four pipelines:
- **P0**: Base → GRPO (no SFT)
- **P1**: SFT only (no GRPO)
- **P2**: SFT → GRPO (two-stage)
- **P3**: SFT → GRPO with matched total trainable params to P0

## Layout

- `src/` : training/eval entrypoints, each run with `python -m`
  - `train_sft.py` : SFT stage (trl `SFTTrainer`, LoRA via peft)
  - `train_grpo.py` : GRPO stage (trl `GRPOTrainer`, LoRA, can continue from an existing adapter via `--init_adapter_path`)
  - `config.py` : shared `BaseRunConfig` dataclass + `TARGET_MODULE_PRESETS` (`attn`/`mlp`/`all`)
  - `merge_adapter.py` : merge LoRA adapter into base model
  - `eval_model.py` : GSM8K (vLLM, exact + format) and lm-eval-harness (mmlu/ifeval/bbh)
  - `utils/wandb_setup.py` : W&B init wrapper
- `configs/` : accelerate (1gpu, 4gpu FSDP) and eval benchmark yaml
- `scripts/` : numbered pipeline scripts (00 setup → 06 analyze), `run_all.sh` chains them
- `data/prepare_gsm8k.py` : builds `data/processed/` (sft doses 500/5000/50000, grpo train/test)
- `analysis/` : result aggregation (`aggregate_results.py`), task-vector, format-vs-reasoning, figure scripts
- `results/raw/` : trained adapters; `results/evals/` : eval output (created at runtime)

## Commands

Setup and data:
```bash
bash scripts/00_setup.sh        # venv, deps, wandb login, .env
bash scripts/01_prepare_data.sh # build data/processed datasets
```

Smoke test (1 GPU, tiny configs):
```bash
accelerate launch --config_file configs/accelerate/1gpu.yaml -m src.train_sft \
  --run_name debug_sft --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --lora_r 4 --lora_target_preset attn -- --sft_dose 500 --num_epochs 1

accelerate launch --config_file configs/accelerate/1gpu.yaml -m src.train_grpo \
  --run_name debug_grpo --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --lora_r 4 --lora_target_preset attn --init_adapter_path results/raw/debug_sft \
  -- --learning_rate 1e-6 --num_epochs 1 --per_device_batch_size 1

python src/merge_adapter.py --adapter_path results/raw/debug_grpo --output_path results/evals/debug_grpo_merged
python src/eval_model.py --model_path results/evals/debug_grpo_merged --run_name debug_grpo
```

Full sweeps (4 GPU FSDP):
```bash
bash scripts/02_run_p0_grpo_only.sh   # P0 GRPO baselines
bash scripts/03_run_p1_sft_only.sh    # P1: dose x {r 4,16,64} x {attn,mlp,all}
bash scripts/04_run_p2_sft_grpo.sh    # P2: dose x {r 4,16,64} attn
bash scripts/05_eval_all.sh           # merge + eval every run in results/raw
bash scripts/06_analyze.sh             # aggregation + figures
```

## Invocation pattern

CLI args split at `--`: flags before it are tyro `BaseRunConfig` fields, after it are the entrypoint's own params. Example:
```bash
python -m src.train_sft --run_name X --pipeline P1 --lora_r 16 \
  -- --sft_dose 500 --learning_rate 2e-4
```

## Environment

Copy `.env.example` to `.env`: `WANDB_PROJECT`, `WANDB_ENTITY`, `HF_TOKEN`, `HF_HOME`, `TRANSFORMERS_CACHE`. `src/utils/wandb_setup.py` loads it via python-dotenv. Never commit `.env` or `results/`.

## Conventions

- Python 3.13, dependencies pinned in `requirements.txt` (torch 2.5.1, trl 0.13, peft 0.13, transformers 4.46).
- Run via `python -m` and `accelerate launch -m`, not `python file.py`.
- bf16 + gradient checkpointing throughout; flash attention in GRPO.
- LoRA `bias="none"`, `lora_alpha = 2 * lora_r` in pipeline scripts.
- Seed 42 everywhere.
- GSM8K answer format: ` thinking... response` reasoning tags + `<answer>...</answer>`.
- Eval: GSM8K exact match + format correctness (target), MMLU/IFEval/BBH (retention).
- Training artifacts → `results/raw/<run_name>`; evals → `results/evals/<run_name>/`.
