# SFT-GRPO ablation study

Investigates whether an SFT stage before GRPO improves GSM8K accuracy while retaining general capability (MMLU, IFEval, BBH), and how SFT config (LoRA rank, target modules, data quantity) affects that net effect.

## Pipelines

- **P0**: Base → GRPO (no SFT)
- **P1**: SFT only (no GRPO)
- **P2**: SFT → GRPO (two-stage)
- **P3**: SFT → GRPO with matched total trainable params to P0

See [EXPERIMENT.md](EXPERIMENT.md) for the question, hypothesis, and eval details.

## Layout

- `src/`: entrypoints, each run with `python -m`: `train_sft.py`, `train_grpo.py`, `config.py`, `merge_adapter.py`, `eval_model.py`, `utils/`
- `configs/`: accelerate + eval yaml
- `scripts/`: numbered pipeline scripts (00 setup → 06 analyze), `run_all.sh` chains them
- `data/prepare_gsm8k.py`: builds `data/processed/`
- `analysis/`: aggregation and figure scripts
- `results/raw/`: trained adapters; `results/evals/` : eval output (runtime)

## Setup and data

```bash
cp .env.example .env   # WANDB_PROJECT, WANDB_ENTITY, HF_TOKEN, HF_HOME
bash scripts/00_setup.sh        # venv, deps, wandb login
bash scripts/01_prepare_data.sh # build data/processed datasets
```

## Smoke test (1 GPU)

```bash
accelerate launch --config_file configs/accelerate/1gpu.yaml -m src.train_sft \
  --run_name debug_sft --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --lora_r 4 --lora_target_preset attn --sft_dose 500 --num_epochs 1

accelerate launch --config_file configs/accelerate/1gpu.yaml -m src.train_grpo \
  --run_name debug_grpo --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --lora_r 4 --lora_target_preset attn --init_adapter_path results/raw/debug_sft \
  --learning_rate 1e-6 --num_epochs 1 --per_device_batch_size 1

python src/merge_adapter.py --adapter_path results/raw/debug_grpo --output_path results/evals/debug_grpo_merged
python src/eval_model.py --model_path results/evals/debug_grpo_merged --run_name debug_grpo
```

## Full sweeps (4 GPU FSDP)

```bash
bash scripts/02_run_p0_grpo_only.sh
bash scripts/03_run_p1_sft_only.sh
bash scripts/04_run_p2_sft_grpo.sh
bash scripts/05_eval_all.sh
bash scripts/06_analyze.sh
```

## Invocation pattern

Entrypoints are plain `argparse` CLIs; flags are flat (no `--` separator). Shared
identity/model/LoRA flags come from `add_run_config()` in `src/config.py` and sit at the
same level as each entrypoint's stage flags.

```bash
python -m src.train_sft --run_name X --pipeline P1 --lora_r 16 --sft_dose 500 --learning_rate 2e-4
```

## Conventions

- Python 3.13, pinned deps (torch 2.5.1, trl 0.13, peft 0.13, transformers 4.46).
- Run via `python -m` / `accelerate launch -m`.
- bf16 + gradient checkpointing throughout; flash attention in GRPO.
- LoRA `bias="none"`, `lora_alpha = 2 * lora_r`. Seed 42 everywhere.
- GSM8K format: `thinking... response` reasoning tags + `<answer>...</answer>`.
- Eval: GSM8K exact + format (target); MMLU/IFEval/BBH (retention).
