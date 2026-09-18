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
cp .env.example .env            # WANDB_PROJECT, WANDB_ENTITY, HF_TOKEN, HF_HOME
bash scripts/00_setup.sh        # Python 3.12 venv + deps from requirements.lock
bash scripts/01_prepare_data.sh # build data/processed datasets
```

`scripts/00_setup.sh` installs the frozen `requirements.lock` by default, so any machine
reproduces the verified set. To deliberately move to the newest compatible versions instead,
run `SETUP_UNPINNED=1 bash scripts/00_setup.sh` and then re-freeze:

```bash
uv pip freeze > requirements.lock
```

## Replicating on a remote GPU box

End to end from a fresh clone:

```bash
git clone https://github.com/elrensmin/sft-grpo-ablation.git
cd sft-grpo-ablation

# uv, the package manager used for the venv and lock install
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cp .env.example .env            # WANDB_PROJECT, WANDB_ENTITY, HF_TOKEN, HF_HOME
bash scripts/00_setup.sh        # Python 3.12 venv + requirements.lock
bash scripts/01_prepare_data.sh

bash scripts/lean_run.sh        # reduced sweep: P1 -> P2, plus P0
bash scripts/05_eval_all.sh     # merge + eval every P0-P3 run
bash scripts/06_analyze.sh      # aggregate table + figures
```

To build the environment by hand instead of via `00_setup.sh`:

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.lock
```

Notes:

- Python **3.12** is required. The lock was frozen against it, and `.python-version` plus
  `pyproject.toml` agree, so `uv` will not silently build a different interpreter.
- Point `HF_HOME` / `TRANSFORMERS_CACHE` at a roomy disk in `.env` — the sweep downloads
  several Qwen2.5 models plus GSM8K/MMLU/IFEval/BBH. The example uses `/scratch/$USER/hf_cache`.
- `lean_run.sh` is resumable: it skips any run that already has an `adapter_config.json`.
- Eval defaults to vLLM over all 1319 GSM8K examples plus the three retention benchmarks.
  For a fast check that touches only GSM8K, bypass vLLM:

  ```bash
  python src/eval_model.py --model_path results/evals/<run>_merged \
    --run_name <run> --no-use_vllm --gsm8k_samples 20
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

- Python 3.12, deps frozen in `requirements.lock` (torch 2.13, transformers 5.17, trl 1.13,
  peft 0.21, vllm 0.28). `requirements.txt` is deliberately unpinned for upgrades; install the
  lock to reproduce a verified set.
- Run via `python -m` / `accelerate launch -m`.
- bf16 + gradient checkpointing throughout. GRPO defaults to `--attn_implementation auto`:
  `flash_attention_2` when `flash_attn` is installed and the GPU is Ampere+, otherwise `sdpa`.
  `flash_attn` is not part of the lock, so install it separately to enable flash attention.
- LoRA `bias="none"`, `lora_alpha = 2 * lora_r`. Seed 42 everywhere.
- GSM8K format: `&lt;think&gt;...&lt;/think&gt;` reasoning tags + `&lt;answer&gt;...&lt;/answer&gt;`.
- Eval: GSM8K exact + format (target); MMLU/IFEval/BBH (retention).
