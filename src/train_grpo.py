import os
import re
import json
import argparse
import importlib.util
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from src.config import add_run_config, TARGET_MODULE_PRESETS
from src.utils.wandb_setup import init_wandb


def resolve_attn_implementation(requested: str) -> str:
    """Return an attention backend this environment can actually use.

    `flash_attention_2` is the fastest option but needs the optional `flash_attn`
    package *and* an Ampere-or-newer GPU. With `"auto"` we fall back to `sdpa`
    when either is missing so the pipeline also runs on laptop GPUs and CPU
    smoke tests. An explicit choice is always honoured as-is.
    """
    if requested != "auto":
        return requested
    if importlib.util.find_spec("flash_attn") is None:
        print("[attn] flash_attn is not installed -> falling back to sdpa")
        return "sdpa"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        print("[attn] GPU does not support flash_attention_2 -> falling back to sdpa")
        return "sdpa"
    return "flash_attention_2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO stage (pipelines P0/P2/P3)")
    add_run_config(parser)

    g = parser.add_argument_group("grpo")
    g.add_argument("--learning_rate", type=float, default=1e-6)
    g.add_argument("--num_epochs", type=int, default=1)
    g.add_argument("--per_device_batch_size", type=int, default=2)
    g.add_argument("--grad_accum", type=int, default=8)
    g.add_argument("--num_generations", type=int, default=4,
                   help="Completions per prompt; must divide "
                        "per_device_batch_size * grad_accum * num_processes")
    g.add_argument("--max_prompt_length", type=int, default=None,
                   help="Deprecated: trl>=1.0 removed prompt truncation from GRPO, "
                        "so this is accepted for backwards compatibility but ignored")
    g.add_argument("--max_completion_length", type=int, default=512)
    g.add_argument("--beta", type=float, default=0.0, help="KL penalty to the reference model")
    g.add_argument("--use_format_shaping", action=argparse.BooleanOptionalAction,
                   default=True, help="Add the format reward to the exact-match reward")
    g.add_argument("--grpo_data_file", default="grpo_train_small.jsonl",
                   help="File under data/processed/ to train on")
    g.add_argument("--max_steps", type=int, default=-1,
                   help="Cap the number of optimizer steps; <= 0 means no cap")
    g.add_argument("--attn_implementation", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="'auto' picks flash_attention_2 when flash-attn is installed "
                        "and the GPU supports it, otherwise sdpa")
    return parser


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(.+)", text)
    return m.group(1).strip() if m else ""


def gsm8k_exact_match(completions, answer, **kwargs):
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_answer(comp)
        rewards.append(1.0 if pred == gold.strip() else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for comp in completions:
        has_think = "<think>" in comp and "</think>" in comp
        has_answer = "<answer>" in comp and "</answer>" in comp
        rewards.append(0.1 * (has_think + has_answer))
    return rewards


def main(cfg: argparse.Namespace) -> None:
    # Stage hyperparameters; defaults are declared in build_parser().
    learning_rate: float = cfg.learning_rate
    num_epochs: int = cfg.num_epochs
    per_device_batch_size: int = cfg.per_device_batch_size
    grad_accum: int = cfg.grad_accum
    num_generations: int = cfg.num_generations
    max_completion_length: int = cfg.max_completion_length
    beta: float = cfg.beta
    use_format_shaping: bool = cfg.use_format_shaping
    grpo_data_file: str = cfg.grpo_data_file
    max_steps: int = cfg.max_steps
    attn_implementation: str = resolve_attn_implementation(cfg.attn_implementation)

    if cfg.max_prompt_length is not None:
        print(
            "[grpo] NOTE: --max_prompt_length is ignored (trl>=1.0 removed prompt "
            "truncation from GRPO); prompts are passed through untruncated."
        )

    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    output_dir = Path("results/raw") / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run = init_wandb(
        run_name=cfg.run_name,
        config={
            "pipeline": cfg.pipeline,
            "stage": "grpo",
            "model": cfg.model_name,
            "init_adapter_path": cfg.init_adapter_path or "none",
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_target_preset": cfg.lora_target_preset,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "num_generations": num_generations,
            "beta": beta,
            "use_format_shaping": use_format_shaping,
        },
        tags=["grpo", cfg.pipeline, f"lr={learning_rate}"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data
    dataset = load_dataset(
        "json",
        data_files=f"data/processed/{grpo_data_file}",
        split="train",
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        # transformers>=5 renamed `torch_dtype` to `dtype`
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )

    # --- config
    peft_config = None
    if cfg.init_adapter_path:
        # Continue training an existing adapter.
        model = PeftModel.from_pretrained(
            model, cfg.init_adapter_path, is_trainable=True
        )
    elif cfg.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=TARGET_MODULE_PRESETS[cfg.lora_target_preset],
            bias="none",
        )

    reward_funcs = [gsm8k_exact_match]
    if use_format_shaping:
        reward_funcs.append(format_reward)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=cfg.run_name,

        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,

        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        loss_type="grpo",
        temperature=0.9,
        top_p=0.95,

        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        seed=42,
        remove_unused_columns=False,
        # parser default is -1, which is transformers' own "no cap" sentinel;
        # passing None here used to be tolerated but is not a valid value.
        max_steps=max_steps,
        use_vllm=False,
    )

    # --- train
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --- log
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # The final log_history entry is the run summary (train_runtime/epoch), which
    # carries no "loss" key, so walk backwards to the last real training step.
    final_loss = next(
        (entry["loss"] for entry in reversed(trainer.state.log_history) if "loss" in entry),
        None,
    )
    summary = {
        "peak_memory_gb": peak_mem,
        "final_loss": final_loss,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    run.log(summary)
    run.finish()


if __name__ == "__main__":
    main(build_parser().parse_args())
