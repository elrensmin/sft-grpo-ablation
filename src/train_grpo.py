import os
import re
import json
from pathlib import Path

import torch
import tyro
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from src.config import BaseRunConfig, TARGET_MODULE_PRESETS
from src.utils.wandb_setup import init_wandb

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


def main(cfg: BaseRunConfig,
         learning_rate: float = 1e-6,
         num_epochs: int = 1,
         per_device_batch_size: int = 2,
         grad_accum: int = 8,
         num_generations: int = 4,
         max_prompt_length: int = 512,
         max_completion_length: int = 512,
         beta: float = 0.0,
         use_format_shaping: bool = True,
         grpo_data_file: str = "grpo_train_small.jsonl",
         max_steps: int = -1):

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
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
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
        max_prompt_length=max_prompt_length,
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
        max_steps=max_steps if max_steps > 0 else None,
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
    summary = {
        "peak_memory_gb": peak_mem,
        "final_loss": trainer.state.log_history[-1].get("loss"),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    run.log(summary)
    run.finish()


if __name__ == "__main__":
    tyro.cli(main)
