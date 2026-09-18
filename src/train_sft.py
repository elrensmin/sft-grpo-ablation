import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from src.config import add_run_config, TARGET_MODULE_PRESETS
from src.utils.wandb_setup import init_wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT stage (pipelines P1/P2/P3)")
    add_run_config(parser)

    g = parser.add_argument_group("sft")
    g.add_argument("--sft_dose", type=int, default=500,
                   help="SFT examples to use; reads data/processed/sft_dose_<N>.jsonl")
    g.add_argument("--learning_rate", type=float, default=2e-4)
    g.add_argument("--num_epochs", type=int, default=1)
    g.add_argument("--per_device_batch_size", type=int, default=4)
    g.add_argument("--grad_accum", type=int, default=4)
    g.add_argument("--max_length", type=int, default=2048)
    return parser


def main(cfg: argparse.Namespace) -> None:
    # Stage hyperparameters; defaults are declared in build_parser().
    sft_dose: int = cfg.sft_dose
    learning_rate: float = cfg.learning_rate
    num_epochs: int = cfg.num_epochs
    per_device_batch_size: int = cfg.per_device_batch_size
    grad_accum: int = cfg.grad_accum
    max_length: int = cfg.max_length

    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    output_dir = Path("results/raw") / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    run = init_wandb(
        run_name=cfg.run_name,
        config={
            "pipeline": cfg.pipeline,
            "stage": "sft",
            "model": cfg.model_name,
            "sft_dose": sft_dose,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_target_preset": cfg.lora_target_preset,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "per_device_batch_size": per_device_batch_size,
            "grad_accum": grad_accum,
            **({"notes": cfg.notes} if cfg.notes else {}),
        },
        tags=["sft", cfg.pipeline, cfg.lora_target_preset, f"dose={sft_dose}"],
    )

    # --- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # --- data
    dose_file = f"data/processed/sft_dose_{sft_dose}.jsonl"
    dataset = load_dataset("json", data_files=dose_file, split="train")

    # Use the model's chat template when available (Qwen etc.); fall back to a
    # plain instruction format for models without one (OPT, GPT-2).
    has_chat = tokenizer.chat_template is not None

    def format_and_tokenize(example):
        if has_chat:
            chat = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
            text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
        else:
            text = (
                f"### Instruction\n{example['prompt']}\n\n"
                f"### Response\n{example['completion']}"
            )
        return {"text": text}

    dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    run.config.update({"train_examples_used": len(dataset)}, allow_val_change=True)

    # -- config
    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=TARGET_MODULE_PRESETS[cfg.lora_target_preset],
            bias="none",
        )
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=cfg.run_name,
        max_length=max_length,
        packing=False,

        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        # transformers>=5 dropped `warmup_ratio`; `warmup_steps` now takes a float
        # in [0, 1) meaning a *ratio* of total steps, so 0.03 == the old 0.03.
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        dataset_text_field="text",
        seed=42,
    )
    # --- train
    trainer = SFTTrainer(
        model=cfg.model_name,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # --- post-training metrics
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    summary = {
        "peak_memory_gb": peak_mem,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "final_loss": trainer.state.log_history[-1].get("train_loss"),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    run.log(summary)
    run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
