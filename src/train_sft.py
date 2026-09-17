import os
import json
from pathlib import Path

import torch
import tyro
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from src.config import BaseRunConfig, TARGET_MODULE_PRESETS
from src.utils.wandb_setup import init_wandb


def main(cfg: BaseRunConfig,
         sft_dose: int = 500,
         learning_rate: float = 2e-4,
         num_epochs: int = 1,
         per_device_batch_size: int = 4,
         grad_accum: int = 4,
         max_length: int = 2048):

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

    def format_and_tokenize(example):
        # Chat template applied at eval time; here we build a training string.
        text = (
            f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['completion']}<|im_end|>"
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
        warmup_ratio=0.03,
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
    tyro.cli(main)
