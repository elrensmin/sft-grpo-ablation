import argparse
import json

TARGET_MODULE_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
}


def add_run_config(parser: argparse.ArgumentParser) -> None:
    """Register the shared run identity / model / LoRA flags on `parser`."""
    g = parser.add_argument_group("run config")

    # Identity
    g.add_argument("--run_name", default="debug",
                   help="Run identifier: names results/raw/<run_name> and the W&B run")
    g.add_argument("--stage", default="sft", choices=["sft", "grpo"])
    g.add_argument("--pipeline", default="P2", choices=["P0", "P1", "P2", "P3"],
                   help="Ablation pipeline this run belongs to")
    g.add_argument("--notes", default="", help="Free-form note, logged to W&B")

    # Model
    g.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    g.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction,
                   default=False)

    # LoRA
    g.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--lora_r", type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=32,
                   help="Pipeline scripts use 2 * lora_r")
    g.add_argument("--lora_dropout", type=float, default=0.0)
    g.add_argument("--lora_target_preset", default="attn",
                   choices=sorted(TARGET_MODULE_PRESETS))
    g.add_argument("--init_adapter_path", default=None,
                   help="Continue training from this existing LoRA adapter")

    # Logging
    g.add_argument("--wandb_project", default="sft-grpo-ablation")
    g.add_argument("--wandb_tags", type=json.loads, default=[],
                   help='JSON list, e.g. \'["P0","baseline"]\'')
