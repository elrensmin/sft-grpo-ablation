from dataclasses import dataclass, field
from typing import Optional


TARGET_MODULE_PRESETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
}


@dataclass
class BaseRunConfig:
    # Identity
    run_name: str = "debug"
    stage: str = "sft"               # sft | grpo
    pipeline: str = "P2"             # P0 | P1 | P2 | P3
    notes: str = ""

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = False

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_preset: str = "attn"    # attn | mlp | all
    init_adapter_path: Optional[str] = None

    # Logging
    wandb_project: str = "sft-grpo-ablation"
    wandb_tags: list = field(default_factory=list)
