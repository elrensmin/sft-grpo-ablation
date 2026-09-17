import os
import wandb
from dotenv import load_dotenv

load_dotenv()


def init_wandb(run_name: str, config: dict, tags: list | None = None):
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "sft-grpo-ablation"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        config=config,
        tags=tags or [],
        reinit=True,
    )


def log_metrics(metrics: dict, step: int | None = None):
    wandb.log(metrics, step=step)
