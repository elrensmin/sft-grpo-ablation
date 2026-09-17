import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

from src.utils.spectral import task_vector_cosine, spectral_drift

BASE = "Qwen/Qwen2.5-3B-Instruct"
RAW = Path("results/raw")
OUT = Path("results/task_vector_analysis.json")


def load_base_state():
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu"
    )
    return {k: v.clone() for k, v in model.state_dict().items()}


def load_adapter_delta(adapter_dir: Path, base_state: dict) -> dict:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model_name_or_path"], torch_dtype=torch.float32, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    delta = {}
    for k, v in merged.state_dict().items():
        if k in base_state:
            delta[k] = v - base_state[k]
    return delta


def main():
    base_state = load_base_state()

    # Collect deltas for P0 (GRPO-only) and P1 (SFT-only) and P2 (SFT→GRPO)
    deltas = {}
    for run_dir in RAW.iterdir():
        if not (run_dir / "adapter_config.json").exists():
            continue
        name = run_dir.name
        deltas[name] = load_adapter_delta(run_dir, base_state)

    # Pairwise cosines between the three "pure" task vectors per dose/rank
    cos_matrix = {}
    names = sorted(deltas.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sim = task_vector_cosine(deltas[a], deltas[b])
            cos_matrix[f"{a}||{b}"] = sim

    # Spectral drift per adapter
    drift = {name: sum(spectral_drift(base_state, {k: base_state[k] + v
                                                  for k, v in d.items()}).values())
             for name, d in deltas.items()}

    OUT.write_text(json.dumps({"cosines": cos_matrix, "drift": drift}, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
