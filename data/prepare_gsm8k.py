"""Prepare GSM8K in two formats:
1. SFT format: <think>...</think><answer>...</answer>
2. GRPO format: prompt + ground-truth answer for reward function.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

ANSWER_RE = re.compile(r"####\s*(.+)")


def split_cot(answer: str) -> tuple[str, str]:
    """GSM8K answers look like '<cot>\n#### 42'. Split into (cot, final)."""
    m = ANSWER_RE.search(answer)
    if not m:
        return answer.strip(), ""
    cot = answer[: m.start()].strip()
    final = m.group(1).strip()
    return cot, final


def to_sft(example, tokenizer_special: str = "qwen"):
    cot, final = split_cot(example["answer"])
    question = example["question"].strip()
    return {
        "prompt": question,
        "completion": (
            f"<think>\n{cot}\n</think>\n"
            f"<answer>\n{final}\n</answer>"
        ),
        "answer": final,
    }


def to_grpo(example):
    _, final = split_cot(example["answer"])
    return {
        "prompt": (
            "Solve the following math problem. "
            "Show your reasoning inside <think>...</think> tags, "
            "then give the final numeric answer inside <answer>...</answer> tags.\n\n"
            f"Problem: {example['question'].strip()}"
        ),
        "answer": final,
    }


def main():
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    sft_train = train.map(to_sft)
    sft_train = sft_train.shuffle(seed=42)

    for n in [500, 5000, 50000]:
        subset = sft_train.select(range(min(n, len(sft_train))))
        subset.to_json(OUT / f"sft_dose_{n}.jsonl")

    grpo_train = train.map(to_grpo)
    grpo_train.to_json(OUT / "grpo_train.jsonl")

    grpo_train_small = grpo_train.select(range(1500))
    grpo_train_small.to_json(OUT / "grpo_train_small.jsonl")

    test_sft = test.map(to_sft)
    test_grpo = test.map(to_grpo)
    test_sft.to_json(OUT / "test_sft.jsonl")
    test_grpo.to_json(OUT / "test_grpo.jsonl")

    meta = {
        "n_train": len(train),
        "n_test": len(test),
        "sft_doses": [500, 5000, 50000],
        "grpo_small": len(grpo_train_small),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote processed datasets to {OUT}")


if __name__ == "__main__":
    main()
