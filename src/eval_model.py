import json
import re
import subprocess
from pathlib import Path

import torch
import tyro
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


BENCHMARKS = {
    "retention": ["mmlu", "ifeval", "bbh"],
}


def eval_gsm8k_format(model_path: str, output_file: str, n_samples: int = 1319):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
        max_model_len=2048,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(test):
        test = test.select(range(n_samples))

    prompts = []
    golds = []
    for ex in test:
        msg = (
            "Solve the following math problem. Show your reasoning inside "
            "<think>...</think> tags, then give the final numeric answer inside "
            "<answer>...</answer> tags.\n\n"
            f"Problem: {ex['question'].strip()}"
        )
        chat = [{"role": "user", "content": msg}]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        golds.append(ex["answer"].split("####")[-1].strip())

    sampling = SamplingParams(temperature=0.0, max_tokens=512)
    outs = llm.generate(prompts, sampling)

    n_format = 0
    n_exact = 0
    records = []
    for out, gold in zip(outs, golds):
        text = out.outputs[0].text
        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        format_ok = has_think and has_answer

        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        pred = m.group(1).strip() if m else ""
        exact = pred == gold

        n_format += int(format_ok)
        n_exact += int(exact)
        records.append({
            "question": out.prompt[-200:],
            "completion": text,
            "gold": gold,
            "pred": pred,
            "format_ok": format_ok,
            "exact": exact,
        })

    result = {
        "gsm8k_format_acc": n_format / len(records),
        "gsm8k_exact_acc": n_exact / len(records),
        "n": len(records),
    }
    Path(output_file).write_text(json.dumps({"metrics": result, "records": records}, indent=2))
    return result


def eval_lm_harness(model_path: str, output_dir: str, limit: int | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for task in BENCHMARKS["retention"]:
        out_file = output_dir / f"{task}.json"
        cmd = [
            "lm_eval",
            "--model", "vllm",
            "--model_args", f"pretrained={model_path},dtype=bfloat16",
            "--tasks", task,
            "--batch_size", "auto",
            "--num_fewshot", "0",
            "--apply_chat_template",
            "--output_path", str(out_file),
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, check=True)

        with open(out_file) as f:
            data = json.load(f)
        # Extract main metric
        for k, v in data["results"][task].items():
            if k.endswith("acc,none") or k == "acc":
                results[task] = v
                break
        else:
            # fallback
            results[task] = list(data["results"][task].values())[0]

    return results


def main(model_path: str, run_name: str, eval_dir: str = "results/evals",
         limit: int | None = None, gsm8k_samples: int = 1319):
    out_dir = Path(eval_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Evaluating {run_name} ==")
    target = eval_gsm8k_format(model_path, str(out_dir / "gsm8k.json"), n_samples=gsm8k_samples)
    retention = eval_lm_harness(model_path, str(out_dir), limit=limit)

    summary = {"run_name": run_name, **target, **retention}
    if limit:
        summary["retention_limit"] = limit
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Add to src/eval_model.py
    import wandb, os
    os.environ.setdefault("WANDB_PROJECT", "sft-grpo-ablation")
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=run_name,
        resume="allow",
        reinit=True,
    )
    run.log({f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
    run.finish()


if __name__ == "__main__":
    tyro.cli(main)
