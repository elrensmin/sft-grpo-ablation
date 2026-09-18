import argparse
import json
import re
import subprocess
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a merged model on GSM8K + retention benchmarks")
    parser.add_argument("--model_path", required=True, help="Merged model directory to evaluate")
    parser.add_argument("--run_name", required=True, help="Names results/evals/<run_name>/")
    parser.add_argument("--eval_dir", default="results/evals")
    parser.add_argument("--limit", type=int, default=None,
                        help="lm-eval --limit (few-shot samples per retention task)")
    parser.add_argument("--gsm8k_samples", type=int, default=1319)
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction,
                        default=True, help="Use vLLM for GSM8K; --no-use_vllm uses greedy HF decoding")
    return parser


BENCHMARKS = {
    "retention": ["mmlu", "ifeval", "bbh"],
}


def _make_prompt(tokenizer, question: str):
    msg = (
        "Solve the following math problem. Show your reasoning inside "
        " thinking... response tags, then give the final numeric answer inside "
        "<answer>...</answer> tags.\n\n"
        f"Problem: {question.strip()}"
    )
    if tokenizer.chat_template is not None:
        chat = [{"role": "user", "content": msg}]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return f"### Instruction\n{msg}\n\n### Response\n"


def eval_gsm8k_hf(model_path: str, output_file: str, n_samples: int = 20,
                  max_new_tokens: int = 256):
    """vLLM-free GSM8K eval via greedy transformers decoding (for small GPUs)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )

    test = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(test):
        test = test.select(range(n_samples))

    n_format = 0
    n_exact = 0
    records = []
    for ex in test:
        text = _make_prompt(tokenizer, ex["question"])
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gold = ex["answer"].split("####")[-1].strip()

        has_think = " thinking" in generated and " response" in generated
        has_answer = "<answer>" in generated and "</answer>" in generated
        format_ok = has_think and has_answer

        m = re.search(r"<answer>\s*(.*?)\s*</answer>", generated, re.DOTALL)
        pred = m.group(1).strip() if m else ""
        exact = pred == gold

        n_format += int(format_ok)
        n_exact += int(exact)
        records.append({
            "completion": generated,
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


def eval_gsm8k_format(model_path: str, output_file: str, n_samples: int = 1319):
    from vllm import LLM, SamplingParams  # deferred: only on vLLM machines
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
         limit: int | None = None, gsm8k_samples: int = 1319,
         use_vllm: bool = True):
    out_dir = Path(eval_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Evaluating {run_name} ==")
    if use_vllm:
        from vllm import LLM, SamplingParams  # deferred: only on vLLM machines
        target = eval_gsm8k_format(model_path, str(out_dir / "gsm8k.json"), n_samples=gsm8k_samples)
    else:
        target = eval_gsm8k_hf(model_path, str(out_dir / "gsm8k.json"), n_samples=gsm8k_samples)
        summary = {"run_name": run_name, **target}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

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
    main(**vars(build_parser().parse_args()))
