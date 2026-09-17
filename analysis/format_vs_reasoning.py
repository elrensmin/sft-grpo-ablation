import json
from pathlib import Path
import pandas as pd

EVAL_DIR = Path("results/evals")


def main():
    rows = []
    for f in EVAL_DIR.glob("*/gsm8k.json"):
        data = json.loads(f.read_text())
        run = f.parent.name
        metrics = data["metrics"]

        # Count: correct format AND correct answer (true reasoning)
        n_reasoning = sum(
            1 for r in data["records"] if r["format_ok"] and r["exact"]
        )
        # Format correct but wrong answer (format learned, reasoning not)
        n_format_only = sum(
            1 for r in data["records"] if r["format_ok"] and not r["exact"]
        )
        n_total = metrics["n"]

        rows.append({
            "run": run,
            "format_only_pct": n_format_only / n_total,
            "reasoning_pct": n_reasoning / n_total,
            "format_acc": metrics["gsm8k_format_acc"],
            "exact_acc": metrics["gsm8k_exact_acc"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/format_vs_reasoning.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
