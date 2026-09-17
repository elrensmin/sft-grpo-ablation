import json
from pathlib import Path
import pandas as pd
import numpy as np

EVAL_DIR = Path("results/evals")
OUT = Path("results/aggregate.csv")

DEFAULT_W_TARGET = 0.5


def parse_run_name(name: str) -> dict:
    parts = name.split("_")
    meta = {"pipeline": parts[0]}
    for p in parts[1:]:
        if p.startswith("dose"):
            meta["dose"] = int(p.replace("dose", ""))
        elif p.startswith("r") and p[1:].isdigit():
            meta["rank"] = int(p[1:])
        elif p in ("attn", "mlp", "all"):
            meta["target"] = p
        elif p.startswith("lr"):
            meta["lr"] = p
    return meta


def main():
    rows = []
    for summary_file in EVAL_DIR.glob("*/summary.json"):
        data = json.loads(summary_file.read_text())
        run_name = data["run_name"]
        meta = parse_run_name(run_name)

        target = data.get("gsm8k_exact_acc", np.nan)
        format_acc = data.get("gsm8k_format_acc", np.nan)
        retention_vals = [data.get(b, np.nan) for b in ["mmlu", "ifeval", "bbh"]]
        retention = np.nanmean(retention_vals)

        rows.append({
            "run_name": run_name,
            **meta,
            "gsm8k_exact": target,
            "gsm8k_format": format_acc,
            "mmlu": data.get("mmlu"),
            "ifeval": data.get("ifeval"),
            "bbh": data.get("bbh"),
            "retention_mean": retention,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
