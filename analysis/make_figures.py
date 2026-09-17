from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)


def fig_net_score_vs_dose(df):
    """Composite net score vs SFT dose, comparing P0, P1, P2."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for pipeline, sub in df.groupby("pipeline"):
        sub = sub.dropna(subset=["dose"])
        if len(sub) == 0:
            continue
        agg = sub.groupby("dose").agg(
            target=("gsm8k_exact", "mean"),
            retention=("retention_mean", "mean"),
        ).reset_index()
        # Composite: equal weight
        agg["net"] = 0.5 * agg["target"] + 0.5 * agg["retention"]
        ax.plot(agg["dose"], agg["net"], marker="o", label=f"{pipeline} net")
        ax.plot(agg["dose"], agg["target"], marker="^", linestyle="--",
                label=f"{pipeline} target")
        ax.plot(agg["dose"], agg["retention"], marker="v", linestyle=":",
                label=f"{pipeline} retention")
    ax.set_xscale("log")
    ax.set_xlabel("SFT dose (examples)")
    ax.set_ylabel("Score")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "fig1_net_vs_dose.png", dpi=200)
    plt.close()


def fig_rank_heatmap(df):
    sub = df[df["pipeline"] == "P1"].dropna(subset=["rank", "target", "gsm8k_exact"])
    if len(sub) == 0:
        return
    pivot = sub.pivot_table(
        index="rank", columns="target", values="gsm8k_exact", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("SFT-only: GSM8K exact acc")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_rank_target_heatmap.png", dpi=200)
    plt.close()


def fig_pareto(df):
    """Params (proxy: rank × target count) vs. target-task score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for pipeline, sub in df.groupby("pipeline"):
        if sub["rank"].isna().all():
            continue
        ax.scatter(sub["rank"], sub["gsm8k_exact"], label=pipeline, alpha=0.7, s=40)
    ax.set_xscale("log")
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("GSM8K exact accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig3_rank_vs_acc.png", dpi=200)
    plt.close()


def fig_format_vs_reasoning():
    df = pd.read_csv("results/format_vs_reasoning.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.sort_values("exact_acc")
    x = np.arange(len(df_sorted))
    ax.bar(x - 0.2, df_sorted["format_only_pct"], width=0.4, label="Format only")
    ax.bar(x + 0.2, df_sorted["reasoning_pct"], width=0.4, label="Format + correct")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["run"], rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("Fraction of test set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig4_format_vs_reasoning.png", dpi=200)
    plt.close()


def main():
    df = pd.read_csv("results/aggregate.csv")
    fig_net_score_vs_dose(df)
    fig_rank_heatmap(df)
    fig_pareto(df)
    fig_format_vs_reasoning()
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
