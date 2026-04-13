"""Render the capability matrix to a report-quality figure.

Reads results/capability_matrix.json and plots perplexity + arith_mcq
accuracy across quantization levels as a double-panel figure.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data = json.loads(Path("results/capability_matrix.json").read_text())
    quants = [r["quant"] for r in data["rows"]]
    ppl = [r["metrics"]["perplexity"]["perplexity_per_word"] for r in data["rows"]]
    arith = [r["metrics"]["arith_mcq"]["acc"] for r in data["rows"]]
    arith_margin = [r["metrics"]["arith_mcq"]["mean_margin"] for r in data["rows"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bars = ax.bar(quants, ppl, color=["#1f77b4", "#2ca02c", "#d62728"])
    ax.set_ylabel("perplexity per word")
    ax.set_title("Perplexity on fixed English snippet")
    for b, v in zip(bars, ppl):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax2 = ax.twinx()
    bars = ax.bar(quants, arith, color=["#1f77b4", "#2ca02c", "#d62728"], alpha=0.6)
    ax.set_ylabel("arith_mcq accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Arith MCQ (30 items)")
    for b, v in zip(bars, arith):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom")
    ax2.plot(quants, arith_margin, "ko--", label="mean logprob margin")
    ax2.set_ylabel("mean logprob margin")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Qwen2.5-0.5B capability × quantization")
    fig.tight_layout()
    out = Path("results/capability_matrix.png")
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
