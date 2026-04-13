"""Headline plots for the systolic-accelerator simulation study.

Produces:
  - throughput_scaling.png  tokens/sec vs array N for each model,
                            with the peak compute ceiling as a reference
  - utilization.png         MAC utilization vs array N, showing the
                            "array outgrows the model" effect
  - energy_vs_batch.png     μJ/token vs batch size for each model,
                            showing batch amortization of weight DRAM
  - roofline.png            classic compute-vs-BW roofline with every
                            matmul in one forward pass plotted as a dot
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .systolic_sim import ArrayConfig
from .roofline import (
    forward_pass_matmul_shapes,
    matmul_arithmetic_intensity,
    roofline,
)
from .run_sweep import MODELS


COLORS = {
    "gpt2-nano": "#2ca02c",
    "gpt2-tiny": "#1f77b4",
    "gpt2-small": "#d62728",
}


def _load_sweep(path: Path) -> dict:
    return json.loads(path.read_text())


def throughput_scaling(out: Path):
    paths = {
        "gpt2-nano": Path("results/sweep.json"),
        "gpt2-small": Path("results/sweep_gpt2_small.json"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model, path in paths.items():
        data = _load_sweep(path)
        rows = data["rows"]
        Ns = [r["array_N"] for r in rows]
        tok_s = [r["tokens_per_sec"] for r in rows]
        peak = [r["peak_gops"] * 1e9 / (2 * 1e6) for r in rows]  # normalize...
        ax.plot(Ns, tok_s, "o-", color=COLORS[model], lw=2, markersize=7,
                label=f"{model} (actual)")
        # Ideal-peak tokens/sec: tokens/sec if utilization were 100%.
        ideal = [r["tokens_per_sec"] / r["utilization"] for r in rows]
        ax.plot(Ns, ideal, ls="--", color=COLORS[model], alpha=0.5,
                label=f"{model} (peak)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("systolic array side N")
    ax.set_ylabel("tokens / sec (batch=1, ctx=model default)")
    ax.set_title("Throughput vs array size — sweet spot moves with model size")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def utilization_plot(out: Path):
    paths = {
        "gpt2-nano": Path("results/sweep.json"),
        "gpt2-small": Path("results/sweep_gpt2_small.json"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, path in paths.items():
        data = _load_sweep(path)
        rows = data["rows"]
        Ns = [r["array_N"] for r in rows]
        util = [100 * r["utilization"] for r in rows]
        ax.plot(Ns, util, "o-", color=COLORS[model], lw=2, markersize=7,
                label=model)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("systolic array side N")
    ax.set_ylabel("MAC utilization (%)")
    ax.set_title("Array outgrows the model: utilization collapses for gpt2-nano past N=32")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 50)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def energy_vs_batch(out: Path):
    data = json.loads(Path("results/roofline.json").read_text())
    rows = data["rows"]
    # For each model, pick N=64 as a representative array size
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in ["gpt2-nano", "gpt2-tiny", "gpt2-small"]:
        xs, ys = [], []
        for r in rows:
            if r["model"] == model and r["array_N"] == 64:
                xs.append(r["batch"])
                ys.append(r["e_per_token_uJ"])
        ax.plot(xs, ys, "o-", color=COLORS[model], lw=2, markersize=7,
                label=model)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("energy per token (μJ)")
    ax.set_title("Weight-DRAM cost amortizes with batch (N=64, 28nm CMOS)")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def roofline_plot(out: Path):
    """Classic roofline: every matmul in a forward pass as a point."""
    cfg = ArrayConfig(N=64, clock_mhz=400.0, dram_bw_gb_s=8.0)
    peak_compute_gops = 2 * cfg.N * cfg.N * cfg.clock_mhz / 1000.0
    bw = cfg.dram_bw_gb_s

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    intensities = np.logspace(0, 4, 200)
    ridge = peak_compute_gops / bw
    perf = np.minimum(bw * intensities, peak_compute_gops)
    ax.plot(intensities, perf, "k-", lw=2, label=f"roofline (N=64, 400 MHz)")
    ax.axvline(ridge, color="grey", ls=":", lw=1, alpha=0.7)
    ax.text(ridge * 1.05, peak_compute_gops * 0.6, f"ridge\n{ridge:.0f} ops/B",
            fontsize=8, color="grey")

    for model in ["gpt2-nano", "gpt2-tiny", "gpt2-small"]:
        shape = MODELS[model]
        ops = forward_pass_matmul_shapes(shape, batch=1)
        xs = [matmul_arithmetic_intensity(M, K, N) for M, K, N, _ in ops]
        ys = [min(bw * x, peak_compute_gops) for x in xs]
        ax.scatter(xs, ys, s=30, color=COLORS[model], alpha=0.6,
                   edgecolors="white", linewidths=0.5,
                   label=f"{model} matmul ops")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (ops / byte)")
    ax.set_ylabel("achievable compute (GOPs / s)")
    ax.set_title("Roofline — every forward-pass matmul lives below the compute ceiling")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def main():
    Path("results").mkdir(exist_ok=True)
    throughput_scaling(Path("results/throughput_scaling.png"))
    utilization_plot(Path("results/utilization.png"))
    energy_vs_batch(Path("results/energy_vs_batch.png"))
    roofline_plot(Path("results/roofline.png"))


if __name__ == "__main__":
    main()
