"""Plot SAE feature recovery against the ground-truth Fourier basis."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


KEY_FREQUENCIES = [1, 19, 36, 49, 56]


def main():
    data = json.loads(Path("results/sae_features.json").read_text())
    features = data["features"]

    # Histogram of diagonal feature frequencies
    diag_ks = [f["ka"] for f in features if f["ka"] == f["kb"] and f["ka"] > 0]
    max_k = max(diag_ks) if diag_ks else 1
    bins = np.arange(1, max_k + 2) - 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    counts, _, bars = ax.hist(diag_ks, bins=bins, edgecolor="#222")
    for b, k in zip(bars, np.arange(1, max_k + 1)):
        if int(k) in KEY_FREQUENCIES:
            b.set_facecolor("#2ca02c")
        else:
            b.set_facecolor("#888")
    ax.set_xlabel("diagonal feature frequency k")
    ax.set_ylabel("number of SAE features")
    ax.set_title("SAE feature frequencies (green = ground-truth key freq)")
    for k in KEY_FREQUENCIES:
        ax.axvline(k, color="#2ca02c", alpha=0.2, lw=1)
    ax.grid(alpha=0.3, axis="y")

    # Bar chart: recovered-features-per-key-frequency
    ax = axes[1]
    recovered = {int(k): v for k, v in data["recovered_key_freqs"].items()}
    xs = list(recovered.keys())
    ys = [recovered[k] for k in xs]
    ax.bar([str(k) for k in xs], ys, color="#2ca02c")
    for i, v in enumerate(ys):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_xlabel("key frequency")
    ax.set_ylabel("SAE features recovering this frequency")
    ax.set_title(f"Ground-truth recovery: {sum(1 for v in ys if v > 0)}/{len(KEY_FREQUENCIES)}")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("TopK SAE feature recovery on grokked transformer")
    fig.tight_layout()
    out = Path("results/sae_recovery.png")
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
