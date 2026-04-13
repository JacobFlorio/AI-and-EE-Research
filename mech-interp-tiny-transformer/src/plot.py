"""Produce the headline figures for the grokking writeup."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from .model import Config, TinyTransformer
from .fourier import embedding_fourier_power


def train_test_curve(history_path: Path, out: Path):
    data = json.loads(history_path.read_text())
    h = data["history"]
    steps = [r["step"] for r in h]
    train = [r["train"] for r in h]
    test = [r["test"] for r in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, train, label="train", lw=2)
    ax.plot(steps, test, label="test", lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_title("Grokking on modular addition (p=113, train frac 0.3)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def fourier_spectrum(ckpt_path: Path, p: int, out: Path):
    model = TinyTransformer(Config(vocab_size=p + 1))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    power = embedding_fourier_power(model.tok_embed.weight, p).numpy()
    freqs = ["DC"] + [f"k={(i+1)//2}{'c' if i%2 else 's'}" for i in range(1, p)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(p), power)
    ax.set_xlabel("Fourier component index")
    ax.set_ylabel("power")
    ax.set_title("Token-embedding Fourier spectrum after grokking")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="results/grok_p113.json")
    ap.add_argument("--ckpt", default="results/grok_p113.pt")
    ap.add_argument("--p", type=int, default=113)
    args = ap.parse_args()
    train_test_curve(Path(args.history), Path("results/grokking_curve.png"))
    fourier_spectrum(Path(args.ckpt), args.p, Path("results/fourier_spectrum.png"))


if __name__ == "__main__":
    main()
