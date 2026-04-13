"""Post-hoc analysis of a grokked modular-addition transformer.

Loads the saved checkpoint, projects its token embedding onto the real
Fourier basis of Z/pZ, and reports which frequencies the network is
actually using. A grokked model concentrates power on a small handful
of "key frequencies" — this is the signature circuit result from
Nanda et al.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .model import Config, TinyTransformer
from .fourier import embedding_fourier_power, top_frequencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/grok_p113.pt")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    model = TinyTransformer(Config(vocab_size=args.p + 1))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    power = embedding_fourier_power(model.tok_embed.weight, args.p)
    total = float(power.sum())
    top = top_frequencies(power, k=args.top)

    print(f"total Fourier power: {total:.3f}")
    print(f"mean per-component power: {total / args.p:.4f}")
    print(f"\ntop {args.top} Fourier components:")
    for idx, p in top:
        freq = (idx + 1) // 2
        kind = "DC" if idx == 0 else ("cos" if idx % 2 == 1 else "sin")
        frac = p / total
        print(f"  idx {idx:3d}  k={freq:3d} {kind:3s}  power {p:.3f}  ({frac:6.1%})")

    top_power = sum(p for _, p in top)
    print(f"\nfraction of total power in top {args.top}: {top_power/total:.1%}")

    out = Path(args.ckpt).with_name("fourier_analysis.json")
    out.write_text(json.dumps({
        "total_power": total,
        "top": [{"idx": i, "power": p} for i, p in top],
        "concentration_top10": top_power / total,
    }, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
