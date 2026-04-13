"""Run a full transformer forward pass through the simulator at a sweep
of array sizes and report the roofline.

For each array size N in {8, 16, 32, 64, 128}:
  - Forward a GPT-2 small-shaped transformer (d=768, d_ff=3072, L=12)
    at batch=1, context=512 tokens.
  - Report total cycles, tokens/sec, GOPs/sec, per-op breakdown.
  - Compare to arithmetic-only peak (2 * N^2 * clock_mhz ops/sec).

The headline plot: array side N vs tokens/sec, with the arithmetic
ceiling plotted as a dashed line so you can see where MAC utilization
stops improving and overhead dominates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .systolic_sim import ArrayConfig
from .transformer_forward import TransformerShape, forward_one_batch, summarize


MODELS = {
    "gpt2-nano": TransformerShape(
        name="gpt2-nano", d_model=128, d_ff=512,
        n_heads=4, n_layers=4, n_ctx=128, vocab_size=4096,
    ),
    "gpt2-tiny": TransformerShape(
        name="gpt2-tiny", d_model=256, d_ff=1024,
        n_heads=8, n_layers=6, n_ctx=256, vocab_size=8192,
    ),
    "gpt2-small": TransformerShape(
        name="gpt2-small", d_model=768, d_ff=3072,
        n_heads=12, n_layers=12, n_ctx=512, vocab_size=50257,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-sizes", default="8,16,32,64,128")
    ap.add_argument("--model", default="gpt2-nano",
                    choices=list(MODELS.keys()))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--clock-mhz", type=float, default=400.0)
    ap.add_argument("--out", default="results/sweep.json")
    args = ap.parse_args()

    shape = MODELS[args.model]
    sizes = [int(s) for s in args.array_sizes.split(",")]

    print(f"model     : {shape.name}  (d={shape.d_model}, d_ff={shape.d_ff}, "
          f"L={shape.n_layers}, ctx={shape.n_ctx})")
    print(f"batch     : {args.batch}")
    print(f"clock     : {args.clock_mhz} MHz")
    print()
    print(f"{'array N':>8s} {'cycles':>14s} {'time(ms)':>10s} "
          f"{'tok/s':>10s} {'GOPs/s':>10s} {'peak_GOPs':>10s}  util")
    rows = []
    for N in sizes:
        cfg = ArrayConfig(N=N, clock_mhz=args.clock_mhz)
        stats = forward_one_batch(shape, args.batch, cfg)
        s = summarize(stats)
        peak_gops = 2 * N * N * args.clock_mhz / 1000.0  # 2 ops per MAC
        util = s["gops_per_sec"] / peak_gops
        s["peak_gops"] = peak_gops
        s["utilization"] = util
        rows.append(s)
        print(f"{N:>8d} {s['total_cycles']:>14,d} "
              f"{s['total_seconds']*1000:>10.1f} "
              f"{s['tokens_per_sec']:>10.1f} "
              f"{s['gops_per_sec']:>10.1f} {peak_gops:>10.1f}  "
              f"{util:.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": shape.__dict__,
        "clock_mhz": args.clock_mhz,
        "batch": args.batch,
        "rows": rows,
    }, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
