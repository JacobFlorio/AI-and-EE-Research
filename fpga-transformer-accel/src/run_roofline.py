"""Run the roofline + energy analysis on a couple of transformer shapes
and dump results/roofline.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .systolic_sim import ArrayConfig
from .roofline import (
    forward_pass_energy,
    classify_forward_pass,
    matmul_arithmetic_intensity,
    roofline,
    forward_pass_matmul_shapes,
)
from .run_sweep import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-sizes", default="16,32,64,128")
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--out", default="results/roofline.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.array_sizes.split(",")]
    batches = [int(b) for b in args.batches.split(",")]

    out_rows = []
    for model_name, shape in MODELS.items():
        for N in sizes:
            cfg = ArrayConfig(N=N, clock_mhz=400.0, sram_bytes=512 * 1024,
                              sram_bw_gb_s=80.0, dram_bw_gb_s=8.0)
            classify = classify_forward_pass(shape, batch=1, cfg=cfg)
            for B in batches:
                e = forward_pass_energy(shape, batch=B, cfg=cfg)
                row = {
                    "model": model_name,
                    "array_N": N,
                    "batch": B,
                    "peak_compute_gops": classify["peak_compute_gops"],
                    "peak_bw_gb_s": classify["peak_bw_gb_s"],
                    "min_intensity": classify["min_intensity"],
                    "mean_intensity": classify["mean_intensity"],
                    "max_intensity": classify["max_intensity"],
                    "frac_ops_memory_bound_by_count":
                        classify["frac_ops_memory_bound_by_count"],
                    "frac_ops_memory_bound_by_macs":
                        classify["frac_ops_memory_bound_by_macs"],
                    **e,
                }
                out_rows.append(row)

    # Print summary tables
    print(f"\n{'model':>10s} {'N':>4s} {'B':>4s} "
          f"{'weights_fit':>12s} {'e_total_mJ':>12s} {'uJ/token':>10s} "
          f"{'mem_bound_ops%':>16s}")
    for r in out_rows:
        print(f"{r['model']:>10s} {r['array_N']:>4d} {r['batch']:>4d} "
              f"{str(r['weights_fit_in_sram']):>12s} "
              f"{r['e_total_mJ']:>12.2f} {r['e_per_token_uJ']:>10.1f} "
              f"{100*r['frac_ops_memory_bound_by_count']:>15.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": out_rows}, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
