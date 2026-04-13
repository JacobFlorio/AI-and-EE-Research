"""Run the capability-delta matrix across a model × quant × eval grid.

Example:
    python -m src.run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quants fp16,int8,int4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .harness import HFBackend
from .evals import EVAL_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quants", default="fp16,int8,int4")
    ap.add_argument("--evals", default="toy_mcq,multi_step_arith")
    ap.add_argument("--out", default="results/capability_matrix.json")
    args = ap.parse_args()

    quants = args.quants.split(",")
    evals = args.evals.split(",")
    results = {"model": args.model, "rows": []}

    for q in quants:
        print(f"\n=== loading {args.model} @ {q} ===")
        backend = HFBackend(model_id=args.model, quant=q).load()
        row = {"quant": q, "metrics": {}}
        for name in evals:
            fn = EVAL_REGISTRY[name]
            r = fn(backend)
            row["metrics"][name] = r
            print(f"  {name}: {r}")
        results["rows"].append(row)
        del backend
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
