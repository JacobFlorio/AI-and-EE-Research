"""Score SAE features against the ground-truth Fourier basis of Z/pZ.

Premise: the grokked transformer uses a handful of "key frequencies"
k ∈ {1, 19, 36, 49, 56} to compute (a+b) mod p via trig identities.
If the SAE has "recovered" the circuit, individual features should
correspond to these frequencies — each feature's activation as a function
of (a, b) should concentrate on a small number of 2D Fourier components
over Z/pZ × Z/pZ, and those components should be at the key frequencies.

Method:
  1. Load SAE + cached residual activations [p*p, d_model].
  2. Encode via SAE → feature activations [p*p, d_sae].
  3. Reshape to [p, p, d_sae] indexed by (a, b).
  4. For each alive feature, 2D-FFT the response map, find the dominant
     frequency pair (k_a, k_b), and record whether k_a == k_b and whether
     that frequency is one of the ground-truth key frequencies.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .sae import TopKSAE


KEY_FREQUENCIES = [1, 19, 36, 49, 56]


def analyze(ckpt_path: Path, p: int):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    sae = TopKSAE(**cfg)
    sae.load_state_dict(blob["state_dict"])
    sae.eval()

    acts_in = blob["activations"] - blob["mean"]
    with torch.no_grad():
        feats = sae.encode(acts_in)  # [p*p, d_sae]

    d_sae = feats.shape[1]
    feat_map = feats.reshape(p, p, d_sae).numpy()  # [a, b, f]
    # Per-feature total power as a sanity check for "alive" features
    per_feat_var = feat_map.var(axis=(0, 1))  # [d_sae]
    alive = np.where(per_feat_var > 1e-8)[0]

    # Fold both axes into the first half of the frequency spectrum.
    # For a real signal of length p, freq k and p-k carry conjugate info;
    # we fold them into a single power at min(k, p-k).
    def folded_freq(k: int) -> int:
        return min(k, p - k)

    records = []
    for f in alive:
        fm = feat_map[:, :, f]
        fm = fm - fm.mean()
        spec = np.fft.fft2(fm)
        power = np.abs(spec) ** 2
        # argmax ignoring DC
        power_flat = power.copy()
        power_flat[0, 0] = 0
        ka, kb = np.unravel_index(power_flat.argmax(), power_flat.shape)
        ka_f, kb_f = folded_freq(int(ka)), folded_freq(int(kb))
        total = power_flat.sum()
        peak_frac = float(power_flat[ka, kb] / total) if total > 0 else 0.0
        # "Clean" if the top component accounts for most of the non-DC power,
        # counting both (k, k) and its conjugate (p-k, p-k) which carry the
        # same information for real-valued feature maps.
        conj_frac = 0.0
        if ka_f > 0 and kb_f > 0:
            mirror = power_flat[(-ka) % p, (-kb) % p]
            conj_frac = float(mirror / total) if total > 0 else 0.0
        clean_power = peak_frac + conj_frac
        records.append({
            "feature": int(f),
            "ka": ka_f,
            "kb": kb_f,
            "peak_frac": peak_frac,
            "clean_frac": clean_power,
            "variance": float(per_feat_var[f]),
        })

    # Sort by variance so high-impact features come first
    records.sort(key=lambda r: -r["variance"])

    # How many ground-truth frequencies got recovered?
    recovered = {k: 0 for k in KEY_FREQUENCIES}
    diagonal_features = 0  # features where k_a == k_b (the interpretable shape)
    for r in records:
        if r["ka"] == r["kb"] and r["ka"] in KEY_FREQUENCIES:
            recovered[r["ka"]] += 1
        if r["ka"] == r["kb"] and r["ka"] > 0:
            diagonal_features += 1

    print(f"alive features: {len(alive)}/{d_sae}")
    print(f"diagonal features (k_a == k_b > 0): {diagonal_features}/{len(alive)}")
    print(f"\nrecovered key frequencies:")
    for k, n in recovered.items():
        marker = "✓" if n > 0 else " "
        print(f"  k = {k:3d}  {marker}  {n} features")
    total_rec = sum(1 for k, n in recovered.items() if n > 0)
    print(f"\n{total_rec}/{len(KEY_FREQUENCIES)} ground-truth frequencies recovered")

    print(f"\ntop 12 features by variance:")
    print(f"  {'feat':>6} {'k_a':>4} {'k_b':>4} {'peak%':>7} {'clean%':>8} {'var':>8}")
    for r in records[:12]:
        on = "✓" if (r["ka"] == r["kb"] and r["ka"] in KEY_FREQUENCIES) else " "
        print(f"  {r['feature']:>6} {r['ka']:>4} {r['kb']:>4} "
              f"{100*r['peak_frac']:>6.1f}% {100*r['clean_frac']:>7.1f}% "
              f"{r['variance']:>8.2f}  {on}")

    out = ckpt_path.with_name("sae_features.json")
    out.write_text(json.dumps({
        "alive": int(len(alive)),
        "total": int(d_sae),
        "diagonal_count": int(diagonal_features),
        "recovered_key_freqs": {str(k): int(n) for k, n in recovered.items()},
        "features": records,
    }, indent=2))
    print(f"\nsaved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="results/sae.pt")
    ap.add_argument("--p", type=int, default=113)
    args = ap.parse_args()
    analyze(Path(args.sae), args.p)


if __name__ == "__main__":
    main()
