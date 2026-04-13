"""Train a TopK sparse autoencoder on the grokked transformer's residual stream.

Pipeline:
  1. Load grok_p113.pt.
  2. Build the full dataset of (a, b) pairs, collect the last-token
     residual stream for every pair. Shape [p*p, d_model].
  3. Train a TopK SAE on those activations.
  4. Save the SAE + a feature activation cache for downstream analysis.

Run:
    python -m src.train_sae --d-sae 256 --k 6 --epochs 3000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from .model import Config, TinyTransformer
from .train_modular import make_dataset
from .sae import TopKSAE


def collect_activations(ckpt_path: Path, p: int, device: str) -> torch.Tensor:
    cfg = Config(vocab_size=p + 1)
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    x, _ = make_dataset(p, device)
    acts = model.residual_last(x)  # [p*p, d_model]
    return acts.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/grok_p113.pt")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--d-sae", type=int, default=256)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default="results/sae.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"collecting residual activations from {args.ckpt}")
    X = collect_activations(Path(args.ckpt), args.p, args.device)
    print(f"  shape {tuple(X.shape)}  mean {X.mean().item():.4f}  std {X.std().item():.4f}")

    # Center for stability — the SAE decoder bias can absorb the mean,
    # but an explicit pre-centering makes training more stable.
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    sae = TopKSAE(d_in=X.shape[1], d_sae=args.d_sae, k=args.k).to(args.device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    base_var = (Xc ** 2).mean().item()
    print(f"  target variance {base_var:.4f}")

    for step in range(args.epochs + 1):
        loss, acts = sae.loss(Xc)
        opt.zero_grad()
        loss.backward()
        # Project decoder columns to unit norm — a standard SAE hygiene
        # step that prevents encoder/decoder rescaling gaming the objective.
        with torch.no_grad():
            W = sae.decoder.weight  # [d_in, d_sae]
            W.div_(W.norm(dim=0, keepdim=True).clamp(min=1e-8))
        opt.step()

        if step % 200 == 0:
            recon_frac = 1.0 - loss.item() / base_var
            dead = (acts.abs().sum(dim=0) == 0).sum().item()
            print(f"step {step:5d}  mse {loss.item():.5f}  var_explained {recon_frac:.3f}  dead {dead}/{args.d_sae}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"d_in": X.shape[1], "d_sae": args.d_sae, "k": args.k},
        "mean": mean.cpu(),
        "activations": X.cpu(),
    }, out)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
