"""Train a tiny transformer on modular addition and watch it grok.

Task: predict (a + b) mod p from the sequence [a, b, =].
With a small training fraction and weight decay, the model memorizes
first, then generalizes hundreds of epochs later. That phase transition
is "grokking," and the generalizing solution is known to implement a
discrete Fourier transform over Z/pZ.

Run:
    python -m src.train_modular --steps 40000 --train-frac 0.3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .model import Config, TinyTransformer


def make_dataset(p: int, device: str):
    a = torch.arange(p, device=device).repeat_interleave(p)
    b = torch.arange(p, device=device).repeat(p)
    eq = torch.full_like(a, p)  # "=" token id
    x = torch.stack([a, b, eq], dim=1)
    y = (a + b) % p
    return x, y


def split(x, y, frac: float, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)
    n_tr = int(frac * len(x))
    tr, te = idx[:n_tr], idx[n_tr:]
    return x[tr], y[tr], x[te], y[te]


def accuracy(model, x, y):
    with torch.no_grad():
        logits = model(x)[:, -1, :]
        return (logits.argmax(-1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--out", type=str, default="results/grokking_run.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = Config(vocab_size=args.p + 1)
    model = TinyTransformer(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))

    x, y = make_dataset(args.p, args.device)
    x_tr, y_tr, x_te, y_te = split(x, y, args.train_frac)

    history = []
    for step in range(args.steps + 1):
        logits = model(x_tr)[:, -1, :]
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            tr_acc = accuracy(model, x_tr, y_tr)
            te_acc = accuracy(model, x_te, y_te)
            history.append({"step": step, "loss": loss.item(), "train": tr_acc, "test": te_acc})
            print(f"step {step:6d}  loss {loss.item():.4f}  train {tr_acc:.3f}  test {te_acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "history": history}, indent=2))
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
