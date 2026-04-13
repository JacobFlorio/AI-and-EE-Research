"""Train the RF modulation classifier.

Uses synthetic IQ as a stand-in until RadioML 2018.01A is wired in.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "sae-rf-classifier"))
from src.synth_data import generate, CLASSES  # noqa: E402
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .resnet1d import ResNet1D


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = generate(n_per_class=512, snr_db=10.0)
    n_tr = int(0.8 * len(X))
    perm = torch.randperm(len(X))
    tr = TensorDataset(X[perm[:n_tr]], y[perm[:n_tr]])
    te = TensorDataset(X[perm[n_tr:]], y[perm[n_tr:]])
    tr_dl = DataLoader(tr, batch_size=128, shuffle=True)
    te_dl = DataLoader(te, batch_size=128)

    model = ResNet1D(n_classes=len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(10):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(-1) == yb).sum().item()
                total += len(yb)
        print(f"epoch {epoch}  test_acc {correct/total:.3f}")


if __name__ == "__main__":
    main()
