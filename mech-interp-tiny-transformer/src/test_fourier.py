"""Sanity tests for the real Fourier basis of Z/pZ.

Run with:
    python -m src.test_fourier
"""
from __future__ import annotations
import numpy as np
import torch
from .fourier import fourier_basis, embedding_fourier_power


def test_orthonormal(p: int):
    F = fourier_basis(p)
    gram = F @ F.T
    err = (gram - torch.eye(p)).abs().max().item()
    assert err < 1e-4, f"basis not orthonormal for p={p}: max err {err}"
    print(f"  p={p}: orthonormal (max err {err:.2e})")


def test_pure_tone_recovery(p: int, k: int):
    """A pure cos signal should concentrate on exactly its basis index."""
    F = fourier_basis(p)
    x = torch.arange(p).float()
    sig = np.sqrt(2.0 / p) * torch.cos(2 * np.pi * k * x / p)
    proj = F @ sig
    top = int(proj.argmax())
    assert top == 2 * k - 1, f"p={p} k={k}: expected idx {2*k-1}, got {top}"
    assert proj[top].item() > 0.999, f"p={p} k={k}: peak too small {proj[top]}"
    # off-peak energy should be numerically zero
    mask = torch.ones_like(proj, dtype=torch.bool)
    mask[top] = False
    off = (proj[mask] ** 2).sum().item()
    assert off < 1e-8, f"p={p} k={k}: off-peak energy {off}"
    print(f"  p={p} k={k}: clean recovery at idx {top}")


def test_power_conservation():
    """Embedding power in the Fourier basis should equal Euclidean power."""
    p = 113
    W = torch.randn(p, 64)
    power_fourier = embedding_fourier_power(W, p).sum().item()
    power_direct = (W ** 2).sum().item()
    err = abs(power_fourier - power_direct) / power_direct
    assert err < 1e-4, f"power not conserved: {err}"
    print(f"  Parseval: fourier={power_fourier:.3f} direct={power_direct:.3f}")


def main():
    print("orthonormality")
    for p in [7, 59, 113, 257]:
        test_orthonormal(p)
    print("\npure-tone recovery")
    for p, k in [(113, 1), (113, 19), (113, 56), (59, 13)]:
        test_pure_tone_recovery(p, k)
    print("\nparseval")
    test_power_conservation()
    print("\nall passed")


if __name__ == "__main__":
    main()
