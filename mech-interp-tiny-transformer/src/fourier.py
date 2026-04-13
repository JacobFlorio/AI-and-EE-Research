"""Fourier-basis analysis of a grokked modular-addition transformer.

The generalizing solution is known to embed inputs on a set of key
frequencies k ∈ {1..(p-1)/2} of Z/pZ, compute (a+b) via trig identities,
then read out the answer with an inverse DFT. This module projects the
token embedding matrix onto the Fourier basis and reports which
frequencies carry the mass.
"""
from __future__ import annotations
import numpy as np
import torch


def fourier_basis(p: int) -> torch.Tensor:
    """Real Fourier basis for Z/pZ, shape [p, p]."""
    F = torch.zeros(p, p)
    F[0] = 1.0 / np.sqrt(p)
    for k in range(1, (p // 2) + 1):
        x = torch.arange(p).float()
        F[2 * k - 1] = np.sqrt(2.0 / p) * torch.cos(2 * np.pi * k * x / p)
        if 2 * k < p:
            F[2 * k] = np.sqrt(2.0 / p) * torch.sin(2 * np.pi * k * x / p)
    return F


def embedding_fourier_power(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """Return per-frequency power of the token embedding (excluding '=' token)."""
    W = W_E[:p].detach().cpu()  # [p, d_model]
    F = fourier_basis(p)
    proj = F @ W  # [p, d_model]
    power = (proj**2).sum(dim=-1)  # [p]
    return power


def top_frequencies(power: torch.Tensor, k: int = 5):
    vals, idx = power.topk(k)
    return [(int(i), float(v)) for i, v in zip(idx, vals)]
