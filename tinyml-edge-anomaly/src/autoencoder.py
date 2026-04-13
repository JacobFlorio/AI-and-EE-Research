"""1D conv autoencoder for bearing-vibration anomaly detection.

Healthy-only training — reconstruction error on faulted data becomes
the anomaly score. Sized so int8 weights + activations fit in the
Cortex-M4 target budget (~128 KB RAM).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class VibrationAE(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, in_ch, 7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

    @torch.no_grad()
    def anomaly_score(self, x):
        recon = self(x)
        return ((recon - x) ** 2).mean(dim=(1, 2))
