"""Residual 1D-CNN baseline for modulation classification.

Compact enough to run on a Raspberry Pi 5 after INT8 quantization;
deep enough to hit RadioML 2018.01A's expected accuracy band.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(ch)
        self.bn2 = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class ResNet1D(nn.Module):
    def __init__(self, n_classes: int = 11, base: int = 64, n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Conv1d(2, base, 7, padding=3)
        self.blocks = nn.Sequential(*[ResBlock1D(base) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
