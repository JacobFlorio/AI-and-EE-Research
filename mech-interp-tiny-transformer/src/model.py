"""Minimal decoder-only transformer for mech-interp experiments.

Deliberately small and un-fused so every intermediate is easy to hook.
Based on the setup in Nanda et al. "Progress measures for grokking via
mechanistic interpretability" (2023).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 114  # p + 1 for "=" token, where p = 113
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    n_layers: int = 1
    n_ctx: int = 3


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        for p in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(p)

    def forward(self, x):  # x: [B, T, D]
        q = torch.einsum("btd,hdk->bhtk", x, self.W_Q)
        k = torch.einsum("btd,hdk->bhtk", x, self.W_K)
        v = torch.einsum("btd,hdk->bhtk", x, self.W_V)
        scores = torch.einsum("bhtk,bhsk->bhts", q, k) / (self.cfg.d_head**0.5)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhts,bhsk->bhtk", attn, v)
        return torch.einsum("bhtk,hkd->btd", out, self.W_O)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp, bias=True)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model, bias=True)

    def forward(self, x):
        return self.W_out(F.relu(self.W_in(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        scale = cfg.d_model ** -0.5
        nn.init.normal_(self.tok_embed.weight, std=scale)
        nn.init.normal_(self.pos_embed.weight, std=scale)
        nn.init.normal_(self.unembed.weight, std=scale)

    def forward(self, tokens):  # [B, T]
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    @torch.no_grad()
    def residual_last(self, tokens):
        """Residual-stream activations at the last token after all blocks.

        Hook point for SAE training — the final representation of the '='
        position, right before the unembedding. Shape: [B, d_model].
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok_embed(tokens) + self.pos_embed(pos)[None]
        for block in self.blocks:
            x = block(x)
        return x[:, -1, :]
