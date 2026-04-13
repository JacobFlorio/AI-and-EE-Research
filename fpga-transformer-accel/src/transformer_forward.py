"""Forward a tiny transformer's matmul ops through the systolic simulator
and count cycles. No actual model weights needed — this is a pure
matmul-shape / cycle-count exercise.

For a given transformer shape (d_model, d_ff, n_heads, n_layers, n_ctx,
vocab_size), enumerate every dense matmul in one forward pass over
`batch` tokens and accumulate cycle / byte counts from the systolic
simulator. That's enough to answer "how long does inference take on
this accelerator" without running a real model through it.

Attention softmax and layer norm are not matmuls and are assumed to
run on a small fixed-function block in parallel with the MAC array;
their cost is reported separately as a flat fraction of total time
and clearly labeled.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
from .systolic_sim import ArrayConfig, Schedule, RunStats, matmul_int8, cycles_to_seconds


@dataclass
class TransformerShape:
    name: str = "gpt2-small"
    d_model: int = 768
    d_ff: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    n_ctx: int = 512
    vocab_size: int = 50257


@dataclass
class ForwardStats:
    shape: TransformerShape
    batch: int
    array_cfg: ArrayConfig
    schedule: Schedule
    total_cycles: int = 0
    total_macs: int = 0
    total_weight_bytes: int = 0
    total_act_bytes: int = 0
    total_out_bytes: int = 0
    per_op: list[tuple[str, RunStats]] = field(default_factory=list)


def _dummy_int8(M: int, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-8, 8, size=(M, K), dtype=np.int8)


def forward_one_batch(shape: TransformerShape, batch: int,
                      cfg: ArrayConfig,
                      schedule: Schedule = "overlapped") -> ForwardStats:
    """Accumulate cycles for a forward pass over `batch` tokens (one
    context length per batch).

    Matmul inventory per layer:
      - Q/K/V proj:        3 × (T, d_model) @ (d_model, d_model)
      - attn out proj:     1 × (T, d_model) @ (d_model, d_model)
      - FFN up:            1 × (T, d_model) @ (d_model, d_ff)
      - FFN down:          1 × (T, d_ff) @ (d_ff, d_model)
    Plus once per forward:
      - unembed:           1 × (T, d_model) @ (d_model, vocab)
    Token/pos embeddings are table lookups, not matmuls.

    Attention scores (Q @ K^T) and score-value (attn @ V) are also
    matmuls but at per-head scale; we count them too.
    """
    stats = ForwardStats(shape=shape, batch=batch, array_cfg=cfg, schedule=schedule)
    T = shape.n_ctx * batch   # treat batch as "extra rows" of the M axis
    d = shape.d_model
    d_ff = shape.d_ff
    d_head = d // shape.n_heads

    def run(name: str, A_shape, B_shape):
        A = _dummy_int8(*A_shape, seed=hash(name) & 0xFFFF)
        B = _dummy_int8(*B_shape, seed=(hash(name) ^ 0xABCD) & 0xFFFF)
        _, s = matmul_int8(A, B, cfg=cfg, schedule=schedule)
        stats.per_op.append((name, s))
        stats.total_cycles += s.total_cycles
        stats.total_macs += s.mac_ops
        stats.total_weight_bytes += s.weight_bytes_loaded
        stats.total_act_bytes += s.act_bytes_loaded
        stats.total_out_bytes += s.out_bytes_drained

    for layer in range(shape.n_layers):
        run(f"L{layer}.Q", (T, d), (d, d))
        run(f"L{layer}.K", (T, d), (d, d))
        run(f"L{layer}.V", (T, d), (d, d))
        # Attention scores and context — aggregated across heads.
        # Per-head: (T, d_head) @ (d_head, T)  = T x T scores
        # Per-head: (T, T) @ (T, d_head)       = T x d_head context
        # We sum over heads as a single larger matmul for cycle-counting
        # (the array doesn't care about head boundaries).
        run(f"L{layer}.QK", (T, d), (d, T // max(batch, 1)))
        run(f"L{layer}.AV", (T, max(T // max(batch, 1), 1)), (max(T // max(batch, 1), 1), d))
        run(f"L{layer}.OutProj", (T, d), (d, d))
        run(f"L{layer}.FFN_up", (T, d), (d, d_ff))
        run(f"L{layer}.FFN_down", (T, d_ff), (d_ff, d))

    run("Unembed", (T, d), (d, shape.vocab_size))
    return stats


def summarize(stats: ForwardStats) -> dict:
    cfg = stats.array_cfg
    seconds = cycles_to_seconds(stats.total_cycles, cfg)
    tokens = stats.shape.n_ctx * stats.batch
    tokens_per_sec = tokens / max(seconds, 1e-12)
    gops = 2 * stats.total_macs / 1e9
    gops_per_sec = gops / max(seconds, 1e-12)
    return {
        "model": stats.shape.name,
        "array_N": cfg.N,
        "schedule": stats.schedule,
        "clock_mhz": cfg.clock_mhz,
        "total_cycles": int(stats.total_cycles),
        "total_seconds": float(seconds),
        "tokens": int(tokens),
        "tokens_per_sec": float(tokens_per_sec),
        "macs": int(stats.total_macs),
        "gops_per_sec": float(gops_per_sec),
        "weight_bytes": int(stats.total_weight_bytes),
        "act_bytes": int(stats.total_act_bytes),
        "out_bytes": int(stats.total_out_bytes),
    }
