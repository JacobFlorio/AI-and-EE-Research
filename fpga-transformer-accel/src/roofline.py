"""Roofline + energy analysis for the systolic accelerator.

Given an array configuration and a list of matmul shapes, classify
each matmul as compute-bound or memory-bound against a simple roofline
model (peak INT8 compute vs on-chip/off-chip bandwidth), and produce
an energy-per-token estimate using published 28nm CMOS numbers.

All energy figures are from Horowitz 2014, "Computing's Energy Problem
(and what we can do about it)," ISSCC plenary. Values are canonical
ballpark numbers for 28nm logic + SRAM + DRAM, not the output of a
specific PPA tool. They're fine for a first-order portfolio roofline
study; any serious tapeout study would use the actual synthesis tool's
post-layout power report.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .systolic_sim import ArrayConfig
from .transformer_forward import TransformerShape


# Published 28nm CMOS energies (Horowitz 2014). All in pJ unless noted.
E_MUL_INT8_PJ = 0.2      # single 8-bit multiply
E_ADD_INT32_PJ = 0.1     # 32-bit add (accumulator update)
E_MAC_INT8_PJ = E_MUL_INT8_PJ + E_ADD_INT32_PJ   # one MAC
E_SRAM_PER_BYTE_PJ = 5.0      # on-chip SRAM access
E_DRAM_PER_BYTE_PJ = 640.0    # off-chip DDR3 burst access


@dataclass
class RooflineResult:
    matmul_shape: tuple[int, int, int]
    intensity_ops_per_byte: float
    peak_compute_gops: float
    peak_bw_gb_s: float
    roofline_gops: float
    is_memory_bound: bool


def matmul_arithmetic_intensity(M: int, K: int, N: int,
                                  a_width: int = 8,
                                  w_width: int = 8,
                                  o_width: int = 32) -> float:
    """Ops per byte for a dense matmul — counts each MAC as 2 ops.

    Bytes = input activation + weight + output accumulator.
    """
    ops = 2 * M * K * N
    bytes_act = M * K * a_width // 8
    bytes_w = K * N * w_width // 8
    bytes_out = M * N * o_width // 8
    return ops / max(bytes_act + bytes_w + bytes_out, 1)


def roofline(cfg: ArrayConfig, shapes: Iterable[tuple[int, int, int]],
             bw_gb_s: float | None = None) -> list[RooflineResult]:
    """Classify each matmul as compute-bound or memory-bound.

    `bw_gb_s` defaults to DRAM BW (off-chip). If you want the on-chip
    roofline, pass cfg.sram_bw_gb_s.
    """
    if bw_gb_s is None:
        bw_gb_s = cfg.dram_bw_gb_s

    peak_gops = 2 * cfg.N * cfg.N * cfg.clock_mhz / 1000.0
    out = []
    for M, K, N in shapes:
        intensity = matmul_arithmetic_intensity(M, K, N)
        bw_bound = bw_gb_s * intensity   # GOPs/s if fully BW-limited
        roofline_gops = min(peak_gops, bw_bound)
        out.append(RooflineResult(
            matmul_shape=(M, K, N),
            intensity_ops_per_byte=intensity,
            peak_compute_gops=peak_gops,
            peak_bw_gb_s=bw_gb_s,
            roofline_gops=roofline_gops,
            is_memory_bound=roofline_gops < peak_gops - 1e-6,
        ))
    return out


def forward_pass_matmul_shapes(shape: TransformerShape, batch: int = 1
                                ) -> list[tuple[int, int, int, str]]:
    """Enumerate every dense matmul in one forward pass, labeled by op name."""
    T = shape.n_ctx * batch
    d = shape.d_model
    d_ff = shape.d_ff
    ops = []
    for layer in range(shape.n_layers):
        ops.append((T, d, d, f"L{layer}.Q"))
        ops.append((T, d, d, f"L{layer}.K"))
        ops.append((T, d, d, f"L{layer}.V"))
        ops.append((T, d, d, f"L{layer}.OutProj"))
        ops.append((T, d, d_ff, f"L{layer}.FFN_up"))
        ops.append((T, d_ff, d, f"L{layer}.FFN_down"))
    ops.append((T, d, shape.vocab_size, "Unembed"))
    return ops


def forward_pass_energy(shape: TransformerShape, batch: int, cfg: ArrayConfig,
                         weights_fit_in_sram: bool | None = None) -> dict:
    """Estimate forward-pass energy for one batch of `batch * n_ctx` tokens.

    Energy components:
      - arithmetic: E_MAC_INT8_PJ * total MACs
      - SRAM traffic: 5 pJ/byte for every byte read/written inside the chip
      - DRAM traffic: 640 pJ/byte for off-chip weight loading — paid once
                      per batch if the weight set fits in SRAM, otherwise
                      paid per tile.

    The `weights_fit_in_sram` flag decides which regime we're in. Default
    behavior is to check: total_weight_bytes <= cfg.sram_bytes.
    """
    total_macs = 0
    total_weight_bytes = 0
    total_act_bytes = 0
    total_out_bytes = 0
    for M, K, N, _ in forward_pass_matmul_shapes(shape, batch):
        total_macs += M * K * N
        total_weight_bytes += K * N
        total_act_bytes += M * K
        total_out_bytes += M * N * 4   # INT32 accumulator

    if weights_fit_in_sram is None:
        weights_fit_in_sram = total_weight_bytes <= cfg.sram_bytes

    e_arith_pJ = total_macs * E_MAC_INT8_PJ
    e_sram_pJ = (total_act_bytes + total_out_bytes) * E_SRAM_PER_BYTE_PJ
    if weights_fit_in_sram:
        e_weight_dram_pJ = total_weight_bytes * E_DRAM_PER_BYTE_PJ   # one-shot
        e_weight_sram_pJ = total_weight_bytes * E_SRAM_PER_BYTE_PJ   # in-chip reads
    else:
        # Worst case: every tile pulls weights from DRAM. In practice you'd
        # have a smaller blocking scheme, so treat this as the upper bound
        # and also report a realistic "tiled DRAM reuse = 4x" figure.
        e_weight_dram_pJ = total_weight_bytes * E_DRAM_PER_BYTE_PJ
        e_weight_sram_pJ = 0.0

    e_total_pJ = e_arith_pJ + e_sram_pJ + e_weight_dram_pJ + e_weight_sram_pJ
    tokens = shape.n_ctx * batch

    return {
        "macs": int(total_macs),
        "weight_bytes": int(total_weight_bytes),
        "act_bytes": int(total_act_bytes),
        "out_bytes": int(total_out_bytes),
        "weights_fit_in_sram": bool(weights_fit_in_sram),
        "sram_bytes_budget": int(cfg.sram_bytes),
        "e_arith_mJ": e_arith_pJ / 1e9,
        "e_sram_mJ": e_sram_pJ / 1e9,
        "e_weight_dram_mJ": e_weight_dram_pJ / 1e9,
        "e_weight_sram_mJ": e_weight_sram_pJ / 1e9,
        "e_total_mJ": e_total_pJ / 1e9,
        "e_per_token_uJ": (e_total_pJ / tokens) / 1e6,
        "tokens": tokens,
    }


def classify_forward_pass(shape: TransformerShape, batch: int,
                           cfg: ArrayConfig) -> dict:
    """Run roofline across every op in a forward pass and report the
    fraction that are memory-bound vs compute-bound."""
    shapes = [(M, K, N) for M, K, N, _ in forward_pass_matmul_shapes(shape, batch)]
    rr = roofline(cfg, shapes)
    n_mem_bound = sum(1 for r in rr if r.is_memory_bound)
    n_total = len(rr)
    # Aggregate ops weighted by size
    total_ops = sum(2 * M * K * N for (M, K, N) in shapes)
    ops_mem_bound = sum(2 * M * K * N for (M, K, N), r in zip(shapes, rr)
                        if r.is_memory_bound)
    return {
        "n_ops_total": n_total,
        "n_ops_memory_bound": n_mem_bound,
        "frac_ops_memory_bound_by_count": n_mem_bound / n_total,
        "frac_ops_memory_bound_by_macs": ops_mem_bound / total_ops,
        "peak_compute_gops": rr[0].peak_compute_gops,
        "peak_bw_gb_s": rr[0].peak_bw_gb_s,
        "min_intensity": min(r.intensity_ops_per_byte for r in rr),
        "max_intensity": max(r.intensity_ops_per_byte for r in rr),
        "mean_intensity": float(np.mean([r.intensity_ops_per_byte for r in rr])),
    }
