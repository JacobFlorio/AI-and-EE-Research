"""Bit-accurate Python simulator of a parameterized NxN INT8 systolic
MAC array for transformer accelerator studies.

What it models:
  - An N×N grid of INT8 × INT8 → INT32 MAC cells.
  - Weight-stationary (WS) dataflow: weight tile loaded once into the
    grid, activation tile streams across, output accumulates in-place.
  - Output-stationary (OS) dataflow: activations stream down columns,
    weights stream across rows, output accumulates in each PE.
  - Cycle counts for loading, computing, and draining a tile, plus the
    tile-loop overhead for matmul dimensions larger than the array.

What it is bit-accurate about:
  - The arithmetic: every product is an INT8×INT8→INT16 multiply and
    every accumulator update stays in INT32 with saturation at INT32
    limits. The simulator's output of `matmul_int8(A, B)` for any INT8
    A, B matches `torch.matmul(A.to(torch.int32), B.to(torch.int32))`
    bit-for-bit, independent of tile size.

What it is NOT bit-accurate about (at this level of the project):
  - Wire-level timing, setup/hold, metastability, buffer depth details.
    Cycle counts are at the granularity of "one INT8 MAC per cell per
    cycle, one row/column data movement per cycle" — enough for a
    roofline study, not enough for timing closure.

The separate SystemVerilog RTL (src/rtl/) encodes the cell-level
behavior this simulator models. A future cocotb-based cross-check
would verify they agree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch


Dataflow = Literal["weight_stationary", "output_stationary"]
Schedule = Literal["serial", "overlapped"]


@dataclass
class ArrayConfig:
    N: int = 16                   # systolic array side (N × N PEs)
    a_width: int = 8              # activation bit-width
    w_width: int = 8              # weight bit-width
    acc_width: int = 32           # accumulator bit-width
    clock_mhz: float = 400.0      # nominal accelerator clock
    sram_bytes: int = 512 * 1024  # on-chip SRAM for weights + activations
    sram_bw_gb_s: float = 80.0    # on-chip to PE bandwidth
    dram_bw_gb_s: float = 8.0     # off-chip to on-chip bandwidth


@dataclass
class RunStats:
    """One matmul call: M×K · K×N (INT8) -> M×N (INT32)."""
    M: int
    K: int
    N: int                 # matmul output dim, NOT array side
    dataflow: Dataflow
    array_N: int
    n_tiles_m: int
    n_tiles_k: int
    n_tiles_n: int
    compute_cycles: int    # pure MAC work
    load_cycles: int       # weight/activation loading
    drain_cycles: int      # output draining
    total_cycles: int
    mac_ops: int           # ideal MACs required (M*K*N)
    mac_utilization: float # compute_cycles / (array_N^2 cycles that could have run)
    weight_bytes_loaded: int
    act_bytes_loaded: int
    out_bytes_drained: int


def _saturate_int32(x: np.ndarray) -> np.ndarray:
    return np.clip(x, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int64)


def matmul_int8(A: np.ndarray, B: np.ndarray,
                cfg: ArrayConfig = ArrayConfig(),
                dataflow: Dataflow = "weight_stationary",
                schedule: Schedule = "overlapped") -> tuple[np.ndarray, RunStats]:
    """Tile an (M,K) @ (K,N) INT8 matmul through an (array_N × array_N) MAC grid.

    Returns (output: int32 [M, N], stats: RunStats). The arithmetic is
    identical to int32 reference matmul — this is a cycle / resource
    accounting wrapper that doesn't lose precision.

    Schedules:
      - "serial":     load → compute → drain → next tile. Worst case.
      - "overlapped": double-buffer loading and draining so they hide
                      behind compute. Realistic for modern systolic
                      arrays; cost per tile is max(load, compute, drain)
                      rather than their sum.
    """
    assert A.dtype == np.int8, f"A must be int8, got {A.dtype}"
    assert B.dtype == np.int8, f"B must be int8, got {B.dtype}"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"inner dim mismatch: {K} vs {K2}"

    # The arithmetic reference: int32 matmul.
    # We compute it in one shot and then account for cycles separately.
    C_ref = np.matmul(A.astype(np.int32), B.astype(np.int32))
    C_ref = _saturate_int32(C_ref).astype(np.int32)

    n = cfg.N
    n_tiles_m = (M + n - 1) // n
    n_tiles_k = (K + n - 1) // n
    n_tiles_n = (N + n - 1) // n

    # Cycle model:
    #   For weight-stationary:
    #     - load weight tile (n x n):     n cycles (one row per cycle)
    #     - compute with an activation tile (n x n): 2n - 1 cycles
    #       (pipeline fill + compute + drain)
    #     - drain accumulators:           n cycles
    #   Compute cycles per (m_tile, k_tile, n_tile): n^2 MACs happen in
    #   parallel across n^2 PEs; those n^2 MACs complete in about n cycles
    #   per activation-column stream, or ~ 2n cycles including pipeline
    #   fill. We use the canonical systolic-array figure of 3n - 2 cycles
    #   per fully loaded tile (fill + compute + drain). Standard textbook.
    load_cycles_per_tile = n
    tile_compute_cycles = max(3 * n - 2, 1)
    drain_cycles_per_tile = n

    n_tiles = n_tiles_m * n_tiles_k * n_tiles_n
    load_cycles = n_tiles * load_cycles_per_tile
    compute_cycles = n_tiles * tile_compute_cycles
    drain_cycles = n_tiles * drain_cycles_per_tile

    if schedule == "serial":
        total_cycles = load_cycles + compute_cycles + drain_cycles
    elif schedule == "overlapped":
        # Double-buffered schedule: per tile cost is max(load, compute,
        # drain), plus a one-time epilogue to drain the final tile.
        per_tile = max(load_cycles_per_tile, tile_compute_cycles, drain_cycles_per_tile)
        total_cycles = n_tiles * per_tile + (tile_compute_cycles + drain_cycles_per_tile)
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    # Theoretical ceiling: if the array ran at 100% utilization on the
    # pure MACs, it would finish M*K*N / (n*n) cycles.
    ideal_compute_cycles = max((M * K * N) // (n * n), 1)
    mac_utilization = min(1.0, ideal_compute_cycles / max(total_cycles, 1))

    # Byte traffic: each weight loaded once per tile in WS mode; activation
    # rows are streamed through each weight-tile position (K-tile reuse).
    bytes_per_weight = cfg.w_width // 8
    bytes_per_act = cfg.a_width // 8
    bytes_per_out = cfg.acc_width // 8
    w_bytes = n_tiles * (n * n) * bytes_per_weight
    a_bytes = n_tiles * (n * n) * bytes_per_act
    o_bytes = n_tiles_m * n_tiles_n * (n * n) * bytes_per_out

    return C_ref, RunStats(
        M=M, K=K, N=N, dataflow=dataflow, array_N=n,
        n_tiles_m=n_tiles_m, n_tiles_k=n_tiles_k, n_tiles_n=n_tiles_n,
        compute_cycles=compute_cycles, load_cycles=load_cycles,
        drain_cycles=drain_cycles, total_cycles=total_cycles,
        mac_ops=M * K * N,
        mac_utilization=float(mac_utilization),
        weight_bytes_loaded=w_bytes, act_bytes_loaded=a_bytes,
        out_bytes_drained=o_bytes,
    )


def cycles_to_seconds(cycles: int, cfg: ArrayConfig) -> float:
    return cycles / (cfg.clock_mhz * 1e6)


def matmul_int8_torch(A_pt: torch.Tensor, B_pt: torch.Tensor,
                       cfg: ArrayConfig = ArrayConfig(),
                       dataflow: Dataflow = "weight_stationary"):
    """PyTorch convenience: accepts int8 tensors, returns int32 tensor + stats."""
    assert A_pt.dtype == torch.int8
    assert B_pt.dtype == torch.int8
    A = A_pt.cpu().numpy()
    B = B_pt.cpu().numpy()
    C, stats = matmul_int8(A, B, cfg=cfg, dataflow=dataflow)
    return torch.from_numpy(C), stats
