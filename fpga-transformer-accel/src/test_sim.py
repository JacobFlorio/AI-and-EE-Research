"""Bit-accuracy and sanity tests for the systolic simulator.

Run with:
    python -m src.test_sim
"""
from __future__ import annotations
import numpy as np
import torch
from .systolic_sim import ArrayConfig, matmul_int8, cycles_to_seconds


def test_bit_accuracy():
    """Simulator output must equal INT32 reference matmul exactly, for
    every combination of matmul shape and array size we care about."""
    rng = np.random.default_rng(0)
    cases = [
        (4, 4, 4, 4),
        (8, 8, 8, 4),
        (16, 32, 16, 8),
        (33, 65, 17, 16),         # not an array-multiple
        (128, 256, 128, 16),
        (512, 768, 512, 16),      # transformer-ish
    ]
    for M, K, N, array_n in cases:
        A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
        cfg = ArrayConfig(N=array_n)
        C, _ = matmul_int8(A, B, cfg)
        C_ref = A.astype(np.int32) @ B.astype(np.int32)
        assert np.array_equal(C, C_ref), \
            f"bit accuracy failed at M={M} K={K} N={N} array_n={array_n}"
        print(f"  M={M:>4d} K={K:>4d} N={N:>4d} array={array_n:>2d}  "
              f"bit-exact ✓")


def test_cycle_accounting():
    """Total cycles should grow predictably with problem size."""
    rng = np.random.default_rng(1)
    cfg = ArrayConfig(N=16)
    sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
    print("  (M,K,N)           cycles   ops/cycle  MAC util")
    prev_cycles = 0
    for M, K, N in sizes:
        A = rng.integers(-10, 10, size=(M, K), dtype=np.int8)
        B = rng.integers(-10, 10, size=(K, N), dtype=np.int8)
        _, stats = matmul_int8(A, B, cfg)
        ops_per_cycle = stats.mac_ops / stats.total_cycles
        print(f"  ({M},{K},{N})   {stats.total_cycles:>8d}  "
              f"{ops_per_cycle:>8.1f}  {stats.mac_utilization:.3f}")
        assert stats.total_cycles >= prev_cycles
        prev_cycles = stats.total_cycles


def test_throughput_projection():
    """End-to-end: for a transformer-sized matmul, how many GOPs/sec do we hit?"""
    cfg = ArrayConfig(N=16, clock_mhz=400.0)
    M, K, N = 512, 768, 3072   # GPT-2-small-ish FFN up-proj
    rng = np.random.default_rng(2)
    A = rng.integers(-64, 64, size=(M, K), dtype=np.int8)
    B = rng.integers(-64, 64, size=(K, N), dtype=np.int8)
    _, stats = matmul_int8(A, B, cfg)
    seconds = cycles_to_seconds(stats.total_cycles, cfg)
    gops = 2 * stats.mac_ops / 1e9  # one MAC = 2 ops (multiply + add)
    gops_per_sec = gops / seconds
    print(f"  transformer FFN ({M}x{K} @ {K}x{N}):")
    print(f"    total cycles   : {stats.total_cycles:>12,}")
    print(f"    total time     : {seconds*1000:>8.2f} ms @ {cfg.clock_mhz} MHz")
    print(f"    MAC ops        : {stats.mac_ops/1e9:>8.2f} G")
    print(f"    throughput     : {gops_per_sec:>8.1f} GOPs/s")
    print(f"    MAC utilization: {stats.mac_utilization:.1%}")


def main():
    print("=== bit-accuracy vs torch int32 matmul ===")
    test_bit_accuracy()
    print("\n=== cycle accounting ===")
    test_cycle_accounting()
    print("\n=== throughput projection ===")
    test_throughput_projection()
    print("\nall tests passed")


if __name__ == "__main__":
    main()
