"""cocotb sanity test for the NxN systolic array.

Feeds a known A, B through the mesh and compares accumulators against
a numpy reference. Start small (N=4), grow when RTL is stable.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np


@cocotb.test()
async def test_matmul_4x4(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.clear.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0

    rng = np.random.default_rng(0)
    A = rng.integers(-5, 5, size=(4, 4))
    B = rng.integers(-5, 5, size=(4, 4))
    expected = A @ B

    dut.en.value = 1
    for step in range(16):
        for r in range(4):
            col = step - r
            dut.a_row[r].value = int(A[r, col]) if 0 <= col < 4 else 0
        for c in range(4):
            row = step - c
            dut.b_col[c].value = int(B[row, c]) if 0 <= row < 4 else 0
        await RisingEdge(dut.clk)
    dut.en.value = 0

    for r in range(4):
        for c in range(4):
            got = dut.acc_out[r][c].value.signed_integer
            assert got == expected[r, c], f"pe[{r},{c}]: got {got}, want {expected[r,c]}"
