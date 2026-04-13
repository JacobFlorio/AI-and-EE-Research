# FPGA Transformer Accelerator

A systolic-array accelerator for INT8 transformer inference, targeting a mid-range FPGA (Xilinx Artix-7 / Lattice ECP5).

## Research question
What is the practical energy-per-token floor for decoder-only transformer inference (GPT-2 small scale) on a sub-$200 FPGA, and where does the bottleneck actually live — MAC throughput, on-chip SRAM bandwidth, or off-chip DRAM?

## Approach
1. Build a parameterized systolic MAC array in SystemVerilog.
2. Implement fused LayerNorm + softmax in fixed-point.
3. Quantize GPT-2 small to INT8 with per-channel scales.
4. Roofline analysis: measured GOPS/W vs theoretical.
5. Compare against a Jetson Orin Nano baseline.

## Tooling
- SystemVerilog + cocotb for verification
- Vivado / Yosys for synthesis
- PyTorch for quantization-aware training

## Deliverables
- RTL + testbenches in `src/rtl/`
- Synthesis reports in `results/`
- Writeup with roofline plots in `docs/`
