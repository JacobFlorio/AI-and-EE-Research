# Edge LLM Eval Harness

A hardware-aware evaluation harness for quantized small language models across GPU, Jetson, and FPGA targets. Measures capability degradation *per quantization step* and *per hardware backend*, not just on one machine.

## Research question
How much capability (MMLU, HellaSwag, TruthfulQA, IFEval) does a 1B–7B model lose going from FP16 → INT8 → INT4 → INT2, and does the answer change when the same weights run on different hardware (CUDA, Jetson INT8 DLA, FPGA systolic array)? Which quantization errors are *systematic* (hurt specific capabilities like multi-step reasoning) vs *random*?

## Why this matters for alignment
Edge deployments increasingly ship aggressively quantized models. Capability evals are usually run on FP16 reference hardware, so degradation is invisible until users hit it. A hardware-aware eval harness is a small but concrete contribution to deployment-time safety.

## Approach
1. Wrap `lm-evaluation-harness` with a hardware-routing layer.
2. Targets: HuggingFace CUDA (baseline), llama.cpp Q4/Q2, TensorRT-LLM INT8, Jetson Orin Nano INT8, (stretch) FPGA systolic array from sister project.
3. Eval suite: MMLU, HellaSwag, TruthfulQA, IFEval, a custom multi-step arithmetic probe.
4. Per-quantization + per-hardware capability delta matrices.
5. Identify which capabilities degrade *disproportionately* under aggressive quantization.

## Deliverables
- `src/harness/` — hardware-routing layer
- `configs/` — per-model / per-hardware configs
- `results/capability_matrix.md` — headline delta table
- `docs/report.md` — writeup

## Sister project
Links to `fpga-transformer-accel` for the FPGA backend.
