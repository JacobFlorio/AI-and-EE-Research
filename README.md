# AI and EE Research

Independent graduate-level research at the intersection of electrical engineering, machine learning, and AI safety. Built by [Jacob Florio](https://github.com/JacobFlorio) as a public record of research directions I'm pursuing outside a formal lab, on a single RTX 5080.

This repository is an **index**. Projects with shippable results get promoted to their own standalone repositories; projects still in scaffolding stay here as subdirectories until they earn a repo of their own.

---

## Shipped projects

### 🧠 [mech-interp-tiny-transformer](https://github.com/JacobFlorio/mech-interp-tiny-transformer)
**A TopK sparse autoencoder recovers 5/5 ground-truth Fourier frequencies from a grokked transformer.**

A one-layer transformer trained on `(a + b) mod 113` groks, its token embedding and unembedding concentrate on the same five key Fourier frequencies, and a TopK SAE trained on the residual stream finds all five as distinct features — the cleanest SAE validation I've been able to construct with a known analytical ground truth. Runs end-to-end in ~90 seconds on an RTX 5080.

Track: **Interpretability**. 97.8% SAE variance explained, 180/195 alive features on the `k_a == k_b` diagonal, full Nanda-style grokking replication.

---

### 🔬 [edge-llm-eval-harness](https://github.com/JacobFlorio/edge-llm-eval-harness)
**INT8 essentially lossless, INT4 costs +27% perplexity on Qwen2.5-0.5B — plus two real eval bugs caught and documented.**

A hardware-aware evaluation harness that measures LLM capability degradation *per quantization step*. The two bug stories are the point of the project: a regex-based arithmetic scorer that gave a nonsense FP16-worse-than-INT4 inversion, and a fp16 logprob-accumulation bug that was silently hiding the real FP16-vs-INT8 gap behind fp16's coarse rounding grid. Both fixes shipped with full post-mortems.

Track: **Evaluation × hardware for AI**. Working CUDA backends at FP16/INT8/INT4; planned `llama.cpp` Q4_K_M, Jetson INT8, and FPGA systolic-array backends.

---

### 📡 [sae-rf-classifier](https://github.com/JacobFlorio/sae-rf-classifier)
**A TopK SAE trained on a CNN modulation classifier rediscovers classical cumulant and envelope features — 3.83× more correlated with the classical feature set than matched PCA directions (5-seed median), and a causal ablation shows each classical family is load-bearing for the specific modulations classical theory says it should be.**

When you train a sparse autoencoder on the penultimate activations of a CNN that classifies 11 digital modulation schemes, its features match the Swami-Sadler (2000) hand-designed cumulant/envelope feature set with mean max-|r| of 0.50 (5-seed median) vs PCA's 0.13. Multi-seed causal ablation: `C41_mag` → BPSK is unanimous across 5 seeds (median Δ −1.00), `phase_std` → CPFSK likewise (median Δ −0.99), `spec_bandwidth` → 8PSK/QAM16 (5/5), `env_var` → QAM16 (5/5). A rare falsifiable SAE interpretability result in a non-language domain with a known analytical ground truth.

Track: **Interpretability × EE**. Companion to `mech-interp-tiny-transformer`.

---

### ⚡ [fpga-transformer-accel](https://github.com/JacobFlorio/fpga-transformer-accel)
**Bit-accurate Python simulator of a parameterized INT8 systolic MAC array with roofline + 28nm energy analysis — gpt2-small at 41 μJ/token, gpt2-nano at 0.6 μJ/token at batch 128.**

A cycle-accurate Python simulator of a parameterized NxN INT8 systolic array, validated bit-for-bit against `torch.int32` matmul for every shape, with cycle counting through full transformer forward passes. Produces throughput sweeps across array sizes (gpt2-small holds ~33% MAC utilization up to N=256, gpt2-nano collapses past N=64), classic compute-vs-bandwidth roofline plots, and 28nm-CMOS energy-per-token projections from Horowitz 2014 constants. The canonical finding — **weight DRAM dominates inference energy at batch 1, and batch amortization is how you claw it back** — is the right first-order lesson for anyone building inference accelerators. The SystemVerilog RTL in `src/rtl/` encodes the same per-cell behavior the simulator models; cocotb cross-check is the next step.

Track: **Hardware for AI**. Meant to plug into `edge-llm-eval-harness` as an `fpga_systolic` backend so the same quantized model runs across CUDA, llama.cpp, and the simulated accelerator through one eval harness.

---

## In-progress projects (scaffolded here)

Each of the projects below has a README with a research question and runnable starter code, and will be promoted to its own repository once it has shippable results.

| Project | Track | Research question |
|---|---|---|
| [neural-rf-frontend](neural-rf-frontend/) | RF / SDR | Can a <500k-param CNN classify modulation at sub-0 dB SNR on an RTL-SDR? |
| [tinyml-edge-anomaly](tinyml-edge-anomaly/) | Embedded | Sub-100 µJ/inference bearing fault detection on Cortex-M4? |
| [rl-power-converter](rl-power-converter/) | Power electronics | Does deep RL beat tuned PID on buck converter load transients? |
| [neural-beamforming-phased-array](neural-beamforming-phased-array/) | RF / DSP | Can a learned beamformer match MVDR with 10× lower latency? |

## Focus
Three overlapping tracks:
- **Interpretability, evaluation, and safety** — mechanistic interpretability of tiny transformers, hardware-aware LLM evaluation, sparse autoencoders in non-language domains.
- **Hardware for AI** — FPGA accelerators for transformer inference, systolic arrays, and hardware-in-the-loop eval.
- **EE × ML applications** — SDR cognitive radio, phased arrays, power electronics, embedded TinyML.

## Methodology
Each project follows: literature review → reproducible baseline → novel contribution → quantitative evaluation → writeup. Results live as Jupyter notebooks and short technical reports under each project's `docs/`.

## Hardware
Primary rig: RTX 5080, Ryzen 9 9950X3D, 64 GB RAM. Secondary: RTL-SDR v3, STM32L4 Nucleo, TI LAUNCHXL-F28379D, Artix-7 dev board.

## License
MIT for code; CC-BY-4.0 for writeups. See [LICENSE](LICENSE).
