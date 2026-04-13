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

## In-progress projects (scaffolded here)

Each of the projects below has a README with a research question and runnable starter code, and will be promoted to its own repository once it has shippable results.

| Project | Track | Research question |
|---|---|---|
| [sae-rf-classifier](sae-rf-classifier/) | Interp × EE | Do sparse-autoencoder features on a CNN modulation classifier match classical cyclostationary features? |
| [neural-rf-frontend](neural-rf-frontend/) | RF / SDR | Can a <500k-param CNN classify modulation at sub-0 dB SNR on an RTL-SDR? |
| [fpga-transformer-accel](fpga-transformer-accel/) | Hardware for AI | Energy-per-token floor for INT8 transformer inference on a mid-range FPGA? |
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
