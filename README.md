# EE × AI Research Portfolio

Independent graduate-level research at the intersection of electrical engineering, machine learning, and AI safety. Built by [Jacob Florio](https://github.com/JacobFlorio) as a public record of research directions pursued outside a formal lab. Each subdirectory is a self-contained project with a research question, methodology, and reproducible results.

## Focus
Three overlapping tracks:
- **EE × ML applications** — SDR cognitive radio, phased arrays, power electronics, embedded TinyML
- **Hardware for AI** — FPGA accelerators for transformer inference
- **Interpretability, evaluation, and safety** — mech-interp of tiny transformers, hardware-aware LLM evals, sparse autoencoders in non-language domains

## Projects

| Project | Track | Research Question |
|---|---|---|
| [mech-interp-tiny-transformer](mech-interp-tiny-transformer/) | Interp | Do canonical circuits emerge at GPT-nano scale, and can SAEs recover them? |
| [edge-llm-eval-harness](edge-llm-eval-harness/) | Evals | How much capability is lost per quantization step, per hardware backend? |
| [sae-rf-classifier](sae-rf-classifier/) | Interp × EE | Do SAE features on an RF classifier match classical cyclostationary features? |
| [neural-rf-frontend](neural-rf-frontend/) | RF / SDR | Can a <500k-param CNN classify modulation at sub-0dB SNR on an RTL-SDR? |
| [fpga-transformer-accel](fpga-transformer-accel/) | Hardware for AI | Energy-per-token floor for INT8 transformer inference on a mid-range FPGA? |
| [tinyml-edge-anomaly](tinyml-edge-anomaly/) | Embedded | Sub-100 µJ/inference bearing fault detection on Cortex-M4? |
| [rl-power-converter](rl-power-converter/) | Power Electronics | Does deep RL beat tuned PID on buck converter load transients? |
| [neural-beamforming-phased-array](neural-beamforming-phased-array/) | RF / DSP | Can a learned beamformer match MVDR with 10× lower latency? |

## Methodology
Each project follows: literature review → reproducible baseline → novel contribution → quantitative evaluation → writeup. Results live as Jupyter notebooks and short technical reports under each project's `docs/`.

## Hardware
Primary rig: RTX 5080, Ryzen 9 9950X3D, 64 GB RAM. Secondary: RTL-SDR v3, STM32L4 Nucleo, TI LAUNCHXL-F28379D, Artix-7 dev board.

## License
MIT for code. Writeups CC-BY-4.0.
