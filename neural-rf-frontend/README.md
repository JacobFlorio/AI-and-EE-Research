# Neural RF Frontend

Cognitive-radio experiment: a CNN that classifies RF modulation schemes (BPSK, QPSK, 8PSK, QAM16, QAM64, GFSK, CPFSK, PAM4, WBFM, AM-SSB, AM-DSB) from raw IQ samples captured with an RTL-SDR.

## Research question
At what SNR floor can a compact (<500k-parameter) CNN match or beat classical cyclostationary feature detectors on the RadioML 2018.01A dataset, and how does that floor shift when training data comes from a real RTL-SDR rather than simulation?

## Approach
1. Baseline on RadioML 2018.01A with a residual 1D-CNN.
2. Capture a small real-world IQ dataset with an RTL-SDR v3 + HackRF transmitter.
3. Domain adaptation (CORAL / adversarial) from sim → real.
4. Quantize to INT8 and profile on a Raspberry Pi 5.

## Deliverables
- `src/` training + capture scripts
- `results/` confusion matrices vs SNR
- `docs/report.pdf` 4-page technical writeup
