# Mechanistic Interpretability of Tiny Transformers

Train small decoder-only transformers from scratch on algorithmic tasks (modular arithmetic, induction, indirect object identification) and reverse-engineer the circuits that implement them.

## Research question
Do the canonical circuits found in larger models (induction heads, previous-token heads, name-mover heads) emerge at GPT-nano scale (≤4 layers, ≤256 d_model), and if so, at what training step do they crystallize? Can sparse autoencoders trained on these models recover the ground-truth features we know are there?

## Approach
1. Train a 2-layer, 128-dim transformer on modular addition (replicates Nanda et al. "Progress measures for grokking").
2. Track Fourier features throughout training; confirm grokking phase transition.
3. Train sparse autoencoders on residual stream activations.
4. Score SAE features against the known Fourier basis — quantify automated-interp recovery rate.
5. Extend to induction on synthetic bigram data.

## Why this project
Closest approximation to Anthropic-style interp research that can be done on a single RTX 5080. Every step is publishable as a blog post / notebook.

## Deliverables
- `src/train.py` — transformer + modular arithmetic training
- `src/sae.py` — sparse autoencoder implementation
- `notebooks/grokking_circuits.ipynb` — Fourier feature analysis
- `notebooks/sae_feature_recovery.ipynb` — SAE vs ground-truth
- `docs/report.md` — writeup targeting LessWrong / Alignment Forum

## Hardware
RTX 5080, 9950X3D, 64GB — comfortably sufficient.
