# Emulating Dark Matter Density Fields with Generative Models

A Bayesian autoregressive normalizing flow vs. VAEs and Gaussian Processes.

**Artificial Intelligence — Final Project, Universidad EAFIT**
Author: Camilo Bermúdez-Colorado

---

## Overview

Cosmological simulations such as Illustris-3 reproduce the filamentary "cosmic
web" of dark matter, but a single run costs millions of CPU hours. This project
trains cheap statistical **surrogates** that, once trained on a handful of
simulated examples, can generate new, statistically faithful 2D dark-matter
overdensity fields in seconds.

We frame this as a generative-modeling problem — learning the probability
distribution of 64×64 density fields — and compare three families of models:

| Model | What it is | Code |
|-------|------------|------|
| **BTM** (Bayesian Transport Map) | A Bayesian autoregressive normalizing flow with Gaussian-process conditioners (`batram` library). Best scores, exact likelihood, closed-form conditional sampling. | `train_model.py`, `main.py`, `btm_samples.py` |
| **VAE** | MLP variational autoencoder (hidden widths [1024, 512, 256, 128], 32-dim latent). Close second. | `vae.py`, `vaes.ipynb` |
| **GP** | Zero-mean Gaussian process with a Matérn covariance (classical baseline; fit in R with the `fields` package). Collapses when data are scarce. | `gp/matern_astro.R` |

Models are compared with the **log score** (negative log-likelihood of held-out
fields; lower is better) as the number of training fields grows over
`n ∈ {5, 10, 30, 50, 100, 160}`.

## Data

- `data/locs.csv` — pixel locations, shape `(4096, 2)`.
- `data/stacked.csv` — 200 flattened density fields, shape `(4096, 200)`
  (transposed to `(200, 4096)` on load).

Fields are standardized and split into the first 160 (train) and the next 40
(test). The full dataset of 749 fields was processed from Illustris simulation
slices by Tamošiūnas et al. (2021).

## Repository structure

```
.
├── train_model.py     # Train + evaluate the BTM flow for one training size
├── main.py            # Run the BTM log-score-vs-n sweep and plot the result
├── btm_samples.py     # Generate BTM unconditional + conditional sample figures
├── vae.py             # VAE model, training/eval sweep, and sample generation
├── vaes.ipynb         # Original VAE notebook (exploration + result figures)
├── gp/                # Gaussian-process Matérn baseline (R)
│   ├── matern_astro.R             # Fit + evaluate the GP log score
│   └── logscores_matern_astro.csv # GP log scores per training size
├── data/              # Input data
│   ├── locs.csv                   # Pixel locations
│   └── stacked.csv                # 200 density fields
├── figures/           # Generated figures
│   ├── btm_loss_curves.png            # BTM train/test loss curves
│   ├── vae_conditional_samples.png    # VAE conditional samples
│   └── vae_unconditional_samples.png  # VAE unconditional samples
├── requirements.txt
├── AI_USAGE.md        # AI usage statement
└── scratch/           # Exploratory / tutorial notebooks (not part of the pipeline)
```

## Installation

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. The Bayesian Transport Map model and ordering helpers (not on PyPI):
pip install "git+https://github.com/katzfuss-group/batram.git"
pip install "git+https://github.com/katzfuss-group/veccs.git"
```

Python ≥ 3.10 is recommended.

## How to run

### Bayesian Transport Map (flow)

```bash
# Train + evaluate a single model on n training fields
python train_model.py --train_size 160 --save_dir log_score_results

# Run the full sweep over n ∈ {5, 10, 30, 50, 100, 160} and plot log score vs n
python main.py

# Generate unconditional and conditional sample figures
python btm_samples.py
```

### Variational autoencoder

```bash
# Log-score-vs-training-size sweep (saves results + plot to vae_results/)
python vae.py sweep --save_dir vae_results

# Train a final VAE and produce unconditional + conditional sample figures
python vae.py samples --train_size 160 --epochs 150
```

### Gaussian-process baseline (R)

Requires R with the `fields` and `mvtnorm` packages
(`install.packages(c("fields", "mvtnorm"))`).

```bash
cd gp
Rscript matern_astro.R
```

Precomputed GP log scores per training size are in
`gp/logscores_matern_astro.csv` (used in the comparison plot in the report).

## Results

- The **flow (BTM)** attains the best (lowest) log score at every training size
  and is the most data-efficient — well-behaved even at `n = 5`.
- The **VAE** is consistently close behind; its latent bottleneck blurs
  fine-scale detail.
- The **GP** fails in the low-data regime (log score > 40,000 at `n = 5`),
  confirming that a Gaussian assumption is a poor fit for the non-Gaussian
  cosmic web.

Fixing the first `k` flow coordinates (the largest spatial scales) and
resampling the rest yields conditional samples that preserve large-scale
filamentary structure while fine detail varies — exactly what an emulator
should do. See `figures/btm_loss_curves.png`,
`figures/vae_conditional_samples.png`, and
`figures/vae_unconditional_samples.png`.

## AI usage statement

See [AI_USAGE.md](AI_USAGE.md).
