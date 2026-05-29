"""Variational Autoencoder (VAE) baseline for emulating dark-matter density fields.

This is the script version of the experiments in ``vaes.ipynb``. It defines the
MLP VAE described in the report (hidden widths [1024, 512, 256, 128], 32-dim
latent space), trains it on the standardized 64x64 fields, and evaluates an
ELBO-based log score on the held-out test fields.

Examples
--------
Train/evaluate the log-score-vs-training-size sweep and save results::

    python vae.py sweep --save_dir vae_results

Train a single model and generate unconditional + conditional sample figures::

    python vae.py samples --train_size 160 --epochs 150
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


def load_data(locs_path="locs.csv", obs_path="stacked.csv"):
    """Load fields, sort by location, and standardize using train statistics.

    Returns (train_obs, test_obs, train_mean, train_std) where the first 160
    fields are training and the next 40 are test, matching the report's split.
    """
    locs = pd.read_csv(locs_path, header=None).values
    obs = torch.as_tensor(pd.read_csv(obs_path, header=None).to_numpy(), dtype=torch.float32)
    obs = obs.T  # (200, 4096): 200 fields, each 4096 pixels

    order = np.lexsort((locs[:, 1], locs[:, 0]))
    locs = locs[order]
    obs = obs[:, order]

    train_obs_raw = obs[:160, :]
    test_obs_raw = obs[160:200, :]

    train_mean = train_obs_raw.mean(dim=0, keepdim=True)
    train_std = train_obs_raw.std(dim=0, keepdim=True)

    train_obs = (train_obs_raw - train_mean) / train_std
    test_obs = (test_obs_raw - train_mean) / train_std

    return train_obs, test_obs, train_mean, train_std


class VAE(nn.Module):
    """MLP variational autoencoder with a Gaussian likelihood head."""

    def __init__(self, input_dim, latent_dim=32, hidden_dims=(1024, 512, 256, 128)):
        super().__init__()
        hidden_dims = list(hidden_dims)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.encoder_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            self.decoder_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)

        self.log_scale = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = z
        for layer in self.decoder_layers:
            h = F.relu(layer(h))
        return self.final_layer(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def log_prob(self, x):
        """ELBO estimate of log p(x); used as the (approximate) log score."""
        recon_x, mu, logvar = self.forward(x)

        log_scale = self.log_scale.expand_as(x)
        rec_log_prob = -0.5 * torch.sum(
            torch.pow((x - recon_x) / torch.exp(log_scale), 2) + log_scale + np.log(2 * np.pi),
            dim=-1,
        )
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return rec_log_prob - kl_div


def train_and_evaluate_vae(train_obs, test_obs, n_samples, epochs=100,
                           batch_size=32, latent_dim=32, lr=1e-3, verbose=True):
    """Train the VAE on ``n_samples`` fields and return (avg log score, model)."""
    input_dim = train_obs.shape[1]
    train_subset = train_obs[:n_samples]

    train_loader = DataLoader(
        TensorDataset(train_subset),
        batch_size=min(batch_size, n_samples),
        shuffle=True,
    )

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = 0.0
        for (data,) in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            recon_loss = F.mse_loss(recon_batch, data, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

    vae.eval()
    test_scores = []
    with torch.no_grad():
        for i in range(len(test_obs)):
            # log score is the negative log-likelihood (lower is better)
            log_score = -vae.log_prob(test_obs[i]).item()
            test_scores.append(log_score)
    return float(np.mean(test_scores)), vae


def run_sweep(sample_sizes, epochs, save_dir):
    train_obs, test_obs, _, _ = load_data()
    os.makedirs(save_dir, exist_ok=True)

    avg_log_scores = []
    for n in sample_sizes:
        print(f"\nTraining VAE with n = {n} samples")
        avg_score, _ = train_and_evaluate_vae(train_obs, test_obs, n, epochs=epochs)
        avg_log_scores.append(avg_score)
        print(f"Average log score for n = {n}: {avg_score:.4f}")

        with open(os.path.join(save_dir, f"vae_result_n{n}.pkl"), "wb") as f:
            pickle.dump({"train_size": n, "avg_log_score": avg_score}, f)

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, avg_log_scores, "s-", markersize=8, linewidth=2)
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Average Log Score per Test Image")
    plt.title("VAE Test Log Score vs. Training Sample Size")
    plt.xscale("log")
    plt.xticks(sample_sizes, [str(n) for n in sample_sizes])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "vae_log_score_vs_training_size.png")
    plt.savefig(out, dpi=300)
    print(f"\nSaved plot to {out}")


def generate_vae_samples(vae, num_samples=3):
    vae.eval()
    with torch.no_grad():
        latent_dim = vae.fc_mu.out_features
        z = torch.randn(num_samples, latent_dim)
        return vae.decode(z)


def vae_conditional_sample(vae, test_field, num_fixed_pixels, num_samples=2, num_iterations=100):
    """Condition on the first ``num_fixed_pixels`` and resample the rest."""
    vae.eval()
    conditional_samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            cond_sample = torch.randn_like(test_field)
            cond_sample[:num_fixed_pixels] = test_field[:num_fixed_pixels]
            for _ in range(num_iterations):
                mu, logvar = vae.encode(cond_sample)
                z = vae.reparameterize(mu, logvar)
                reconstruction = vae.decode(z)
                cond_sample[num_fixed_pixels:] = reconstruction[num_fixed_pixels:]
            conditional_samples.append(cond_sample)
    return conditional_samples


def run_samples(train_size, epochs, test_index, num_fixed_pixels):
    train_obs, test_obs, train_mean, train_std = load_data()
    print(f"Training final VAE on {train_size} samples for {epochs} epochs...")
    _, vae = train_and_evaluate_vae(train_obs, test_obs, train_size, epochs=epochs)

    vmin, vmax = -1.0, 1.0

    # Unconditional samples
    samples = generate_vae_samples(vae, num_samples=3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, s in zip(axes, samples):
        s_un = s * train_std + train_mean
        ax.imshow(s_un.reshape(64, 64), vmin=vmin, vmax=vmax, origin="upper")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("VAE unconditional samples")
    plt.tight_layout()
    plt.savefig("vae_unconditional_samples.png", dpi=300, bbox_inches="tight")
    print("Saved vae_unconditional_samples.png")

    # Conditional samples
    test_field = test_obs[test_index].clone()
    cond = vae_conditional_sample(vae, test_field, num_fixed_pixels=num_fixed_pixels, num_samples=2)
    fields = [test_field, cond[0], cond[1]]
    titles = ["Test data", "VAE conditional sample 1", "VAE conditional sample 2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, fld, title in zip(axes, fields, titles):
        fld_un = fld * train_std + train_mean
        ax.imshow(fld_un.reshape(64, 64), vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("vae_conditional_samples.png", dpi=300, bbox_inches="tight")
    print("Saved vae_conditional_samples.png")


def main():
    parser = argparse.ArgumentParser(description="VAE emulator for dark-matter density fields")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sweep = sub.add_parser("sweep", help="Log score vs. training size sweep")
    p_sweep.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 30, 50, 100, 160])
    p_sweep.add_argument("--epochs", type=int, default=100)
    p_sweep.add_argument("--save_dir", type=str, default="vae_results")

    p_samp = sub.add_parser("samples", help="Generate unconditional + conditional figures")
    p_samp.add_argument("--train_size", type=int, default=160)
    p_samp.add_argument("--epochs", type=int, default=150)
    p_samp.add_argument("--test_index", type=int, default=23)
    p_samp.add_argument("--num_fixed_pixels", type=int, default=1000)

    args = parser.parse_args()
    if args.command == "sweep":
        run_sweep(args.sizes, args.epochs, args.save_dir)
    elif args.command == "samples":
        run_samples(args.train_size, args.epochs, args.test_index, args.num_fixed_pixels)


if __name__ == "__main__":
    main()
