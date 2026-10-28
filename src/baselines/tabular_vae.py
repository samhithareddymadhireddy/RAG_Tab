"""Tabular VAE: Variational Autoencoder Oversampling for Imbalanced Tabular Data.

Reference:
    Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
    Applied to tabular class imbalance by generative manifold sampling.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class TabularVAE(nn.Module):
    """Variational Autoencoder architecture for continuous tabular features."""
    def __init__(self, in_features: int, latent_dim: int = 16):
        super().__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, in_features)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar


class TabularVAEResampler:
    """Tabular VAE generative oversampler.
    
    1. Trains a Variational Autoencoder with ELBO objective (MSE + KL divergence).
    2. Learns the posterior latent distribution for minority classes.
    3. Synthesizes new minority instances by sampling latent embeddings and decoding.
    """
    def __init__(
        self,
        latent_dim: int = 16,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 0.002,
        beta_kl: float = 0.01,
        random_state: int = 42
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta_kl = beta_kl
        self.random_state = random_state

    def fit_resample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_synthetic: Optional[int] = None
    ) -> Tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        is_df = isinstance(X, pd.DataFrame)
        cols = X.columns if is_df else None
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=str)

        classes, counts = np.unique(y_arr, return_counts=True)
        class_counts = dict(zip(classes, counts))
        maj_class = max(class_counts, key=class_counts.get)
        n_maj = class_counts[maj_class]
        minority_classes = [c for c in classes if c != maj_class]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        n_samples, n_features = X_scaled.shape

        vae = TabularVAE(n_features, self.latent_dim)
        optimizer = optim.Adam(vae.parameters(), lr=self.lr, weight_decay=1e-5)

        dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        vae.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                x_rec, mu, logvar = vae(batch_x)
                recon_loss = nn.functional.mse_loss(x_rec, batch_x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)
                loss = recon_loss + self.beta_kl * kl_loss
                loss.backward()
                optimizer.step()

        vae.eval()
        all_syn_X = []
        all_syn_y = []

        with torch.no_grad():
            for min_c in minority_classes:
                min_mask = (y_arr == min_c)
                X_min = X_scaled[min_mask]
                n_min = len(X_min)
                if n_min == 0:
                    continue

                if n_synthetic is not None:
                    if len(minority_classes) == 1:
                        target_synth = n_synthetic
                    else:
                        deficits = {c: n_maj - class_counts[c] for c in minority_classes}
                        tot_def = sum(deficits.values())
                        if tot_def > 0:
                            target_synth = max(1, int(round(n_synthetic * (deficits[min_c] / tot_def))))
                        else:
                            target_synth = max(1, n_synthetic // len(minority_classes))
                else:
                    target_synth = n_maj - n_min

                if target_synth <= 0:
                    continue

                mu_min, logvar_min = vae.encode(torch.tensor(X_min, dtype=torch.float32))
                mu_mean = mu_min.mean(dim=0)
                std_mean = torch.exp(0.5 * logvar_min).mean(dim=0)

                eps = torch.randn(target_synth, self.latent_dim)
                z_syn = mu_mean + eps * std_mean

                x_syn_scaled = vae.decoder(z_syn).numpy()
                x_syn_orig = scaler.inverse_transform(x_syn_scaled)

                all_syn_X.append(x_syn_orig)
                all_syn_y.extend([min_c] * target_synth)

        if all_syn_X:
            X_res = np.vstack([X_arr, np.vstack(all_syn_X)])
            y_res = np.concatenate([y_arr, np.array(all_syn_y)])
        else:
            X_res = X_arr.copy()
            y_res = y_arr.copy()

        if is_df:
            X_res = pd.DataFrame(X_res, columns=cols)
            y_res = pd.Series(y_res, name=y.name if hasattr(y, 'name') else 'target')

        return X_res, y_res
