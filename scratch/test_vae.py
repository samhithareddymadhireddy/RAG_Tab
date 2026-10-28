import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data.loader import load_splits
from src.evaluation.metrics import evaluate_model


class TabularVAE(nn.Module):
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

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar


class TabularVAEResampler:
    """Variational Autoencoder (VAE) for Deep Tabular Minority Oversampling."""
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

                target_synth = n_maj - n_min if n_synthetic is None else n_synthetic
                if target_synth <= 0:
                    continue

                mu_min, logvar_min = vae.encode(torch.tensor(X_min, dtype=torch.float32))
                mu_mean = mu_min.mean(dim=0)
                std_mean = torch.exp(0.5 * logvar_min).mean(dim=0)

                # Sample synthetic points around minority latent distribution
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


# Test on page-blocks0
X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
    dataset_name="page-blocks0",
    splits_dir=PROJECT_ROOT / "data/splits"
)
minority_class = "positive"
majority_class = "negative"

print("Training Tabular VAE...")
vae_resampler = TabularVAEResampler(epochs=50, random_state=42)
X_res_vae, y_res_vae = vae_resampler.fit_resample(X_train, y_train)
print(f"Tabular VAE resampled shape: {X_res_vae.shape}, distribution: {pd.Series(y_res_vae).value_counts().to_dict()}")

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_res_vae, y_res_vae)
m_vae = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
print("Tabular VAE Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_vae.items()})
