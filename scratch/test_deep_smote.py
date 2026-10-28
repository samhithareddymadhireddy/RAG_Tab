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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.data.loader import load_splits
from src.evaluation.metrics import evaluate_model


class DeepSMOTEAutoencoder(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, in_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


class DeepSMOTEResampler:
    """DeepSMOTE: Deep Learning Feature Representation + Latent Space SMOTE.
    
    Reference: Dinsdale et al. (2021). DeepSMOTE: Analyzing and Improving Deep Learning for Imbalanced Data.
    """
    def __init__(
        self,
        latent_dim: int = 16,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 0.002,
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k_neighbors = k_neighbors
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

        # Train Autoencoder on all samples
        ae = DeepSMOTEAutoencoder(n_features, self.latent_dim)
        optimizer = optim.Adam(ae.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        ae.train()
        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                x_rec, _ = ae(batch_x)
                loss = criterion(x_rec, batch_x)
                loss.backward()
                optimizer.step()

        ae.eval()
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

                # Encode minority samples to latent space
                _, z_min = ae(torch.tensor(X_min, dtype=torch.float32))
                z_min_np = z_min.numpy()

                # Generate synthetic latent vectors using SMOTE in latent space
                k = min(self.k_neighbors, max(1, n_min - 1))
                # Create dummy binary labels for latent SMOTE
                z_dummy_y = np.ones(n_min, dtype=int)
                # We need majority reference in latent space or sample pairs
                # Latent interpolation between k-nearest neighbors in latent space:
                from sklearn.neighbors import NearestNeighbors
                nn_model = NearestNeighbors(n_neighbors=min(k + 1, n_min)).fit(z_min_np)
                indices = nn_model.kneighbors(z_min_np, return_distance=False)

                syn_z_list = []
                for _ in range(target_synth):
                    idx = np.random.randint(0, n_min)
                    # Pick random neighbor
                    neighbor_idx = np.random.choice(indices[idx, 1:]) if len(indices[idx]) > 1 else idx
                    lam = np.random.uniform(0.0, 1.0)
                    z_new = z_min_np[idx] + lam * (z_min_np[neighbor_idx] - z_min_np[idx])
                    syn_z_list.append(z_new)

                syn_z_arr = np.array(syn_z_list, dtype=np.float32)
                # Decode synthetic latent vectors back to feature space
                x_syn_scaled = ae.decoder(torch.tensor(syn_z_arr)).numpy()
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

print("Training DeepSMOTE...")
deep_smote = DeepSMOTEResampler(epochs=50, random_state=42)
X_res_ds, y_res_ds = deep_smote.fit_resample(X_train, y_train)
print(f"DeepSMOTE resampled shape: {X_res_ds.shape}, distribution: {pd.Series(y_res_ds).value_counts().to_dict()}")

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_res_ds, y_res_ds)
m_ds = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
print("DeepSMOTE Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_ds.items()})
