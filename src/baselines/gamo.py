"""Generative Adversarial Minority Oversampling (GAMO) for Tabular Imbalanced Learning.

Reference:
    Mullick, S. S., Datta, S., & Das, S. (2019).
    Generative Adversarial Minority Oversampling.
    IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class ConvexGenerator(nn.Module):
    """Generates synthetic minority samples strictly as convex combinations of real minority instances.
    
    Given latent noise z, computes softmax simplex weights w such that:
    x_syn = sum_i (w_i * x_i), where w_i >= 0 and sum(w_i) = 1.
    """
    def __init__(self, latent_dim: int, n_minority_samples: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, n_minority_samples),
            nn.Softmax(dim=-1)
        )

    def forward(self, z: torch.Tensor, X_minority: torch.Tensor) -> torch.Tensor:
        weights = self.net(z)
        return torch.matmul(weights, X_minority)


class Discriminator(nn.Module):
    """Adversarial discriminator enforcing true distribution conformity."""
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierAuxNet(nn.Module):
    """Auxiliary multi-class classifier guiding generator towards decision boundaries."""
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAMOResampler:
    """Three-player Generative Adversarial Minority Oversampling (GAMO) framework.
    
    Implements a 3-player competitive game between:
    1. Convex Generator G (synthesizes boundary points via convex combination).
    2. Real/Fake Discriminator D (enforces realistic feature distributions).
    3. Multi-class Classifier C (steers synthetic points toward decision margins).
    
    Compatible with Scikit-Learn and Imbalanced-Learn resampler protocols.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 0.002,
        lambda_boundary: float = 0.5,
        random_state: int = 42
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_boundary = lambda_boundary
        self.random_state = random_state

    def fit_resample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_synthetic: Optional[int] = None
    ) -> Tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
        """Fits the GAMO adversarial network and returns resampled dataset."""
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

        # Standardize features for stable GAN optimization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        label_map = {c: i for i, c in enumerate(classes)}
        y_int = np.array([label_map[c] for c in y_arr])
        n_features = X_scaled.shape[1]
        n_classes = len(classes)

        all_synthetic_X = []
        all_synthetic_y = []

        for min_c in minority_classes:
            min_mask = (y_arr == min_c)
            X_min = X_scaled[min_mask]
            n_min = len(X_min)
            if n_min == 0:
                continue

            target_class_idx = label_map[min_c]

            # Determine synthetic budget
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

            X_min_tensor = torch.tensor(X_min, dtype=torch.float32)
            X_all_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_all_tensor = torch.tensor(y_int, dtype=torch.long)

            generator = ConvexGenerator(self.latent_dim, n_min)
            discriminator = Discriminator(n_features)
            classifier = ClassifierAuxNet(n_features, n_classes)

            opt_g = optim.Adam(generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            opt_d = optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            opt_c = optim.Adam(classifier.parameters(), lr=self.lr)

            bce_loss = nn.BCELoss()
            ce_loss = nn.CrossEntropyLoss()

            # Train GAMO 3-player game
            for epoch in range(self.epochs):
                # 1. Train Classifier on existing data
                opt_c.zero_grad()
                pred_all = classifier(X_all_tensor)
                loss_c = ce_loss(pred_all, y_all_tensor)
                loss_c.backward()
                opt_c.step()

                # 2. Train Discriminator
                opt_d.zero_grad()
                batch_sz = min(self.batch_size, n_min)
                idx_real = np.random.choice(n_min, batch_sz, replace=(batch_sz > n_min))
                x_real = X_min_tensor[idx_real]

                z = torch.randn(batch_sz, self.latent_dim)
                x_fake = generator(z, X_min_tensor)

                d_real = discriminator(x_real)
                d_fake = discriminator(x_fake.detach())

                loss_d = bce_loss(d_real, torch.ones_like(d_real)) + bce_loss(d_fake, torch.zeros_like(d_fake))
                loss_d.backward()
                opt_d.step()

                # 3. Train Generator (fool D and reach classification boundary)
                opt_g.zero_grad()
                z = torch.randn(batch_sz, self.latent_dim)
                x_fake = generator(z, X_min_tensor)

                d_fake_g = discriminator(x_fake)
                adv_loss = bce_loss(d_fake_g, torch.ones_like(d_fake_g))

                # Boundary guidance loss
                clf_logits = classifier(x_fake)
                boundary_loss = ce_loss(clf_logits, torch.full((batch_sz,), target_class_idx, dtype=torch.long))

                loss_g = adv_loss + self.lambda_boundary * boundary_loss
                loss_g.backward()
                opt_g.step()

            # Generate synthetic minority samples
            generator.eval()
            with torch.no_grad():
                z_syn = torch.randn(target_synth, self.latent_dim)
                x_syn = generator(z_syn, X_min_tensor).numpy()

            all_synthetic_X.append(x_syn)
            all_synthetic_y.extend([min_c] * target_synth)

        if all_synthetic_X:
            syn_X_concat = np.vstack(all_synthetic_X)
            # Invert standardization to restore original feature scaling
            syn_X_orig = scaler.inverse_transform(syn_X_concat)
            syn_y_arr = np.array(all_synthetic_y)

            X_res = np.vstack([X_arr, syn_X_orig])
            y_res = np.concatenate([y_arr, syn_y_arr])
        else:
            X_res = X_arr.copy()
            y_res = y_arr.copy()

        if is_df:
            X_res = pd.DataFrame(X_res, columns=cols)
            y_res = pd.Series(y_res, name=y.name if hasattr(y, 'name') else 'target')

        return X_res, y_res
