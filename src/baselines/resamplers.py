"""Resampling baselines implementation with scaling support and sample provenance tracking."""
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, SMOTENC, SMOTEN
)
from imblearn.combine import SMOTETomek, SMOTEENN
from src.baselines.gamo import GAMOResampler
from src.baselines.deep_smote import DeepSMOTEResampler
from src.baselines.tabular_vae import TabularVAEResampler


def get_resampler(
    method_name: str,
    random_state: int = 42,
    sampling_strategy="auto",
    k_neighbors: int = 5,
    categorical_features: Optional[List[int]] = None
):
    """Instantiates a resampling algorithm by name."""
    name = method_name.lower().replace("-", "_").replace(" ", "_")
    if name in ("none", "original"):
        return None
    elif name in ("ros", "random_oversampling", "random_oversampler"):
        return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif name == "smote":
        return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    elif name in ("borderline_smote", "borderlinesmote", "bsmote", "borderline_1", "bsmote_1"):
        return BorderlineSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, kind="borderline-1", random_state=random_state)
    elif name in ("borderline_2", "bsmote_2"):
        return BorderlineSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, kind="borderline-2", random_state=random_state)
    elif name == "adasyn":
        return ADASYN(sampling_strategy=sampling_strategy, n_neighbors=k_neighbors, random_state=random_state)
    elif name in ("kmeans_smote", "kmeanssmote"):
        return KMeansSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, cluster_balance_threshold=0.01, random_state=random_state)
    elif name in ("smote_enn", "smoteenn"):
        return SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif name in ("smote_tomek", "smotetomek"):
        return SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    elif name in ("smotenc", "smote_nc"):
        cats = categorical_features if categorical_features is not None else []
        return SMOTENC(categorical_features=cats, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    elif name in ("smoten", "smote_n"):
        return SMOTEN(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    elif name in ("gamo", "gamo_resampler"):
        return GAMOResampler(random_state=random_state)
    elif name in ("deep_smote", "deepsmote"):
        return DeepSMOTEResampler(random_state=random_state)
    elif name in ("tabular_vae", "vae", "vae_resampler"):
        return TabularVAEResampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: '{method_name}'")


def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "smote",
    scale_features: bool = False,
    n_synthetic: Optional[int] = None,
    random_state: int = 42,
    categorical_features: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Applies a resampling method with optional feature scaling and provenance tracking.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training target vector.
    method : str
        Resampling method name ('original', 'random_oversampling', 'smote',
        'borderline_smote', 'bsmote_2', 'adasyn', 'kmeans_smote', 'smote_enn',
        'smote_tomek', 'smotenc', 'smoten', 'gamo', 'deep_smote', 'tabular_vae').
    scale_features : bool, default=False
        If True, standardizes features before computing nearest neighbors,
        then inverts the resampled features back to the original feature space.
    n_synthetic : int, optional
        Fixed budget of synthetic minority samples to generate (equal budget mode).
    random_state : int, default=42
        Seed for reproducibility.
    categorical_features : list of int, optional
        Column indices of categorical variables (for SMOTENC).

    Returns
    -------
    X_resampled : pd.DataFrame
        Resampled feature matrix in original feature units.
    y_resampled : pd.Series
        Resampled target vector.
    provenance : pd.DataFrame
        Table with 'sample_id', 'origin' ('original', 'duplicate', 'synthetic'),
        and class label for every row.
    """
    sampling_strat = "auto"
    k_neighbors = 5
    if n_synthetic is not None:
        class_counts = {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}
        maj_class = max(class_counts, key=class_counts.get)
        n_maj = class_counts[maj_class]
        minority_classes = [c for c in class_counts if c != maj_class]

        if len(class_counts) <= 2:
            pos_label = minority_classes[0] if minority_classes else maj_class
            n_pos = class_counts.get(pos_label, 0)
            target_pos = min(n_pos + n_synthetic, n_maj)
            sampling_strat = {pos_label: target_pos, maj_class: n_maj}
            k_neighbors = min(5, max(1, n_pos - 1))
        else:
            deficits = {c: n_maj - class_counts[c] for c in minority_classes}
            total_deficit = sum(deficits.values())
            sampling_strat = {maj_class: n_maj}
            min_k = 5
            for c in minority_classes:
                if total_deficit > 0:
                    alloc = max(1, int(round(n_synthetic * (deficits[c] / total_deficit))))
                else:
                    alloc = max(1, n_synthetic // len(minority_classes))
                sampling_strat[c] = min(class_counts[c] + alloc, n_maj)
                min_k = min(min_k, max(1, class_counts[c] - 1))
            k_neighbors = min_k

    method_lower = method.lower().replace("-", "_").replace(" ", "_")

    # Deep learning generative methods
    if "gamo" in method_lower:
        sampler = GAMOResampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y, n_synthetic=n_synthetic)
    elif "deep_smote" in method_lower or "deepsmote" in method_lower:
        sampler = DeepSMOTEResampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y, n_synthetic=n_synthetic)
    elif "vae" in method_lower:
        sampler = TabularVAEResampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y, n_synthetic=n_synthetic)
    else:
        sampler = get_resampler(
            method,
            random_state=random_state,
            sampling_strategy=sampling_strat,
            k_neighbors=k_neighbors,
            categorical_features=categorical_features
        )
        if sampler is None:
            prov = pd.DataFrame({
                "sample_id": range(len(X)),
                "origin": "original",
                "Class": y.values
            })
            return X.copy(), y.copy(), prov

        feature_cols = list(X.columns)
        if scale_features:
            scaler = StandardScaler()
            X_in = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
        else:
            scaler = None
            X_in = X.copy()

        X_res, y_res = sampler.fit_resample(X_in, y)

        if scaler is not None:
            X_res = pd.DataFrame(scaler.inverse_transform(X_res), columns=feature_cols)
        else:
            X_res = pd.DataFrame(X_res, columns=feature_cols)

        y_res = pd.Series(y_res, name=y.name)

    n_orig = len(X)
    n_total = len(X_res)
    is_ros = "random" in method_lower

    origins = []
    for i in range(n_total):
        if i < n_orig:
            origins.append("original")
        else:
            origins.append("duplicate" if is_ros else "synthetic")

    prov = pd.DataFrame({
        "sample_id": range(n_total),
        "origin": origins,
        "Class": y_res.values
    })

    return X_res, y_res, prov
