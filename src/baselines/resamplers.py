"""Resampling baselines implementation with scaling support and sample provenance tracking."""
from typing import Tuple, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek


def get_resampler(
    method_name: str,
    random_state: int = 42,
    sampling_strategy="auto",
    k_neighbors: int = 5
):
    """Instantiates a resampling algorithm by name."""
    name = method_name.lower().replace("-", "_").replace(" ", "_")
    if name in ("none", "original"):
        return None
    elif name in ("ros", "random_oversampling", "random_oversampler"):
        return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif name == "smote":
        return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    elif name in ("borderline_smote", "borderlinesmote"):
        return BorderlineSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    elif name == "adasyn":
        return ADASYN(sampling_strategy=sampling_strategy, n_neighbors=k_neighbors, random_state=random_state)
    elif name in ("smote_tomek", "smotetomek"):
        return SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: '{method_name}'")


def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "smote",
    scale_features: bool = False,
    n_synthetic: Optional[int] = None,
    random_state: int = 42
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
        'borderline_smote', 'adasyn', 'smote_tomek').
    scale_features : bool, default=False
        If True, standardizes features before computing nearest neighbors,
        then inverts the resampled features back to the original feature space.
    n_synthetic : int, optional
        Fixed budget of synthetic minority samples to generate (equal budget mode).
    random_state : int, default=42
        Seed for reproducibility.

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
        pos_label = "positive" if "positive" in y.values else 1
        neg_label = "negative" if "negative" in y.values else 0
        n_pos = int((y == pos_label).sum())
        n_neg = int((y == neg_label).sum())
        target_pos = min(n_pos + n_synthetic, n_neg)
        sampling_strat = {pos_label: target_pos, neg_label: n_neg}
        k_neighbors = min(5, max(1, n_pos - 1))

    sampler = get_resampler(
        method,
        random_state=random_state,
        sampling_strategy=sampling_strat,
        k_neighbors=k_neighbors
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
        X_res = pd.DataFrame(
            scaler.inverse_transform(X_res),
            columns=feature_cols
        )
    else:
        X_res = pd.DataFrame(X_res, columns=feature_cols)

    y_res = pd.Series(y_res, name=y.name)

    # Track sample provenance
    n_orig = len(X)
    n_total = len(X_res)

    is_ros = "random" in method.lower()
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
