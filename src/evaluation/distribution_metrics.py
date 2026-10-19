"""Statistical fidelity and distribution metrics for evaluating synthetic tabular data."""
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp


def compute_marginal_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Computes Wasserstein distance and Kolmogorov-Smirnov test statistic per feature.

    Lower scores indicate higher fidelity to the real empirical distribution.
    """
    records = []
    for col in feature_cols:
        r_vals = real_df[col].dropna().to_numpy()
        s_vals = synth_df[col].dropna().to_numpy()

        w_dist = wasserstein_distance(r_vals, s_vals)
        ks_stat, ks_pval = ks_2samp(r_vals, s_vals)

        records.append({
            "Feature": col,
            "Real Mean": float(np.mean(r_vals)),
            "Synth Mean": float(np.mean(s_vals)),
            "Real Std": float(np.std(r_vals)),
            "Synth Std": float(np.std(s_vals)),
            "Wasserstein Distance": float(w_dist),
            "KS Statistic": float(ks_stat),
            "KS p-value": float(ks_pval)
        })

    return pd.DataFrame(records)


def compute_correlation_difference(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: List[str]
) -> float:
    """Computes Frobenius norm distance between feature correlation matrices.

    || Corr(Real) - Corr(Synthetic) ||_F
    Lower value indicates better preservation of multi-variable relationships.
    """
    r_corr = real_df[feature_cols].corr().fillna(0).to_numpy()
    s_corr = synth_df[feature_cols].corr().fillna(0).to_numpy()

    diff = np.linalg.norm(r_corr - s_corr, ord="fro")
    return float(diff)
