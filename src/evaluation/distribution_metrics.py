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


def compute_distance_to_closest_record(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, float]:
    """Computes Distance to Closest Record (DCR) for privacy and non-memorization audit.

    For each synthetic row, calculates the minimum Euclidean distance to all real training rows
    in standardized space.
    - DCR == 0: Exact memorization (leakage/privacy violation)
    - DCR > 0: Novel synthetic generation
    """
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import StandardScaler

    r_mat = real_df[feature_cols].dropna().to_numpy()
    s_mat = synth_df[feature_cols].dropna().to_numpy()

    if len(s_mat) == 0 or len(r_mat) == 0:
        return {
            "DCR_Mean": 0.0,
            "DCR_Median": 0.0,
            "DCR_Min": 0.0,
            "DCR_5th_Percentile": 0.0
        }

    scaler = StandardScaler()
    r_scaled = scaler.fit_transform(r_mat)
    s_scaled = scaler.transform(s_mat)

    dists = cdist(s_scaled, r_scaled, metric="euclidean")
    min_dists = np.min(dists, axis=1)

    return {
        "DCR_Mean": float(np.mean(min_dists)),
        "DCR_Median": float(np.median(min_dists)),
        "DCR_Min": float(np.min(min_dists)),
        "DCR_5th_Percentile": float(np.percentile(min_dists, 5))
    }


def evaluate_distribution_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, float]:
    """Computes comprehensive distribution fidelity and privacy metrics.

    Returns:
    - Mean_Wasserstein: Average 1D Wasserstein distance across feature marginals
    - Mean_KS_Statistic: Average Kolmogorov-Smirnov two-sample test statistic
    - Correlation_Frobenius_Diff: Frobenius norm distance between feature correlation matrices
    - DCR_Mean, DCR_Median, DCR_Min, DCR_5th_Percentile: Distance to closest record
    """
    marginals = compute_marginal_fidelity(real_df, synth_df, feature_cols)
    mean_w1 = float(marginals["Wasserstein Distance"].mean()) if len(marginals) > 0 else 0.0
    mean_ks = float(marginals["KS Statistic"].mean()) if len(marginals) > 0 else 0.0
    corr_diff = compute_correlation_difference(real_df, synth_df, feature_cols)
    dcr_stats = compute_distance_to_closest_record(real_df, synth_df, feature_cols)

    res = {
        "Mean_Wasserstein": mean_w1,
        "Mean_KS_Statistic": mean_ks,
        "Correlation_Frobenius_Diff": corr_diff,
        **dcr_stats
    }
    return res

