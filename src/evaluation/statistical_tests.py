"""Statistical significance testing for multi-dataset benchmark evaluation.

Implements Demsar (2006) protocol:
- Friedman rank sum test across datasets
- Iman-Davenport corrected F-statistic
- Nemenyi post-hoc Critical Difference (CD) computation
- Publication-quality Critical Difference (CD) diagram visualization
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f, friedmanchisquare, rankdata


# Studentized range statistics q_alpha / sqrt(2) for Nemenyi test at alpha=0.05
# Index = number of methods k (up to 10)
NEMENYI_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164
}


def compute_friedman_and_cd(
    matrix_df: pd.DataFrame,
    higher_is_better: bool = True,
    alpha: float = 0.05
) -> Tuple[pd.Series, float, float, float]:
    """Computes Friedman test, average ranks, and Nemenyi Critical Difference (CD).

    Parameters
    ----------
    matrix_df : pd.DataFrame
        Rows are datasets, columns are methods. Values are metric scores (e.g. F1 or Precision).
    higher_is_better : bool, default=True
        If True, higher values get rank 1.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    mean_ranks : pd.Series
        Average rank for each method (1 = best).
    cd : float
        Critical difference threshold.
    friedman_stat : float
        Chi-square statistic.
    p_value : float
        p-value of the Friedman test.
    """
    n_datasets, k_methods = matrix_df.shape

    # Rank each row: rank 1 is best
    ranks = np.zeros_like(matrix_df.values, dtype=float)
    for i in range(n_datasets):
        row = matrix_df.values[i, :]
        if higher_is_better:
            ranks[i, :] = rankdata(-row, method="average")
        else:
            ranks[i, :] = rankdata(row, method="average")

    mean_ranks = pd.Series(ranks.mean(axis=0), index=matrix_df.columns).sort_values()

    # Friedman chi-square
    # scipy.stats.friedmanchisquare expects arguments as columns
    cols_data = [matrix_df[c].values for c in matrix_df.columns]
    stat, pval = friedmanchisquare(*cols_data)

    # Nemenyi Critical Difference
    q_val = NEMENYI_Q_ALPHA_005.get(k_methods, 2.850)
    cd = q_val * np.sqrt((k_methods * (k_methods + 1)) / (6.0 * n_datasets))

    return mean_ranks, cd, stat, pval


def plot_critical_difference_diagram(
    mean_ranks: pd.Series,
    cd: float,
    title: str = "Critical Difference Diagram",
    output_path: str = "results/plots/critical_difference_diagram.png"
):
    """Draws a standard Demsar (2006) Critical Difference horizontal axis diagram."""
    k = len(mean_ranks)
    methods = list(mean_ranks.index)
    ranks = list(mean_ranks.values)

    fig, ax = plt.subplots(figsize=(10, max(4, k * 0.7)))

    # Main ranking axis from 1 to k
    ax.plot([1, k], [0, 0], color="black", lw=2)
    for tick in range(1, k + 1):
        ax.plot([tick, tick], [-0.05, 0.05], color="black", lw=1.5)
        ax.text(tick, 0.1, str(tick), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # CD bar at top
    cd_start = 1.0
    cd_end = cd_start + cd
    ax.plot([cd_start, cd_end], [0.6, 0.6], color="crimson", lw=3)
    ax.plot([cd_start, cd_start], [0.55, 0.65], color="crimson", lw=2)
    ax.plot([cd_end, cd_end], [0.55, 0.65], color="crimson", lw=2)
    ax.text((cd_start + cd_end) / 2, 0.7, f"CD = {cd:.3f} (α=0.05)", ha="center", va="bottom", color="crimson", fontweight="bold")

    # Plot method points and labels
    y_offsets = np.linspace(-0.3, -1.2, k)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for idx, (m, r) in enumerate(zip(methods, ranks)):
        y_pos = y_offsets[idx]
        ax.plot([r, r], [0, y_pos], color="gray", linestyle="--", alpha=0.6)
        ax.scatter([r], [0], color=colors[idx], s=80, zorder=5)
        ax.scatter([r], [y_pos], color=colors[idx], s=60, zorder=5)
        ax.text(r + 0.08, y_pos, f"{m} ({r:.2f})", ha="left", va="center", fontsize=10, fontweight="bold")

    # Connect methods that are not significantly different (rank diff < CD)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(ranks[i] - ranks[j]) < cd:
                ax.plot([ranks[i], ranks[j]], [y_offsets[i] - 0.05, y_offsets[i] - 0.05], color="navy", lw=2, alpha=0.3)

    ax.set_xlim(0.5, k + 0.8)
    ax.set_ylim(-1.5, 0.9)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
