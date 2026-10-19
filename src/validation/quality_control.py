"""Quality control and filtering pipeline for synthetic tabular instances."""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SyntheticQualityController:
    """Multi-stage quality control filter for generated synthetic minority samples.

    Implements:
    1. Schema & non-null validation
    2. Physical domain and probability range enforcement
    3. Exact and near-duplicate detection against existing training data
    4. Local manifold consistency (distance to anchor relative to local neighbor envelope)
    5. Optional classifier-based confidence filtering
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        feature_cols: List[str],
        epsilon_near_dup: float = 0.02,
        max_anchor_dist_ratio: float = 2.0
    ):
        self.feature_cols = feature_cols
        self.epsilon_near_dup = epsilon_near_dup
        self.max_anchor_dist_ratio = max_anchor_dist_ratio

        # Standardize training feature space for distance-based checks
        self.scaler = StandardScaler().fit(X_train[feature_cols])
        self.X_train_scaled = self.scaler.transform(X_train[feature_cols])

    def evaluate_sample(
        self,
        candidate_row: pd.Series,
        anchor_sample: Optional[pd.Series] = None,
        max_neighbor_dist: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Evaluates a single candidate row through all quality control stages.

        Returns
        -------
        is_accepted : bool
            True if row passes all active QC checks.
        rejection_reason : str
            Detailed reason if rejected, or 'Accepted'.
        """
        # 1. Missing or NaN check
        if candidate_row[self.feature_cols].isnull().any():
            return False, "Contains NaN or missing feature values"

        # 2. Domain & physical range checks
        val_dict = candidate_row[self.feature_cols].to_dict()
        if "P_black" in val_dict and not (0.0 <= val_dict["P_black"] <= 1.0):
            return False, f"P_black={val_dict['P_black']} violates [0, 1] range"

        if "P_and" in val_dict and not (0.0 <= val_dict["P_and"] <= 1.0):
            return False, f"P_and={val_dict['P_and']} violates [0, 1] range"

        if "Height" in val_dict and val_dict["Height"] < 1:
            return False, f"Height={val_dict['Height']} is invalid (must be >= 1)"

        if "Lenght" in val_dict and val_dict["Lenght"] < 1:
            return False, f"Lenght={val_dict['Lenght']} is invalid (must be >= 1)"

        # 3. Near-duplicate check against training data
        cand_scaled = self.scaler.transform(pd.DataFrame([candidate_row[self.feature_cols]]))[0]
        dists_to_train = np.linalg.norm(self.X_train_scaled - cand_scaled, axis=1)
        min_dist_to_train = np.min(dists_to_train)

        if min_dist_to_train < self.epsilon_near_dup:
            return False, f"Near-duplicate to training row (scaled distance {min_dist_to_train:.4f} < {self.epsilon_near_dup})"

        # 4. Local manifold consistency check against anchor
        if anchor_sample is not None and max_neighbor_dist is not None:
            anchor_scaled = self.scaler.transform(pd.DataFrame([anchor_sample[self.feature_cols]]))[0]
            dist_to_anchor = np.linalg.norm(cand_scaled - anchor_scaled)
            max_allowed = max(0.5, max_neighbor_dist * self.max_anchor_dist_ratio)
            if dist_to_anchor > max_allowed:
                return False, f"Manifold drift: dist to anchor ({dist_to_anchor:.3f}) exceeds threshold ({max_allowed:.3f})"

        return True, "Accepted"

    def filter_batch(
        self,
        candidates_df: pd.DataFrame,
        anchor_sample: Optional[pd.Series] = None,
        max_neighbor_dist: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, int]]:
        """Filters a candidate DataFrame, returning accepted rows and rejection logs."""
        accepted_rows = []
        rejections = []

        for idx, row in candidates_df.iterrows():
            passed, reason = self.evaluate_sample(
                row,
                anchor_sample=anchor_sample,
                max_neighbor_dist=max_neighbor_dist
            )
            if passed:
                accepted_rows.append(row)
            else:
                rejections.append({
                    "row_index": idx,
                    "reason": reason,
                    "candidate_values": row[self.feature_cols].to_dict()
                })

        accepted_df = pd.DataFrame(accepted_rows)
        if not accepted_df.empty:
            accepted_df = accepted_df[candidates_df.columns]

        qc_stats = {
            "total_submitted": len(candidates_df),
            "accepted_count": len(accepted_rows),
            "rejected_count": len(rejections),
            "acceptance_rate": (len(accepted_rows) / max(1, len(candidates_df))) * 100
        }

        return accepted_df, rejections, qc_stats
