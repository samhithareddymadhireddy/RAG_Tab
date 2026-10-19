"""Local minority-neighborhood retrieval module for RAG-Tab.

Retrieves localized minority-class context around an anchor sample to guide
synthetic generation, analyzing local density and decision-boundary purity.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


@dataclass
class NeighborhoodContext:
    """Encapsulates the retrieved local context for an anchor sample."""
    anchor_idx: Any
    anchor_sample: pd.Series
    neighbor_indices: List[Any]
    neighbors: pd.DataFrame
    distances: np.ndarray
    k: int
    mean_distance: float
    local_density: float
    boundary_purity: float
    feature_ranges: Dict[str, Dict[str, float]]
    retrieval_method: str = "knn"

    def to_dict(self) -> Dict[str, Any]:
        """Converts context into a serializable dictionary for prompt construction."""
        return {
            "anchor_index": self.anchor_idx,
            "anchor": self.anchor_sample.to_dict(),
            "k": self.k,
            "mean_distance": round(float(self.mean_distance), 4),
            "local_density": round(float(self.local_density), 4),
            "boundary_purity": round(float(self.boundary_purity), 4),
            "retrieval_method": self.retrieval_method,
            "neighbors": self.neighbors.to_dict(orient="records"),
            "feature_ranges": self.feature_ranges
        }


class MinorityNeighborhoodRetriever:
    """Manages distance-based neighborhood retrieval for minority class samples."""

    def __init__(self, metric: str = "euclidean", scale_features: bool = True):
        self.metric = metric
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: List[str] = []
        self.minority_class: str = "positive"
        self.majority_class: str = "negative"

        self.X_train_orig: Optional[pd.DataFrame] = None
        self.y_train_orig: Optional[pd.Series] = None
        self.X_min_orig: Optional[pd.DataFrame] = None
        self.X_min_scaled: Optional[np.ndarray] = None
        self.min_indices: Optional[pd.Index] = None

        self.nn_minority: Optional[NearestNeighbors] = None
        self.nn_global: Optional[NearestNeighbors] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        minority_class: str = "positive",
        majority_class: str = "negative"
    ) -> "MinorityNeighborhoodRetriever":
        """Fits the scaler and nearest-neighbor indices using training data only."""
        self.feature_cols = list(X_train.columns)
        self.minority_class = minority_class
        self.majority_class = majority_class
        self.X_train_orig = X_train.copy()
        self.y_train_orig = y_train.copy()

        if self.scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None
            X_train_scaled = X_train.to_numpy()

        # Isolate minority training samples
        min_mask = (y_train == minority_class).to_numpy()
        self.min_indices = X_train.index[min_mask]
        self.X_min_orig = X_train.loc[self.min_indices].copy()
        self.X_min_scaled = X_train_scaled[min_mask]

        # Fit nearest neighbors on minority samples (max K up to minority size)
        n_min = len(self.X_min_scaled)
        max_k_min = min(30, n_min)
        self.nn_minority = NearestNeighbors(
            n_neighbors=max_k_min,
            metric=self.metric
        ).fit(self.X_min_scaled)

        # Fit nearest neighbors on ALL training samples (to assess boundary proximity)
        max_k_global = min(30, len(X_train_scaled))
        self.nn_global = NearestNeighbors(
            n_neighbors=max_k_global,
            metric=self.metric
        ).fit(X_train_scaled)

        return self

    def retrieve_by_index(
        self,
        anchor_idx: Any,
        k: int = 5
    ) -> NeighborhoodContext:
        """Retrieves K nearest minority neighbors for an anchor specified by index.

        Parameters
        ----------
        anchor_idx : Any
            DataFrame index of the minority anchor row.
        k : int
            Number of nearest neighbors to retrieve.
        """
        if self.nn_minority is None:
            raise RuntimeError("Retriever is not fitted. Call .fit() first.")

        if anchor_idx not in self.min_indices:
            raise ValueError(f"Index {anchor_idx} is not in the minority training set.")

        pos_in_min = list(self.min_indices).index(anchor_idx)
        anchor_vec_scaled = self.X_min_scaled[pos_in_min].reshape(1, -1)
        anchor_sample = self.X_min_orig.loc[anchor_idx]

        # Query minority neighborhood: search k + 1 because the anchor itself will be at distance 0
        distances_min, indices_min = self.nn_minority.kneighbors(
            anchor_vec_scaled,
            n_neighbors=k + 1
        )
        distances_min = distances_min[0]
        indices_min = indices_min[0]

        # Filter out self-match
        valid_pairs = [
            (dist, idx) for dist, idx in zip(distances_min, indices_min)
            if idx != pos_in_min
        ][:k]

        neighbor_dists = np.array([p[0] for p in valid_pairs])
        neighbor_pos = [p[1] for p in valid_pairs]
        neighbor_df_indices = [self.min_indices[p] for p in neighbor_pos]
        neighbors_df = self.X_min_orig.loc[neighbor_df_indices].copy()

        # Local density metric: inverse of average Euclidean distance to neighbors
        mean_dist = float(np.mean(neighbor_dists)) if len(neighbor_dists) > 0 else 0.0
        local_density = float(1.0 / (mean_dist + 1e-6))

        # Boundary analysis: query global training neighbors
        # Purity = fraction of top-K global neighbors that also belong to minority class
        _, global_indices = self.nn_global.kneighbors(
            anchor_vec_scaled,
            n_neighbors=k + 1
        )
        global_indices = global_indices[0]
        global_labels = self.y_train_orig.iloc[global_indices].values
        # Exclude anchor itself
        global_neighbor_labels = global_labels[1:k + 1]
        boundary_purity = float(
            np.mean(global_neighbor_labels == self.minority_class)
        )

        # Feature range bounding box over local neighborhood (anchor + neighbors)
        combined = pd.concat([anchor_sample.to_frame().T, neighbors_df])
        feature_ranges = {}
        for col in self.feature_cols:
            c_min = float(combined[col].min())
            c_max = float(combined[col].max())
            c_mean = float(combined[col].mean())
            c_std = float(combined[col].std()) if len(combined) > 1 else 0.0
            feature_ranges[col] = {
                "min": c_min,
                "max": c_max,
                "mean": c_mean,
                "std": c_std
            }

        return NeighborhoodContext(
            anchor_idx=anchor_idx,
            anchor_sample=anchor_sample,
            neighbor_indices=neighbor_df_indices,
            neighbors=neighbors_df,
            distances=neighbor_dists,
            k=k,
            mean_distance=mean_dist,
            local_density=local_density,
            boundary_purity=boundary_purity,
            feature_ranges=feature_ranges,
            retrieval_method="knn"
        )

    def retrieve_natural_neighbors(
        self,
        anchor_idx: Any,
        search_radius_k: int = 15
    ) -> NeighborhoodContext:
        """Retrieves natural neighbors (mutual nearest neighbors) for an anchor.

        Two points x and y are natural neighbors if x is in KNN(y) AND y is in KNN(x).
        This adaptively discovers neighbors according to local density without
        forcing a fixed arbitrary K.
        """
        if anchor_idx not in self.min_indices:
            raise ValueError(f"Index {anchor_idx} is not in minority set.")

        pos_in_min = list(self.min_indices).index(anchor_idx)
        anchor_sample = self.X_min_orig.loc[anchor_idx]

        # Retrieve nearest candidates within search_radius_k
        k_search = min(search_radius_k + 1, len(self.X_min_scaled))
        dists, indices = self.nn_minority.kneighbors(
            self.X_min_scaled,
            n_neighbors=k_search
        )

        # Forward neighbors of anchor (excluding self)
        forward_cand_pos = [i for i in indices[pos_in_min] if i != pos_in_min]
        forward_cand_dists = [d for d, i in zip(dists[pos_in_min], indices[pos_in_min]) if i != pos_in_min]

        # Check reciprocity: does candidate 'cand' include 'pos_in_min' in its forward neighbors?
        natural_pos = []
        natural_dists = []
        for cand, dist in zip(forward_cand_pos, forward_cand_dists):
            cand_neighbors = indices[cand]
            if pos_in_min in cand_neighbors:
                natural_pos.append(cand)
                natural_dists.append(dist)

        # Fallback to standard 3-NN if no mutual neighbors found in very sparse region
        if len(natural_pos) == 0:
            return self.retrieve_by_index(anchor_idx, k=3)

        neighbor_df_indices = [self.min_indices[p] for p in natural_pos]
        neighbors_df = self.X_min_orig.loc[neighbor_df_indices].copy()
        neighbor_dists_arr = np.array(natural_dists)

        mean_dist = float(np.mean(neighbor_dists_arr))
        local_density = float(1.0 / (mean_dist + 1e-6))

        # Boundary purity
        anchor_vec_scaled = self.X_min_scaled[pos_in_min].reshape(1, -1)
        _, global_indices = self.nn_global.kneighbors(
            anchor_vec_scaled,
            n_neighbors=len(natural_pos) + 1
        )
        global_labels = self.y_train_orig.iloc[global_indices[0]].values
        boundary_purity = float(np.mean(global_labels[1:] == self.minority_class))

        combined = pd.concat([anchor_sample.to_frame().T, neighbors_df])
        feature_ranges = {}
        for col in self.feature_cols:
            feature_ranges[col] = {
                "min": float(combined[col].min()),
                "max": float(combined[col].max()),
                "mean": float(combined[col].mean()),
                "std": float(combined[col].std()) if len(combined) > 1 else 0.0
            }

        return NeighborhoodContext(
            anchor_idx=anchor_idx,
            anchor_sample=anchor_sample,
            neighbor_indices=neighbor_df_indices,
            neighbors=neighbors_df,
            distances=neighbor_dists_arr,
            k=len(natural_pos),
            mean_distance=mean_dist,
            local_density=local_density,
            boundary_purity=boundary_purity,
            feature_ranges=feature_ranges,
            retrieval_method="natural_neighbors"
        )
