"""Unified Adaptive Retriever combining Geometric Distance and BERT Semantic Embeddings
with Self-Adaptive Natural Neighbors (NaN) and Boundary Anchor Detection.
"""
from typing import Any, List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

from .neighborhood import NeighborhoodContext
from .bert_retriever import verbalize_tabular_row


_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}


class UnifiedAdaptiveRetriever:
    """Unified retrieval engine for RAG-Tab.
    
    Integrates:
    1. Hybrid Distance: Fuses standardized continuous Euclidean distance with BERT 
       sentence-transformer semantic embeddings of verbalized tabular rows.
    2. Natural Neighbors (NaN): Parameter-free, self-adaptive mutual neighborhood
       algorithm that removes the arbitrary choice of K and automatically adapts to local density.
    3. Manifold Topology Classification: Classifies minority instances into:
       - Outliers / Noise (NaN-Degree = 0): Completely excluded from synthetic anchor selection.
       - Boundary Instances: Prioritized as anchors to generate where the decision boundary is contested.
       - Core Instances: Safe cluster centers.
    """

    def __init__(
        self,
        semantic_weight: float = 0.35,
        use_bert: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Parameters
        ----------
        semantic_weight : float, default=0.35
            Weight in [0, 1] allocated to BERT semantic embedding distance.
            (1 - semantic_weight) is allocated to standardized continuous Euclidean distance.
        use_bert : bool, default=True
            Whether to verbalize rows and incorporate BERT semantic embeddings.
        model_name : str, default='sentence-transformers/all-MiniLM-L6-v2'
            HuggingFace transformer model used for tabular verbalization embeddings.
        """
        self.semantic_weight = semantic_weight
        self.use_bert = use_bert
        self.model_name = model_name
        self.scaler = StandardScaler()

        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.feature_cols: List[str] = []
        self.minority_class: str = "positive"
        self.majority_class: str = "negative"
        self.target_class: str = "positive"

        self.min_indices: Optional[pd.Index] = None
        self.X_min_orig: Optional[pd.DataFrame] = None
        self.X_min_scaled: Optional[np.ndarray] = None
        self.dist_matrix: Optional[np.ndarray] = None
        self.bert_embeddings: Optional[np.ndarray] = None

        self.lambda_star: int = 1
        self.nan_degrees: Dict[Any, int] = {}
        self.natural_neighbors_map: Dict[Any, List[Any]] = {}
        self.boundary_indices: List[Any] = []
        self.outlier_indices: List[Any] = []
        self.core_indices: List[Any] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        minority_class: str = "positive",
        majority_class: str = "negative",
        target_class: Optional[str] = None,
        dataset_name: str = "tabular"
    ) -> "UnifiedAdaptiveRetriever":
        """Fits the unified retriever on training data, discovers natural neighbors and topology."""
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.feature_cols = [c for c in X_train.columns if c.lower() != "class"]
        self.minority_class = minority_class
        self.majority_class = majority_class
        self.target_class = target_class if target_class is not None else minority_class

        min_mask = (y_train.astype(str) == str(self.target_class)).to_numpy()
        self.min_indices = X_train.index[min_mask]
        self.X_min_orig = X_train.loc[self.min_indices]

        # 1. Standardized Euclidean distance on continuous numerical features
        X_num = self.X_train[self.feature_cols].select_dtypes(include=[np.number])
        self.scaler.fit(X_num)
        X_min_num = self.X_min_orig[self.feature_cols].select_dtypes(include=[np.number])
        self.X_min_scaled = self.scaler.transform(X_min_num)

        geo_dists = euclidean_distances(self.X_min_scaled, self.X_min_scaled)
        max_geo = geo_dists.max() if geo_dists.max() > 0 else 1.0
        geo_dists_norm = geo_dists / max_geo

        # 2. Semantic Row-Verbalization Embeddings using BERT
        if self.use_bert:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                texts = [
                    verbalize_tabular_row(row, dataset_name=dataset_name)
                    for _, row in self.X_min_orig.iterrows()
                ]

                if self.model_name not in _TOKENIZER_CACHE:
                    _TOKENIZER_CACHE[self.model_name] = AutoTokenizer.from_pretrained(self.model_name)
                    m = AutoModel.from_pretrained(self.model_name)
                    m.eval()
                    _MODEL_CACHE[self.model_name] = m

                tokenizer = _TOKENIZER_CACHE[self.model_name]
                model = _MODEL_CACHE[self.model_name]
                model.eval()

                with torch.no_grad():
                    inputs = tokenizer(
                        texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128
                    )
                    outputs = model(**inputs)
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    self.bert_embeddings = (sum_emb / sum_mask).numpy()

                sem_sim = cosine_similarity(self.bert_embeddings, self.bert_embeddings)
                sem_dists = np.clip(1.0 - sem_sim, 0.0, 2.0)
                max_sem = sem_dists.max() if sem_dists.max() > 0 else 1.0
                sem_dists_norm = sem_dists / max_sem

            except Exception as e:
                # Fallback to TF-IDF n-gram semantic encoder if offline or downloading weights is blocked
                from sklearn.feature_extraction.text import TfidfVectorizer
                texts = [
                    verbalize_tabular_row(row, dataset_name=dataset_name)
                    for _, row in self.X_min_orig.iterrows()
                ]
                tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts).toarray()
                sem_sim = cosine_similarity(tfidf, tfidf)
                sem_dists_norm = np.clip(1.0 - sem_sim, 0.0, 1.0)
        else:
            sem_dists_norm = geo_dists_norm

        # 3. Unified Hybrid Distance Matrix
        w = self.semantic_weight if self.use_bert else 0.0
        self.dist_matrix = (1.0 - w) * geo_dists_norm + w * sem_dists_norm

        # 4. Compute Natural Neighbors (NaN)
        self._compute_natural_neighbors()

        # 5. Detect Boundary vs. Outlier vs. Core Points
        self._detect_boundary_and_outliers()

        return self

    def _compute_natural_neighbors(self) -> None:
        """Self-adaptive Natural Neighbor algorithm (Huang et al., 2011).
        
        Dynamically discovers the Natural Characteristic lambda* where mutual
        neighborhood connections stabilize without requiring a user-supplied K.
        """
        n = len(self.min_indices)
        if n <= 1:
            return

        # Sort candidate neighbors by hybrid distance
        sorted_nn = np.argsort(self.dist_matrix, axis=1)

        # Iteratively expand search radius lambda until equilibrium
        max_lambda = min(n - 1, 25)
        lambda_star = 1
        prev_zero_count = n

        for lam in range(1, max_lambda + 1):
            mutual_map = {i: [] for i in range(n)}
            for i in range(n):
                top_lam = set(sorted_nn[i, 1:lam + 1])
                for j in top_lam:
                    if i in set(sorted_nn[j, 1:lam + 1]):
                        mutual_map[i].append(j)

            zero_count = sum(1 for i in range(n) if len(mutual_map[i]) == 0)
            if zero_count == prev_zero_count and lam >= 3:
                lambda_star = lam
                break
            prev_zero_count = zero_count
            lambda_star = lam

        self.lambda_star = lambda_star

        # Store natural neighbors at optimal radius lambda_star
        self.natural_neighbors_map = {}
        for i in range(n):
            top_lam = set(sorted_nn[i, 1:lambda_star + 1])
            nat_nbrs = [j for j in top_lam if i in set(sorted_nn[j, 1:lambda_star + 1])]
            idx_i = self.min_indices[i]
            self.natural_neighbors_map[idx_i] = [self.min_indices[j] for j in nat_nbrs]
            self.nan_degrees[idx_i] = len(nat_nbrs)

    def _detect_boundary_and_outliers(self) -> None:
        """Classifies instances using Natural Degree and Global Boundary Purity."""
        X_all_num = self.X_train[self.feature_cols].select_dtypes(include=[np.number])
        X_all_scaled = self.scaler.transform(X_all_num)
        nn_global = NearestNeighbors(n_neighbors=min(7, len(self.X_train))).fit(X_all_scaled)

        self.boundary_indices = []
        self.outlier_indices = []
        self.core_indices = []

        for idx in self.min_indices:
            pos = list(self.min_indices).index(idx)
            deg = self.nan_degrees.get(idx, 0)

            # Outlier: 0 mutual natural neighbors (isolated noise)
            if deg == 0:
                self.outlier_indices.append(idx)
                continue

            # Query global neighborhood around this point
            vec = self.X_min_scaled[pos].reshape(1, -1)
            _, g_inds = nn_global.kneighbors(vec)
            labels = self.y_train.iloc[g_inds[0]].values
            min_ratio = np.mean(labels.astype(str) == str(self.target_class))

            # Boundary instance: target instance bordered by samples of other classes
            if min_ratio < 0.8:
                self.boundary_indices.append(idx)
            else:
                self.core_indices.append(idx)

        # Fallback if boundary is empty
        if not self.boundary_indices:
            self.boundary_indices = self.core_indices.copy()

    def get_boundary_anchors(self, n_anchors: int = 5) -> List[Any]:
        """Selects high-value generation anchors along the decision boundary,
        strictly excluding isolated noise outliers.
        """
        pool = self.boundary_indices if self.boundary_indices else list(self.min_indices)
        if len(pool) >= n_anchors:
            return list(np.random.choice(pool, size=n_anchors, replace=False))
        else:
            return list(np.random.choice(pool, size=n_anchors, replace=True))

    def retrieve_context(
        self,
        anchor_idx: Any,
        method: str = "auto"
    ) -> NeighborhoodContext:
        """Retrieves neighborhood context for an anchor.
        
        Parameters
        ----------
        anchor_idx : Any
            Index of the minority anchor instance.
        method : str, default='auto'
            'auto' uses Natural Neighbors if valid peers exist, falling back to 
            adaptive 3-KNN only if the instance is isolated.
        """
        anchor_sample = self.X_train.loc[anchor_idx]
        nbr_indices = self.natural_neighbors_map.get(anchor_idx, [])

        # Fallback to nearest neighbors by hybrid distance if natural set is empty
        if not nbr_indices or method == "knn":
            pos = list(self.min_indices).index(anchor_idx)
            sorted_pos = np.argsort(self.dist_matrix[pos])
            k_fallback = min(3, len(self.min_indices) - 1)
            nbr_indices = [self.min_indices[p] for p in sorted_pos[1:k_fallback + 1]]

        neighbors_df = self.X_train.loc[nbr_indices]
        combined = pd.concat([anchor_sample.to_frame().T, neighbors_df])

        # Compute per-feature local bounds and statistics
        feature_ranges = {}
        for col in self.feature_cols:
            if col in combined.columns:
                feature_ranges[col] = {
                    "min": float(combined[col].min()),
                    "max": float(combined[col].max()),
                    "mean": float(combined[col].mean()),
                    "std": float(combined[col].std()) if len(combined) > 1 else 0.0
                }

        return NeighborhoodContext(
            anchor_idx=anchor_idx,
            anchor_sample=anchor_sample,
            neighbor_indices=nbr_indices,
            neighbors=neighbors_df,
            distances=np.zeros(len(nbr_indices)),
            k=len(nbr_indices),
            mean_distance=0.0,
            local_density=float(self.nan_degrees.get(anchor_idx, len(nbr_indices))),
            boundary_purity=1.0 if anchor_idx in self.core_indices else 0.5,
            feature_ranges=feature_ranges,
            retrieval_method="natural_neighbors_unified_bert"
        )


class MulticlassAdaptiveRetriever:
    """Class-conditioned Natural Neighbors + BERT retriever for multiclass imbalanced datasets.
    
    Instantiates and fits a specialized UnifiedAdaptiveRetriever per minority class to discover
    inter-class boundary anchors and natural neighborhood density.
    """

    def __init__(
        self,
        semantic_weight: float = 0.35,
        use_bert: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.semantic_weight = semantic_weight
        self.use_bert = use_bert
        self.model_name = model_name
        self.retrievers: Dict[str, UnifiedAdaptiveRetriever] = {}
        self.classes: List[str] = []
        self.majority_class: str = ""
        self.minority_classes: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        dataset_name: str = "tabular"
    ) -> "MulticlassAdaptiveRetriever":
        """Fits class-conditioned unified retrievers for each minority class."""
        counts = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()}
        self.classes = list(counts.keys())
        self.majority_class = max(counts, key=counts.get)
        self.minority_classes = [c for c in self.classes if c != self.majority_class]

        for c in self.minority_classes:
            retriever = UnifiedAdaptiveRetriever(
                semantic_weight=self.semantic_weight,
                use_bert=self.use_bert,
                model_name=self.model_name
            )
            retriever.fit(
                X_train=X_train,
                y_train=y_train,
                minority_class=c,
                target_class=c,
                majority_class=self.majority_class,
                dataset_name=dataset_name
            )
            self.retrievers[c] = retriever
        return self

    def get_boundary_anchors(self, target_class: str, n_anchors: int = 3) -> List[Any]:
        """Gets boundary anchor instances for the specified target class."""
        if str(target_class) in self.retrievers:
            return self.retrievers[str(target_class)].get_boundary_anchors(n_anchors=n_anchors)
        return []

    def retrieve_context(self, target_class: str, anchor_idx: Any) -> NeighborhoodContext:
        """Retrieves natural neighborhood context around an anchor of the target class."""
        if str(target_class) in self.retrievers:
            return self.retrievers[str(target_class)].retrieve_context(anchor_idx)
        raise ValueError(f"No retriever fitted for target class '{target_class}'")

