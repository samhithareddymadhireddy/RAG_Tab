"""BERT-based semantic verbalization and embedding retriever for tabular RAG."""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .neighborhood import NeighborhoodContext


def verbalize_tabular_row(
    row: pd.Series,
    feature_descriptions: Optional[Dict[str, str]] = None,
    dataset_name: str = "tabular"
) -> str:
    """Verbalizes a single tabular row into a descriptive natural language sentence."""
    parts = []
    for col, val in row.items():
        if col.lower() == "class":
            parts.append(f"classified as {val}")
        else:
            desc = feature_descriptions.get(col, col) if feature_descriptions else col
            if isinstance(val, (int, float)):
                parts.append(f"{desc} is {val:.2f}" if isinstance(val, float) else f"{desc} is {val}")
            else:
                parts.append(f"{desc} is {val}")

    return f"A {dataset_name} sample where " + ", ".join(parts) + "."


class BertTabularRetriever:
    """Retrieves local minority neighbors using BERT semantic embeddings and cosine similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        feature_descriptions: Optional[Dict[str, str]] = None
    ):
        self.model_name = model_name
        self.feature_descriptions = feature_descriptions or {}
        self.tokenizer = None
        self.model = None
        self.use_deep_bert = False

        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.min_indices: Optional[pd.Index] = None
        self.minority_texts: List[str] = []
        self.minority_embeddings: Optional[np.ndarray] = None
        self.minority_class: str = "positive"
        self.majority_class: str = "negative"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        minority_class: str = "positive",
        majority_class: str = "negative",
        dataset_name: str = "tabular"
    ) -> "BertTabularRetriever":
        """Verbalizes minority samples and computes dense semantic embeddings."""
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.minority_class = minority_class
        self.majority_class = majority_class

        min_mask = (y_train == minority_class).to_numpy()
        self.min_indices = X_train.index[min_mask]
        X_min = X_train.loc[self.min_indices]

        # Verbalize each minority row into text
        self.minority_texts = [
            verbalize_tabular_row(row, self.feature_descriptions, dataset_name=dataset_name)
            for _, row in X_min.iterrows()
        ]

        # Try to initialize BERT with transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            print(f"Loading BERT encoder '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.use_deep_bert = True

            embeddings = []
            with torch.no_grad():
                for text in self.minority_texts:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    outputs = self.model(**inputs)
                    # Mean pooling over token embeddings
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    mean_pooled = sum_embeddings / sum_mask
                    embeddings.append(mean_pooled.squeeze().numpy())

            self.minority_embeddings = np.array(embeddings)
            print(f"Computed {len(self.minority_embeddings)} BERT embeddings (dim: {self.minority_embeddings.shape[1]}).")

        except Exception as e:
            # Fallback to TF-IDF semantic character n-gram encoder if offline or downloading weights is blocked
            print(f"[Note] HuggingFace weights download skipped ({e}). Using Scikit-Learn TF-IDF semantic encoder.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.minority_embeddings = vectorizer.fit_transform(self.minority_texts).toarray()
            self.use_deep_bert = False

        return self

    def retrieve_by_index(
        self,
        anchor_idx: Any,
        k: int = 5
    ) -> NeighborhoodContext:
        """Retrieves K nearest minority neighbors using BERT semantic cosine similarity."""
        if self.minority_embeddings is None:
            raise RuntimeError("Retriever is not fitted. Call .fit() first.")

        if anchor_idx not in self.min_indices:
            raise ValueError(f"Index {anchor_idx} not in minority set.")

        pos_in_min = list(self.min_indices).index(anchor_idx)
        anchor_emb = self.minority_embeddings[pos_in_min].reshape(1, -1)
        anchor_sample = self.X_train.loc[anchor_idx]

        # Compute cosine similarity
        similarities = cosine_similarity(anchor_emb, self.minority_embeddings)[0]
        # Cosine distance = 1 - similarity
        dists = 1.0 - similarities

        # Rank by distance, exclude self
        ranked_indices = np.argsort(dists)
        valid_pairs = [(dists[i], i) for i in ranked_indices if i != pos_in_min][:k]

        neighbor_dists = np.array([p[0] for p in valid_pairs])
        neighbor_pos = [p[1] for p in valid_pairs]
        neighbor_df_indices = [self.min_indices[p] for p in neighbor_pos]
        neighbors_df = self.X_train.loc[neighbor_df_indices].copy()

        mean_dist = float(np.mean(neighbor_dists)) if len(neighbor_dists) > 0 else 0.0
        local_density = float(1.0 / (mean_dist + 1e-6))

        # Local bounding box
        combined = pd.concat([anchor_sample.to_frame().T, neighbors_df])
        feature_ranges = {}
        for col in self.X_train.columns:
            if col.lower() != "class":
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
            distances=neighbor_dists,
            k=k,
            mean_distance=mean_dist,
            local_density=local_density,
            boundary_purity=1.0,
            feature_ranges=feature_ranges,
            retrieval_method="bert_semantic_cosine"
        )
