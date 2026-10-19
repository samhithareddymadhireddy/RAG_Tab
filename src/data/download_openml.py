"""Automated dataset downloader for the 60 imbalanced benchmark datasets from OpenML."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Mapping of Table 9 dataset names to OpenML names/aliases
TABLE_9_DATASETS: Dict[str, Dict[str, Any]] = {
    "haberman": {"openml_name": "haberman", "target_col": "Survival_status", "minority_val": "2"},
    "diabetes": {"openml_name": "diabetes", "target_col": "class", "minority_val": "tested_positive"},
    "blood-transfusion-service-center": {"openml_name": "blood-transfusion-service-center", "target_col": "Class", "minority_val": "2"},
    "credit-g": {"openml_name": "credit-g", "target_col": "class", "minority_val": "bad"},
    "dermatology": {"openml_name": "dermatology", "target_col": "class", "minority_val": None},
    "ionosphere": {"openml_name": "ionosphere", "target_col": "class", "minority_val": "b"},
    "wdbc": {"openml_name": "wdbc", "target_col": "Class", "minority_val": "2"},
    "spambase": {"openml_name": "spambase", "target_col": "class", "minority_val": "1"},
    "segment": {"openml_name": "segment", "target_col": "class", "minority_val": None},
    "vehicle": {"openml_name": "vehicle", "target_col": "Class", "minority_val": None},
    "car": {"openml_name": "car", "target_col": "class", "minority_val": None},
    "kc2": {"openml_name": "kc2", "target_col": "problems", "minority_val": "yes"},
    "kc3": {"openml_name": "kc3", "target_col": "problems", "minority_val": "yes"},
    "mw1": {"openml_name": "mw1", "target_col": "Defective", "minority_val": "Y"},
    "pc1_req": {"openml_name": "pc1_req", "target_col": "Defective", "minority_val": "Y"},
    "tic-tac-toe": {"openml_name": "tic-tac-toe", "target_col": "Class", "minority_val": "negative"},
    "breast-w": {"openml_name": "breast-w", "target_col": "Class", "minority_val": "malignant"},
    "sa-heart": {"openml_name": "SAheart", "target_col": "chd", "minority_val": "1"},
    "qsar-biodeg": {"openml_name": "qsar-biodeg", "target_col": "Class", "minority_val": "2"},
    "vowel": {"openml_name": "vowel", "target_col": "Class", "minority_val": None},
    "sonar": {"openml_name": "sonar", "target_col": "Class", "minority_val": "M"},
    "yeast": {"openml_name": "yeast", "target_col": "class_protein_localization", "minority_val": None},
}


def download_dataset(
    dataset_key: str,
    raw_dir: str | Path = "data/raw",
    splits_dir: str | Path = "data/splits",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Downloads an OpenML dataset, encodes target, and prepares frozen splits."""
    if dataset_key not in TABLE_9_DATASETS:
        openml_name = dataset_key
        target_override = None
        min_override = None
    else:
        cfg = TABLE_9_DATASETS[dataset_key]
        openml_name = cfg["openml_name"]
        target_override = cfg["target_col"]
        min_override = cfg["minority_val"]

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    # Fetch from OpenML
    print(f"Fetching '{openml_name}' from OpenML...")
    bunch = fetch_openml(name=openml_name, as_frame=True, parser="auto")
    df = bunch.frame.copy()

    # Determine target column
    target_col = target_override or (bunch.target_names[0] if bunch.target_names else df.columns[-1])
    if target_col not in df.columns and len(df.columns) > 0:
        target_col = df.columns[-1]

    # Handle missing values
    df = df.dropna().reset_index(drop=True)

    # Encode target into binary minority ('positive') and majority ('negative')
    target_counts = df[target_col].value_counts()
    if min_override and min_override in target_counts:
        minority_label = min_override
    else:
        minority_label = target_counts.idxmin()

    majority_labels = [c for c in target_counts.index if c != minority_label]
    
    # Binary mapping: minority -> 'positive', others -> 'negative'
    binary_y = df[target_col].apply(lambda x: "positive" if x == minority_label else "negative")
    df["Class"] = binary_y

    feature_cols = [c for c in df.columns if c not in (target_col, "Class")]

    # Convert non-numeric categorical features to category codes
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes.astype(float)

    clean_df = df[feature_cols + ["Class"]].copy()

    # Save raw CSV
    csv_file = raw_path / f"{dataset_key}.csv"
    clean_df.to_csv(csv_file, index=False)

    # Stratified split
    X = clean_df[feature_cols]
    y = clean_df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    train_df = X_tr.copy()
    train_df["Class"] = y_tr.values
    test_df = X_te.copy()
    test_df["Class"] = y_te.values

    tr_file = splits_path / f"{dataset_key}_train.csv"
    te_file = splits_path / f"{dataset_key}_test.csv"
    train_df.to_csv(tr_file, index=False)
    test_df.to_csv(te_file, index=False)

    n_maj = (clean_df["Class"] == "negative").sum()
    n_min = (clean_df["Class"] == "positive").sum()
    ir = n_maj / max(1, n_min)

    meta = {
        "dataset": dataset_key,
        "total_samples": len(clean_df),
        "n_features": len(feature_cols),
        "majority_count": int(n_maj),
        "minority_count": int(n_min),
        "imbalance_ratio": round(float(ir), 2),
        "raw_file": str(csv_file),
        "train_file": str(tr_file),
        "test_file": str(te_file)
    }

    print(f"Successfully processed '{dataset_key}': {len(clean_df)} samples, {len(feature_cols)} features, IR: {ir:.2f}")
    return tr_file, te_file, meta


def download_batch_datasets(dataset_keys: List[str]) -> pd.DataFrame:
    """Downloads a batch of datasets from Table 9 and returns a summary metadata DataFrame."""
    records = []
    for key in dataset_keys:
        try:
            _, _, meta = download_dataset(key)
            records.append(meta)
        except Exception as e:
            print(f"Failed to download '{key}': {e}")
    return pd.DataFrame(records)
