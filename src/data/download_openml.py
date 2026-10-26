"""Automated dataset downloader for the 60 imbalanced benchmark datasets from OpenML."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Complete mapping of the 60 datasets from Table 9 (OpenML benchmark suite)
TABLE_9_DATASETS: Dict[str, Dict[str, Any]] = {
    "mw1": {"openml_name": "mw1", "target_col": "Defective", "minority_val": "Y"},
    "PizzaCutter1": {"openml_name": "PizzaCutter1", "target_col": "Class", "minority_val": None},
    "analcatdata_draft": {"openml_name": "analcatdata_draft", "target_col": "Class", "minority_val": None},
    "PieChart1": {"openml_name": "PieChart1", "target_col": "Class", "minority_val": None},
    "vowel": {"openml_name": "vowel", "target_col": "Class", "minority_val": None},
    "meta": {"openml_name": "meta", "target_col": "Class", "minority_val": None},
    "kc3": {"openml_name": "kc3", "target_col": "problems", "minority_val": "yes"},
    "analcatdata_halloffame": {"openml_name": "analcatdata_halloffame", "target_col": "Class", "minority_val": None},
    "mfeat-morphological": {"openml_name": "mfeat-morphological", "target_col": "class", "minority_val": None},
    "PizzaCutter3": {"openml_name": "PizzaCutter3", "target_col": "Class", "minority_val": None},
    "analcatdata_apnea3": {"openml_name": "analcatdata_apnea3", "target_col": "Class", "minority_val": None},
    "analcatdata_apnea1": {"openml_name": "analcatdata_apnea1", "target_col": "Class", "minority_val": None},
    "CastMetal1": {"openml_name": "CastMetal1", "target_col": "Class", "minority_val": None},
    "analcatdata_apnea2": {"openml_name": "analcatdata_apnea2", "target_col": "Class", "minority_val": None},
    "segment": {"openml_name": "segment", "target_col": "class", "minority_val": None},
    "arsenic-female-bladder": {"openml_name": "arsenic-female-bladder", "target_col": "Class", "minority_val": None},
    "thoracic-surgery": {"openml_name": "ThoracicSurgery", "target_col": "Risk1Yr", "minority_val": "T"},
    "water-treatment": {"openml_name": "water-treatment", "target_col": "Class", "minority_val": None},
    "collins": {"openml_name": "collins", "target_col": "Class", "minority_val": None},
    "soybean": {"openml_name": "soybean", "target_col": "class", "minority_val": None},
    "solar-flare": {"openml_name": "solar-flare", "target_col": "class", "minority_val": None},
    "analcatdata_dmft": {"openml_name": "analcatdata_dmft", "target_col": "Class", "minority_val": None},
    "kc2": {"openml_name": "kc2", "target_col": "problems", "minority_val": "yes"},
    "mofn-3-7-10": {"openml_name": "mofn-3-7-10", "target_col": "Class", "minority_val": None},
    "socmob": {"openml_name": "socmob", "target_col": "Class", "minority_val": None},
    "analcatdata_germangss": {"openml_name": "analcatdata_germangss", "target_col": "Class", "minority_val": None},
    "braziltourism": {"openml_name": "braziltourism", "target_col": "Class", "minority_val": None},
    "blood-transfusion-service-center": {"openml_name": "blood-transfusion-service-center", "target_col": "Class", "minority_val": "2"},
    "primary-tumor": {"openml_name": "primary-tumor", "target_col": "class", "minority_val": None},
    "vehicle": {"openml_name": "vehicle", "target_col": "Class", "minority_val": None},
    "autoUniv-au1-1000": {"openml_name": "autoUniv-au1-1000", "target_col": "Class", "minority_val": None},
    "haberman": {"openml_name": "haberman", "target_col": "Survival_status", "minority_val": "2"},
    "ilpd": {"openml_name": "ilpd", "target_col": "Class", "minority_val": "2"},
    "car": {"openml_name": "car", "target_col": "class", "minority_val": None},
    "credit-g": {"openml_name": "credit-g", "target_col": "class", "minority_val": "bad"},
    "dermatology": {"openml_name": "dermatology", "target_col": "class", "minority_val": None},
    "analcatdata_marketing": {"openml_name": "analcatdata_marketing", "target_col": "Class", "minority_val": None},
    "eucalyptus": {"openml_name": "eucalyptus", "target_col": "Utility", "minority_val": None},
    "emotions": {"openml_name": "emotions", "target_col": "Class", "minority_val": None},
    "vertebra-column": {"openml_name": "vertebra-column", "target_col": "class", "minority_val": None},
    "Titanic": {"openml_name": "titanic", "target_col": "survived", "minority_val": "1"},
    "colleges_aaup": {"openml_name": "colleges_aaup", "target_col": "Class", "minority_val": None},
    "xd6": {"openml_name": "xd6", "target_col": "Class", "minority_val": None},
    "pc1_req": {"openml_name": "pc1_req", "target_col": "Defective", "minority_val": "Y"},
    "qsar-biodeg": {"openml_name": "qsar-biodeg", "target_col": "Class", "minority_val": "2"},
    "sa-heart": {"openml_name": "SAheart", "target_col": "chd", "minority_val": "1"},
    "tic-tac-toe": {"openml_name": "tic-tac-toe", "target_col": "Class", "minority_val": "negative"},
    "diabetes": {"openml_name": "diabetes", "target_col": "class", "minority_val": "tested_positive"},
    "breast-w": {"openml_name": "breast-w", "target_col": "Class", "minority_val": "malignant"},
    "ionosphere": {"openml_name": "ionosphere", "target_col": "class", "minority_val": "b"},
    "tokyo1": {"openml_name": "tokyo1", "target_col": "Class", "minority_val": None},
    "kdd_el_nino-small": {"openml_name": "kdd_el_nino-small", "target_col": "Class", "minority_val": None},
    "wdbc": {"openml_name": "wdbc", "target_col": "Class", "minority_val": "2"},
    "cars": {"openml_name": "cars", "target_col": "class", "minority_val": None},
    "cylinder-bands": {"openml_name": "cylinder-bands", "target_col": "band_type", "minority_val": "band"},
    "spambase": {"openml_name": "spambase", "target_col": "class", "minority_val": "1"},
    "sonar": {"openml_name": "sonar", "target_col": "Class", "minority_val": "M"},
    "yeast": {"openml_name": "yeast", "target_col": "class_protein_localization", "minority_val": None},
}


def download_dataset(
    dataset_key: str,
    raw_dir: str | Path = "data/raw",
    splits_dir: str | Path = "data/splits",
    test_size: float = 0.20,
    as_multiclass: bool = False,
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

    # Drop missing values
    df = df.dropna().reset_index(drop=True)

    save_prefix = f"{dataset_key}_multiclass" if as_multiclass else dataset_key

    if as_multiclass:
        # Preserve all original multiclass labels
        df["Class"] = df[target_col].astype(str)
    else:
        # Encode target into binary minority ('positive') and majority ('negative')
        target_counts = df[target_col].value_counts()
        if min_override and min_override in target_counts:
            minority_label = min_override
        else:
            minority_label = target_counts.idxmin()

        binary_y = df[target_col].apply(lambda x: "positive" if str(x) == str(minority_label) else "negative")
        df["Class"] = binary_y

    feature_cols = [c for c in df.columns if c not in (target_col, "Class")]

    # Convert non-numeric categorical features to category codes
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes.astype(float)

    clean_df = df[feature_cols + ["Class"]].copy()

    # Save raw CSV
    csv_file = raw_path / f"{save_prefix}.csv"
    clean_df.to_csv(csv_file, index=False)

    # Stratified split
    X = clean_df[feature_cols]
    y = clean_df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    train_df = X_tr.copy()
    train_df["Class"] = y_tr.values
    test_df = X_te.copy()
    test_df["Class"] = y_te.values

    tr_file = splits_path / f"{save_prefix}_train.csv"
    te_file = splits_path / f"{save_prefix}_test.csv"
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
