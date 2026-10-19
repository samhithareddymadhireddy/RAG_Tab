"""Dataset loading, validation, train/test splitting, and frozen split management for RAG-Tab."""
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split


def find_data_file(filename: str = "page-blocks0.dat") -> Path:
    """Finds a dataset file checking common relative locations."""
    candidates = [
        Path(filename),
        Path("data/raw") / filename,
        Path("../data/raw") / filename,
        Path("../../data/raw") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Could not locate {filename}. Searched: {[str(c) for c in candidates]}"
    )


def load_keel(file_path: str | Path) -> pd.DataFrame:
    """Parses a KEEL .dat format file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the KEEL .dat file.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed attributes and data rows.
    """
    path = Path(file_path)
    if not path.exists():
        path = find_data_file(path.name)

    attributes = []
    data_started = False
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split()
                attributes.append(parts[1])
            elif lower.startswith("@data"):
                data_started = True
            elif data_started:
                rows.append([x.strip() for x in line.split(",")])

    df = pd.DataFrame(rows, columns=attributes)
    return df


def prepare_data(
    file_path: str | Path,
    target_col: str = "Class",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """Loads a KEEL dataset, converts numeric features, and produces a stratified split.

    Parameters
    ----------
    file_path : str or Path
        Path to the dataset file.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data reserved for the test set.
    random_state : int
        Seed for reproducible splitting.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_cols, target_col)
    """
    df = load_keel(file_path)
    feature_cols = [c for c in df.columns if c != target_col]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_cols, target_col


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_col: str = "Class",
    dataset_name: str = "page-blocks0",
    splits_dir: str | Path = "data/splits"
) -> Tuple[Path, Path]:
    """Saves frozen train and test splits to disk for reproducible experiments."""
    sdir = Path(splits_dir)
    if not sdir.exists():
        # Check parent relative path for notebooks
        alt_sdir = Path("..") / splits_dir
        if alt_sdir.parent.exists():
            sdir = alt_sdir
    sdir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    train_path = sdir / f"{dataset_name}_train.csv"
    test_path = sdir / f"{dataset_name}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


def load_splits(
    dataset_name: str = "page-blocks0",
    target_col: str = "Class",
    splits_dir: str | Path = "data/splits"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """Loads previously saved frozen train and test splits."""
    sdir = Path(splits_dir)
    train_path = sdir / f"{dataset_name}_train.csv"
    test_path = sdir / f"{dataset_name}_test.csv"

    if not train_path.exists() or not test_path.exists():
        # Try relative to parent (if in notebooks/)
        alt_train = Path("..") / splits_dir / f"{dataset_name}_train.csv"
        alt_test = Path("..") / splits_dir / f"{dataset_name}_test.csv"
        if alt_train.exists() and alt_test.exists():
            train_path, test_path = alt_train, alt_test
        else:
            raise FileNotFoundError(
                f"Frozen splits not found for {dataset_name} in {splits_dir}. "
                "Run prepare_data() and save_splits() first."
            )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    return X_train, X_test, y_train, y_test, feature_cols, target_col
