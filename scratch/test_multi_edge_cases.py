import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.loader import load_splits
from src.evaluation.metrics import evaluate_model
from src.baselines.gamo import GAMOResampler
from src.baselines.deep_smote import DeepSMOTEResampler
from src.baselines.tabular_vae import TabularVAEResampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier

test_datasets = ["haberman", "yeast", "credit-g"]
for d in test_datasets:
    X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(dataset_name=d, splits_dir="data/splits")
    n_pos = int((y_train == "positive").sum())
    k = min(5, max(1, n_pos - 1))
    print(f"\n--- Testing {d}: n_pos={n_pos}, k={k} ---")
    
    # Test SMOTE
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: resampled to {len(X_res)}")
    
    # Test Borderline-1 & 2
    # BorderlineSMOTE needs at least k_neighbors + 1 samples
    k_bs = min(k, max(1, n_pos - 2)) if n_pos > 2 else 1
    bs1 = BorderlineSMOTE(k_neighbors=k_bs, kind="borderline-1", random_state=42)
    X_b1, y_b1 = bs1.fit_resample(X_train, y_train)
    print(f"BorderlineSMOTE: resampled to {len(X_b1)}")
    
    # Test GAMO
    gamo = GAMOResampler(epochs=20, random_state=42)
    X_gamo, y_gamo = gamo.fit_resample(X_train, y_train)
    print(f"GAMO: resampled to {len(X_gamo)}")
    
    # Test DeepSMOTE
    deep_smote = DeepSMOTEResampler(epochs=20, k_neighbors=k, random_state=42)
    X_ds, y_ds = deep_smote.fit_resample(X_train, y_train)
    print(f"DeepSMOTE: resampled to {len(X_ds)}")
    
    # Test Tabular VAE
    vae = TabularVAEResampler(epochs=20, random_state=42)
    X_vae, y_vae = vae.fit_resample(X_train, y_train)
    print(f"Tabular VAE: resampled to {len(X_vae)}")

print("\nAll test datasets passed successfully!")
