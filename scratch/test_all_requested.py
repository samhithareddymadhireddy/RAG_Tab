import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, SMOTENC, SMOTEN
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier

from src.data.loader import load_splits
from src.evaluation.metrics import evaluate_model
from src.baselines.gamo import GAMOResampler

# Load data
X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
    dataset_name="page-blocks0",
    splits_dir=PROJECT_ROOT / "data/splits"
)
minority_class = "positive"
majority_class = "negative"

print(f"Loaded page-blocks0. X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train value counts:\n{y_train.value_counts()}")

# Test GAMO on page-blocks0
print("\n--- Testing GAMO on binary page-blocks0 ---")
gamo = GAMOResampler(epochs=50, random_state=42)
X_res_gamo, y_res_gamo = gamo.fit_resample(X_train, y_train)
print(f"GAMO resampled shape: {X_res_gamo.shape}, y distribution: {pd.Series(y_res_gamo).value_counts().to_dict()}")

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_res_gamo, y_res_gamo)
y_pred_gamo = clf.predict(X_test)
m_gamo = evaluate_model(y_test, y_pred_gamo, minority_class, majority_class)
print("GAMO Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_gamo.items()})

# Test KMeansSMOTE
print("\n--- Testing KMeansSMOTE ---")
try:
    km_smote = KMeansSMOTE(cluster_balance_threshold=0.01, random_state=42)
    X_km, y_km = km_smote.fit_resample(X_train, y_train)
    clf.fit(X_km, y_km)
    m_km = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
    print("KMeansSMOTE Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_km.items()})
except Exception as e:
    print("KMeansSMOTE failed:", e)

# Test SMOTEENN
print("\n--- Testing SMOTEENN ---")
try:
    smote_enn = SMOTEENN(random_state=42)
    X_enn, y_enn = smote_enn.fit_resample(X_train, y_train)
    clf.fit(X_enn, y_enn)
    m_enn = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
    print("SMOTEENN Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_enn.items()})
except Exception as e:
    print("SMOTEENN failed:", e)

# Test Borderline-1 and Borderline-2
print("\n--- Testing Borderline-1 & Borderline-2 ---")
for kind in ["borderline-1", "borderline-2"]:
    bs = BorderlineSMOTE(kind=kind, random_state=42)
    X_bs, y_bs = bs.fit_resample(X_train, y_train)
    clf.fit(X_bs, y_bs)
    m_bs = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
    print(f"BorderlineSMOTE ({kind}) Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_bs.items()})

# Test EasyEnsembleClassifier
print("\n--- Testing EasyEnsembleClassifier ---")
eec = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)
eec.fit(X_train, y_train)
m_eec = evaluate_model(y_test, eec.predict(X_test), minority_class, majority_class)
print("EasyEnsembleClassifier Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_eec.items()})

# Test BalancedRandomForestClassifier
print("\n--- Testing BalancedRandomForestClassifier ---")
brf = BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
brf.fit(X_train, y_train)
m_brf = evaluate_model(y_test, brf.predict(X_test), minority_class, majority_class)
print("BalancedRandomForestClassifier Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_brf.items()})

# Test SMOTENC & SMOTEN
print("\n--- Testing SMOTENC & SMOTEN ---")
try:
    # Check if there are categorical features or treat integer features as categorical
    int_cols = [i for i, col in enumerate(feature_cols) if X_train[col].dtype == 'int64' or (X_train[col] % 1 == 0).all()]
    print(f"Integer/Discrete columns for SMOTENC: {int_cols} -> {[feature_cols[i] for i in int_cols]}")
    if int_cols:
        smotenc = SMOTENC(categorical_features=int_cols, random_state=42)
        X_nc, y_nc = smotenc.fit_resample(X_train, y_train)
        clf.fit(X_nc, y_nc)
        m_nc = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
        print("SMOTENC Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_nc.items()})
    else:
        print("No categorical columns for SMOTENC.")
except Exception as e:
    print("SMOTENC failed:", e)

try:
    # SMOTEN requires nominal data
    # Let's test if SMOTEN works directly or if it needs discrete/binned features
    smoten = SMOTEN(random_state=42)
    X_n, y_n = smoten.fit_resample(X_train, y_train)
    clf.fit(X_n, y_n)
    m_n = evaluate_model(y_test, clf.predict(X_test), minority_class, majority_class)
    print("SMOTEN Metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_n.items()})
except Exception as e:
    print("SMOTEN directly failed (expected since features are continuous):", e)
