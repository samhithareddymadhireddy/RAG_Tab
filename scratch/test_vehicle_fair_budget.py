import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from src.data.loader import load_splits
from src.baselines.resamplers import apply_resampling
from run_multiclass_experiments import run_rag_tab_multiclass
from src.generation.llm_client import GeminiTabularClient
from src.evaluation.metrics import evaluate_model
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits("vehicle_multiclass", splits_dir="data/splits")
y_train = y_train.astype(str)
y_test = y_test.astype(str)

print("Original Training Distribution:")
print(y_train.value_counts())

# 1. Original
hgb = HistGradientBoostingClassifier(random_state=42)
hgb.fit(X_train, y_train)
m_orig = evaluate_model(y_test, hgb.predict(X_test))
print(f"\n1. Original (0 added rows)         -> Macro F1: {m_orig['F1']:.4f}, Prec: {m_orig['Precision']:.4f}")

# 2. SMOTE (Default unconstrained: 20 rows to reach perfect balance)
X_sm20, y_sm20, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, random_state=42)
hgb.fit(X_sm20, y_sm20)
m_sm20 = evaluate_model(y_test, hgb.predict(X_test))
print(f"2. SMOTE (Full balance, 20 rows)   -> Macro F1: {m_sm20['F1']:.4f}, Prec: {m_sm20['Precision']:.4f}")

# 3. SMOTE with budget = 45 rows (matching RAG-Tab's 45 rows)
X_sm45, y_sm45, _ = apply_resampling(X_train, y_train, method="smote", n_synthetic=45, scale_features=True, random_state=42)
hgb.fit(X_sm45, y_sm45)
m_sm45 = evaluate_model(y_test, hgb.predict(X_test))
print(f"3. SMOTE (Matched budget, 45 rows) -> Macro F1: {m_sm45['F1']:.4f}, Prec: {m_sm45['Precision']:.4f}")

# 4. RAG-Tab with smaller budget (~20 rows)
llm_client = GeminiTabularClient(mock_mode=True)
X_rag20, y_rag20, n_rag20 = run_rag_tab_multiclass(
    X_train=X_train,
    y_train=y_train,
    feature_cols=feature_cols,
    target_col=target_col,
    dataset_name="vehicle",
    llm_client=llm_client,
    samples_per_minority=7
)
hgb.fit(X_rag20, y_rag20)
m_rag20 = evaluate_model(y_test, hgb.predict(X_test))
print(f"4. RAG-Tab (Matched budget, {n_rag20} rows) -> Macro F1: {m_rag20['F1']:.4f}, Prec: {m_rag20['Precision']:.4f}")

# 5. RAG-Tab with 45 rows
X_rag45, y_rag45, n_rag45 = run_rag_tab_multiclass(
    X_train=X_train,
    y_train=y_train,
    feature_cols=feature_cols,
    target_col=target_col,
    dataset_name="vehicle",
    llm_client=llm_client,
    samples_per_minority=15
)
hgb.fit(X_rag45, y_rag45)
m_rag45 = evaluate_model(y_test, hgb.predict(X_test))
print(f"5. RAG-Tab (Full setting, {n_rag45} rows)   -> Macro F1: {m_rag45['F1']:.4f}, Prec: {m_rag45['Precision']:.4f}")
