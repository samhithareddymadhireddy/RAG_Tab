"""Comprehensive Binary Imbalanced Classification Benchmark.

Evaluates and compiles all requested oversampling, deep learning, and ensemble methods
on the page-blocks0 binary benchmark dataset.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, SMOTENC, SMOTEN
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier

from src.data.loader import load_splits
from src.evaluation.metrics import evaluate_model
from src.baselines.gamo import GAMOResampler
from src.baselines.deep_smote import DeepSMOTEResampler
from src.baselines.tabular_vae import TabularVAEResampler


def run_benchmark():
    TABLES_DIR = PROJECT_ROOT / "results" / "tables"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset splits
    X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
        dataset_name="page-blocks0",
        splits_dir=PROJECT_ROOT / "data" / "splits"
    )
    minority_class = "positive"
    majority_class = "negative"

    n_orig = len(X_train)
    n_min_orig = int((y_train == minority_class).sum())

    print(f"Loaded page-blocks0: Train={X_train.shape}, Test={X_test.shape}")
    print(f"Class distribution: {y_train.value_counts().to_dict()}")

    records = []

    def evaluate_resampled(method_name, X_res, y_res, n_syn=None):
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, minority_class, majority_class)

        n_min_total = int((pd.Series(y_res) == minority_class).sum())
        syn_added = len(X_res) - n_orig if n_syn is None else n_syn

        rec = {
            "Method": method_name,
            "Total Train Rows": len(X_res),
            "Minority Train Rows": n_min_total,
            "Synthetic Rows Added": syn_added,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "Specificity": metrics["Specificity"],
            "G-Mean": metrics["G-Mean"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"]
        }
        records.append(rec)
        print(f"[{method_name:35s}] F1: {metrics['F1']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | Spec: {metrics['Specificity']:.4f} | FP: {metrics['FP']}")

    # 1. Original Baseline (No Resampling)
    evaluate_resampled("Original (No Resampling)", X_train, y_train, n_syn=0)

    # 2. Random Oversampler (ROS)
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_train, y_train)
    evaluate_resampled("Random Oversampling", X_ros, y_ros)

    # 3. SMOTE (Unscaled)
    smote_unscaled = SMOTE(random_state=42)
    X_sm_u, y_sm_u = smote_unscaled.fit_resample(X_train, y_train)
    evaluate_resampled("SMOTE (Unscaled)", X_sm_u, y_sm_u)

    # 4. SMOTE (Standardized)
    from src.baselines.resamplers import apply_resampling
    X_sm_s, y_sm_s, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, random_state=42)
    evaluate_resampled("SMOTE (Standardized)", X_sm_s, y_sm_s)

    # 5. Borderline-SMOTE (BSMOTE-1)
    bsmote1 = BorderlineSMOTE(kind="borderline-1", random_state=42)
    X_bs1, y_bs1 = bsmote1.fit_resample(X_train, y_train)
    evaluate_resampled("Borderline-SMOTE (BSMOTE-1)", X_bs1, y_bs1)

    # 6. Borderline-SMOTE (BSMOTE-2)
    bsmote2 = BorderlineSMOTE(kind="borderline-2", random_state=42)
    X_bs2, y_bs2 = bsmote2.fit_resample(X_train, y_train)
    evaluate_resampled("Borderline-SMOTE (BSMOTE-2)", X_bs2, y_bs2)

    # 7. ADASYN
    adasyn = ADASYN(random_state=42)
    X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
    evaluate_resampled("ADASYN", X_ada, y_ada)

    # 8. KMeansSMOTE
    try:
        km_smote = KMeansSMOTE(cluster_balance_threshold=0.01, random_state=42)
        X_km, y_km = km_smote.fit_resample(X_train, y_train)
        evaluate_resampled("KMeans-SMOTE", X_km, y_km)
    except Exception as e:
        print(f"KMeansSMOTE skipped: {e}")

    # 9. SMOTE-Tomek
    sm_tomek = SMOTETomek(random_state=42)
    X_st, y_st = sm_tomek.fit_resample(X_train, y_train)
    evaluate_resampled("SMOTE-Tomek", X_st, y_st, n_syn=len(X_st) - n_orig)

    # 10. SMOTE-ENN
    sm_enn = SMOTEENN(random_state=42)
    X_se, y_se = sm_enn.fit_resample(X_train, y_train)
    evaluate_resampled("SMOTE-ENN", X_se, y_se, n_syn=len(X_se) - n_orig)

    # 11. SMOTE-NC (Nominal & Continuous)
    int_cols = [i for i, col in enumerate(feature_cols) if (X_train[col] % 1 == 0).all()]
    smotenc = SMOTENC(categorical_features=int_cols, random_state=42)
    X_nc, y_nc = smotenc.fit_resample(X_train, y_train)
    evaluate_resampled("SMOTE-NC", X_nc, y_nc)

    # 12. SMOTE-N (Nominal with Binned Features)
    kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    X_binned = pd.DataFrame(kbd.fit_transform(X_train), columns=feature_cols)
    smoten = SMOTEN(random_state=42)
    X_n_binned, y_n = smoten.fit_resample(X_binned, y_train)
    # Map back approximately to continuous feature medians
    X_n_cont = X_train.copy()
    X_n_resampled, _ = smoten.fit_resample(X_train.round().astype(int), y_train)
    evaluate_resampled("SMOTE-N", X_n_resampled, y_n)

    # 13. Deep Learning: GAMO (Full Resampling)
    print("Training GAMO (Full Resampling)...")
    gamo = GAMOResampler(epochs=50, random_state=42)
    X_gamo, y_gamo = gamo.fit_resample(X_train, y_train)
    evaluate_resampled("GAMO (Adversarial, Full)", X_gamo, y_gamo)

    # 14. Deep Learning: GAMO (Equal Budget N=50)
    print("Training GAMO (Equal Budget N=50)...")
    gamo_50 = GAMOResampler(epochs=50, random_state=42)
    X_gamo_50, y_gamo_50 = gamo_50.fit_resample(X_train, y_train, n_synthetic=50)
    evaluate_resampled("GAMO (Adversarial, N=50)", X_gamo_50, y_gamo_50, n_syn=50)

    # 15. Deep Learning: DeepSMOTE (Autoencoder + Latent SMOTE)
    print("Training DeepSMOTE...")
    deep_smote = DeepSMOTEResampler(epochs=50, random_state=42)
    X_ds, y_ds = deep_smote.fit_resample(X_train, y_train)
    evaluate_resampled("DeepSMOTE (Autoencoder + SMOTE)", X_ds, y_ds)

    # 16. Deep Learning: Tabular VAE (Variational Autoencoder)
    print("Training Tabular VAE...")
    vae = TabularVAEResampler(epochs=50, random_state=42)
    X_vae, y_vae = vae.fit_resample(X_train, y_train)
    evaluate_resampled("Tabular VAE (Generative)", X_vae, y_vae)

    # 17. Ensemble Classifiers (Direct Classification)
    print("Evaluating EasyEnsembleClassifier...")
    eec = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    eec.fit(X_train, y_train)
    y_pred_eec = eec.predict(X_test)
    m_eec = evaluate_model(y_test, y_pred_eec, minority_class, majority_class)
    records.append({
        "Method": "EasyEnsembleClassifier",
        "Total Train Rows": n_orig,
        "Minority Train Rows": n_min_orig,
        "Synthetic Rows Added": 0,
        "Precision": m_eec["Precision"],
        "Recall": m_eec["Recall"],
        "F1": m_eec["F1"],
        "Specificity": m_eec["Specificity"],
        "G-Mean": m_eec["G-Mean"],
        "TP": m_eec["TP"],
        "FP": m_eec["FP"],
        "TN": m_eec["TN"],
        "FN": m_eec["FN"]
    })
    print(f"[{'EasyEnsembleClassifier':35s}] F1: {m_eec['F1']:.4f} | Prec: {m_eec['Precision']:.4f} | Rec: {m_eec['Recall']:.4f} | Spec: {m_eec['Specificity']:.4f} | FP: {m_eec['FP']}")

    print("Evaluating BalancedRandomForestClassifier...")
    brf = BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    brf.fit(X_train, y_train)
    y_pred_brf = brf.predict(X_test)
    m_brf = evaluate_model(y_test, y_pred_brf, minority_class, majority_class)
    records.append({
        "Method": "BalancedRandomForestClassifier",
        "Total Train Rows": n_orig,
        "Minority Train Rows": n_min_orig,
        "Synthetic Rows Added": 0,
        "Precision": m_brf["Precision"],
        "Recall": m_brf["Recall"],
        "F1": m_brf["F1"],
        "Specificity": m_brf["Specificity"],
        "G-Mean": m_brf["G-Mean"],
        "TP": m_brf["TP"],
        "FP": m_brf["FP"],
        "TN": m_brf["TN"],
        "FN": m_brf["FN"]
    })
    print(f"[{'BalancedRandomForestClassifier':35s}] F1: {m_brf['F1']:.4f} | Prec: {m_brf['Precision']:.4f} | Rec: {m_brf['Recall']:.4f} | Spec: {m_brf['Specificity']:.4f} | FP: {m_brf['FP']}")

    # 18. Retain LLM-Only and RAG-Tab records from existing benchmark
    comp_old_file = TABLES_DIR / "all_methods_comparison_page_blocks0.csv"
    if comp_old_file.exists():
        old_df = pd.read_csv(comp_old_file)
        llm_rows = old_df[old_df["Method"].str.contains("LLM|RAG-Tab", regex=True, na=False)]
        for _, row in llm_rows.iterrows():
            if not any(r["Method"] == row["Method"] for r in records):
                records.append(row.to_dict())

    # Check final_research_leaderboard for any RAG-Tab rows
    final_old_file = TABLES_DIR / "final_research_leaderboard_page_blocks0.csv"
    if final_old_file.exists():
        f_old_df = pd.read_csv(final_old_file)
        rag_rows = f_old_df[f_old_df["Method"].str.contains("RAG-Tab", regex=True, na=False)]
        for _, row in rag_rows.iterrows():
            if not any(r["Method"] == row["Method"] for r in records):
                records.append(row.to_dict())

    benchmark_df = pd.DataFrame(records)

    # Save to both comparison files
    benchmark_df.to_csv(TABLES_DIR / "all_methods_comparison_page_blocks0.csv", index=False)
    benchmark_df.to_csv(TABLES_DIR / "final_research_leaderboard_page_blocks0.csv", index=False)

    print("\nBenchmark updated successfully!")
    print(f"Total methods evaluated: {len(benchmark_df)}")
    print(f"Saved to: {TABLES_DIR / 'all_methods_comparison_page_blocks0.csv'}")
    print(f"Saved to: {TABLES_DIR / 'final_research_leaderboard_page_blocks0.csv'}")


if __name__ == "__main__":
    run_benchmark()
