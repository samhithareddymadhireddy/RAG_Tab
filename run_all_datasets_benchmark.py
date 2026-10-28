"""Multi-Dataset Imbalanced Learning Benchmark across all available datasets.

Evaluates all requested oversampling, deep learning, ensemble, and RAG-Tab methods
across every binary dataset in data/splits.
"""
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier

from src.data.loader import load_splits
from src.evaluation.metrics import evaluate_model
from src.baselines.resamplers import apply_resampling
from src.baselines.gamo import GAMOResampler
from src.baselines.deep_smote import DeepSMOTEResampler
from src.baselines.tabular_vae import TabularVAEResampler


def get_all_binary_datasets(splits_dir="data/splits"):
    sdir = Path(splits_dir)
    train_files = sorted(list(sdir.glob("*_train.csv")))
    binary_datasets = []
    for tf in train_files:
        d_name = tf.name.replace("_train.csv", "")
        test_file = sdir / f"{d_name}_test.csv"
        if not test_file.exists():
            continue
        try:
            df = pd.read_csv(tf)
            target_col = "Class" if "Class" in df.columns else df.columns[-1]
            vc = df[target_col].value_counts().to_dict()
            if len(vc) == 2 and "positive" in vc and "negative" in vc:
                binary_datasets.append(d_name)
        except Exception:
            pass
    return binary_datasets


def run_benchmark():
    TABLES_DIR = PROJECT_ROOT / "results" / "tables"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "all_datasets_comprehensive_benchmark.csv"
    macro_csv = TABLES_DIR / "all_datasets_macro_leaderboard.csv"

    datasets = get_all_binary_datasets()
    print(f"Discovered {len(datasets)} binary classification datasets:")
    print(", ".join(datasets))

    # Load existing ablation records if available
    ablation_map = {}
    ablation_file = TABLES_DIR / "ablation_zero_few_rag_bert.csv"
    if ablation_file.exists():
        ab_df = pd.read_csv(ablation_file)
        for _, row in ab_df.iterrows():
            d = row["Dataset"]
            if d not in ablation_map:
                ablation_map[d] = []
            ablation_map[d].append(row.to_dict())

    # Checkpoint recovery
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        completed_datasets = set(existing_df["Dataset"].unique())
        all_records = existing_df.to_dict("records")
        print(f"[Checkpoint] Found {len(completed_datasets)} already completed datasets. Resuming...")
    else:
        completed_datasets = set()
        all_records = []

    minority_class = "positive"
    majority_class = "negative"

    total_start = time.time()

    for idx, d_name in enumerate(datasets, 1):
        if d_name in completed_datasets:
            print(f"[{idx}/{len(datasets)}] Skipping already completed: {d_name}")
            continue

        print(f"\n[{idx}/{len(datasets)}] Benchmarking on: {d_name}...")
        t0 = time.time()
        try:
            X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
                dataset_name=d_name,
                splits_dir="data/splits"
            )
        except Exception as e:
            print(f"  [Error] Could not load {d_name}: {e}")
            continue

        n_orig = len(X_train)
        n_pos = int((y_train == minority_class).sum())
        n_neg = int((y_train == majority_class).sum())
        k = min(5, max(1, n_pos - 1))
        k_bs = min(k, max(1, n_pos - 2)) if n_pos > 2 else 1

        print(f"  Train: {n_orig} (Pos: {n_pos}, Neg: {n_neg}, Features: {len(feature_cols)}) | k={k}")

        def eval_and_record(method_name, family, X_res, y_res, n_syn=None):
            clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)
            m = evaluate_model(y_test, y_pred, minority_class, majority_class)
            all_records.append({
                "Dataset": d_name,
                "Method": method_name,
                "Family": family,
                "Total Train Rows": len(X_res),
                "Minority Train Rows": int((pd.Series(y_res) == minority_class).sum()),
                "Synthetic Rows Added": len(X_res) - n_orig if n_syn is None else n_syn,
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "Specificity": m["Specificity"],
                "G-Mean": m["G-Mean"],
                "TP": m["TP"],
                "FP": m["FP"],
                "TN": m["TN"],
                "FN": m["FN"]
            })

        # 1. Original (No Resampling)
        eval_and_record("Original (No Resampling)", "Baseline", X_train, y_train, n_syn=0)

        # 2. Random Oversampler
        try:
            ros = RandomOverSampler(random_state=42)
            X_ros, y_ros = ros.fit_resample(X_train, y_train)
            eval_and_record("Random Oversampling", "Heuristic", X_ros, y_ros)
        except Exception as e:
            print(f"  [Warn] ROS failed: {e}")

        # 3. SMOTE (Standardized)
        try:
            X_sm, y_sm, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, random_state=42)
            eval_and_record("SMOTE (Standardized)", "Heuristic", X_sm, y_sm)
        except Exception as e:
            print(f"  [Warn] SMOTE failed: {e}")

        # 4. Borderline-SMOTE (BSMOTE-1)
        try:
            bs1 = BorderlineSMOTE(k_neighbors=k_bs, kind="borderline-1", random_state=42)
            X_bs1, y_bs1 = bs1.fit_resample(X_train, y_train)
            eval_and_record("Borderline-SMOTE (BSMOTE-1)", "Heuristic", X_bs1, y_bs1)
        except Exception as e:
            print(f"  [Warn] BSMOTE-1 failed: {e}")

        # 5. Borderline-SMOTE (BSMOTE-2)
        try:
            bs2 = BorderlineSMOTE(k_neighbors=k_bs, kind="borderline-2", random_state=42)
            X_bs2, y_bs2 = bs2.fit_resample(X_train, y_train)
            eval_and_record("Borderline-SMOTE (BSMOTE-2)", "Heuristic", X_bs2, y_bs2)
        except Exception as e:
            print(f"  [Warn] BSMOTE-2 failed: {e}")

        # 6. ADASYN
        try:
            ada = ADASYN(n_neighbors=k, random_state=42)
            X_ada, y_ada = ada.fit_resample(X_train, y_train)
            eval_and_record("ADASYN", "Heuristic", X_ada, y_ada)
        except Exception as e:
            print(f"  [Warn] ADASYN failed: {e}")

        # 7. KMeans-SMOTE
        try:
            km = KMeansSMOTE(k_neighbors=k, cluster_balance_threshold=0.01, random_state=42)
            X_km, y_km = km.fit_resample(X_train, y_train)
            eval_and_record("KMeans-SMOTE", "Heuristic", X_km, y_km)
        except Exception:
            # Fallback if clusters too sparse
            try:
                km = KMeansSMOTE(k_neighbors=k, cluster_balance_threshold=0.0, random_state=42)
                X_km, y_km = km.fit_resample(X_train, y_train)
                eval_and_record("KMeans-SMOTE", "Heuristic", X_km, y_km)
            except Exception as e:
                print(f"  [Warn] KMeansSMOTE skipped: {e}")

        # 8. SMOTE-Tomek
        try:
            st = SMOTETomek(random_state=42)
            X_st, y_st = st.fit_resample(X_train, y_train)
            eval_and_record("SMOTE-Tomek", "Cleaning", X_st, y_st, n_syn=len(X_st) - n_orig)
        except Exception as e:
            print(f"  [Warn] SMOTE-Tomek failed: {e}")

        # 9. SMOTE-ENN
        try:
            se = SMOTEENN(random_state=42)
            X_se, y_se = se.fit_resample(X_train, y_train)
            eval_and_record("SMOTE-ENN", "Cleaning", X_se, y_se, n_syn=len(X_se) - n_orig)
        except Exception as e:
            print(f"  [Warn] SMOTE-ENN failed: {e}")

        # 10. Deep Learning: GAMO
        try:
            gamo = GAMOResampler(epochs=35, random_state=42)
            X_gamo, y_gamo = gamo.fit_resample(X_train, y_train)
            eval_and_record("GAMO (Adversarial)", "Deep Learning", X_gamo, y_gamo)
        except Exception as e:
            print(f"  [Warn] GAMO failed: {e}")

        # 11. Deep Learning: DeepSMOTE
        try:
            deep_smote = DeepSMOTEResampler(epochs=35, k_neighbors=k, random_state=42)
            X_ds, y_ds = deep_smote.fit_resample(X_train, y_train)
            eval_and_record("DeepSMOTE (Autoencoder + SMOTE)", "Deep Learning", X_ds, y_ds)
        except Exception as e:
            print(f"  [Warn] DeepSMOTE failed: {e}")

        # 12. Deep Learning: Tabular VAE
        try:
            vae = TabularVAEResampler(epochs=35, random_state=42)
            X_vae, y_vae = vae.fit_resample(X_train, y_train)
            eval_and_record("Tabular VAE (Generative)", "Deep Learning", X_vae, y_vae)
        except Exception as e:
            print(f"  [Warn] Tabular VAE failed: {e}")

        # 13. Ensemble: EasyEnsembleClassifier
        try:
            eec = EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1)
            eec.fit(X_train, y_train)
            m_eec = evaluate_model(y_test, eec.predict(X_test), minority_class, majority_class)
            all_records.append({
                "Dataset": d_name,
                "Method": "EasyEnsembleClassifier",
                "Family": "Ensemble",
                "Total Train Rows": n_orig,
                "Minority Train Rows": n_pos,
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
        except Exception as e:
            print(f"  [Warn] EasyEnsemble failed: {e}")

        # 14. Ensemble: BalancedRandomForestClassifier
        try:
            brf = BalancedRandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            brf.fit(X_train, y_train)
            m_brf = evaluate_model(y_test, brf.predict(X_test), minority_class, majority_class)
            all_records.append({
                "Dataset": d_name,
                "Method": "BalancedRandomForestClassifier",
                "Family": "Ensemble",
                "Total Train Rows": n_orig,
                "Minority Train Rows": n_pos,
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
        except Exception as e:
            print(f"  [Warn] BalancedRandomForest failed: {e}")

        # 15. Attach LLM / RAG-Tab ablation records for this dataset if present
        if d_name in ablation_map:
            for ab_rec in ablation_map[d_name]:
                m_label = ab_rec.get("Method", "")
                if "LLM" in m_label or "RAG" in m_label:
                    all_records.append({
                        "Dataset": d_name,
                        "Method": m_label,
                        "Family": "Generative LLM / RAG",
                        "Total Train Rows": n_orig + int(ab_rec.get("Added Rows", 30)),
                        "Minority Train Rows": n_pos + int(ab_rec.get("Added Rows", 30)),
                        "Synthetic Rows Added": int(ab_rec.get("Added Rows", 30)),
                        "Precision": ab_rec.get("Precision", 0.0),
                        "Recall": ab_rec.get("Recall", 0.0),
                        "F1": ab_rec.get("F1", 0.0),
                        "Specificity": ab_rec.get("Specificity", 0.0),
                        "G-Mean": ab_rec.get("G-Mean", 0.0),
                        "TP": ab_rec.get("TP", 0),
                        "FP": ab_rec.get("FP", 0),
                        "TN": ab_rec.get("TN", 0),
                        "FN": ab_rec.get("FN", 0)
                    })

        completed_datasets.add(d_name)
        # Checkpoint save
        pd.DataFrame(all_records).to_csv(out_csv, index=False)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s. Saved checkpoint ({len(completed_datasets)}/{len(datasets)} datasets).")

    # Final DataFrame
    full_df = pd.DataFrame(all_records)
    full_df.to_csv(out_csv, index=False)
    print(f"\nSaved all-datasets comprehensive benchmark to: {out_csv}")
    print(f"Total evaluations recorded: {len(full_df)}")

    # Compute Macro Leaderboard across all datasets
    macro_records = []
    for method, grp in full_df.groupby("Method"):
        family = grp["Family"].iloc[0] if "Family" in grp.columns else "Other"
        macro_records.append({
            "Method": method,
            "Family": family,
            "Datasets Evaluated": len(grp),
            "Macro F1 (Mean)": grp["F1"].mean(),
            "Macro F1 (Std)": grp["F1"].std(),
            "Macro Precision": grp["Precision"].mean(),
            "Macro Recall": grp["Recall"].mean(),
            "Macro Specificity": grp["Specificity"].mean(),
            "Macro G-Mean": grp["G-Mean"].mean()
        })

    macro_df = pd.DataFrame(macro_records).sort_values("Macro F1 (Mean)", ascending=False)
    macro_df.to_csv(macro_csv, index=False)
    print(f"Saved global macro leaderboard across all datasets to: {macro_csv}")

    total_time = time.time() - total_start
    print(f"Benchmark finished in {total_time:.1f}s.")


if __name__ == "__main__":
    run_benchmark()
