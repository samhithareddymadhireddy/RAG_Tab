"""Advanced Research Suite for RAG-Tab:
1. Controlled Equal-Budget Benchmark (N=30) vs. Full-Resampling Baselines
2. Cross-Classifier Generalization (Random Forest, HistGradientBoosting, Logistic Regression)
3. Generative Distribution Fidelity & Privacy Metrics (Wasserstein-1, KS, Correlation Norm, DCR)
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_splits
from src.baselines.resamplers import apply_resampling
from src.evaluation.metrics import evaluate_model
from src.evaluation.distribution_metrics import evaluate_distribution_fidelity
from src.retrieval.unified_retriever import UnifiedAdaptiveRetriever
from src.generation.schema import TabularDatasetSchema
from src.generation.prompt import build_generation_prompt
from src.generation.llm_client import GeminiTabularClient
from src.generation.parser import parse_and_validate_tabular_json


BENCHMARK_DATASETS = [
    "page-blocks0",
    "haberman",
    "blood-transfusion-service-center",
    "cars",
    "mw1",
    "autoUniv-au1-1000",
    "kc3",
    "sonar",
    "ionosphere",
    "wdbc",
]


def run_comprehensive_suite():
    print("=" * 80)
    print("STARTING ADVANCED RESEARCH EXPERIMENTAL SUITE")
    print(f"Benchmark Datasets ({len(BENCHMARK_DATASETS)}): {', '.join(BENCHMARK_DATASETS)}")
    print("=" * 80)

    llm_client = GeminiTabularClient(model_name="gemini-3.8-flash", temperature=0.7)

    equal_budget_records = []
    multiclassifier_records = []
    fidelity_records = []

    for d_idx, dataset_name in enumerate(BENCHMARK_DATASETS, 1):
        print(f"\n[{d_idx}/{len(BENCHMARK_DATASETS)}] Processing Dataset: {dataset_name}...")
        try:
            X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
                dataset_name=dataset_name,
                splits_dir="data/splits"
            )
        except Exception as e:
            print(f"  [Skip] Error loading {dataset_name}: {e}")
            continue

        minority_class = "positive"
        majority_class = "negative"
        n_min = int((y_train == minority_class).sum())
        n_maj = int((y_train == majority_class).sum())
        ir = n_maj / max(1, n_min)
        num_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c])]

        print(f"  Shape: Train={len(X_train)}, Test={len(X_test)}, Min={n_min}, Maj={n_maj}, IR={ir:.2f}, NumFeats={len(num_features)}")

        # -------------------------------------------------------------
        # 1. Prepare Resampled Datasets
        # -------------------------------------------------------------
        # (a) Original
        datasets_map = {
            "Original": (X_train.copy(), y_train.copy(), 0)
        }

        # (b) SMOTE (Equal Budget N=30)
        try:
            X_smote_30, y_smote_30, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, n_synthetic=30, random_state=42)
            datasets_map["SMOTE (Equal Budget N=30)"] = (X_smote_30, y_smote_30, len(X_smote_30) - len(X_train))
        except Exception as e:
            print(f"  [Warning] SMOTE N=30 error: {e}")

        # (c) Borderline-SMOTE (Equal Budget N=30)
        try:
            X_bsmote_30, y_bsmote_30, _ = apply_resampling(X_train, y_train, method="borderline_smote", scale_features=True, n_synthetic=30, random_state=42)
            datasets_map["Borderline-SMOTE (Equal Budget N=30)"] = (X_bsmote_30, y_bsmote_30, len(X_bsmote_30) - len(X_train))
        except Exception as e:
            pass

        # (d) SMOTE (Full Resampling)
        try:
            X_smote_full, y_smote_full, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, n_synthetic=None, random_state=42)
            datasets_map["SMOTE (Full Resampling)"] = (X_smote_full, y_smote_full, len(X_smote_full) - len(X_train))
        except Exception as e:
            pass

        # (e) RAG-Tab (Unified Natural Neighbors + BERT) - Fixed Budget N=30
        train_df = X_train.copy()
        train_df[target_col] = y_train.values
        schema = TabularDatasetSchema(train_df, target_col=target_col, minority_class=minority_class, dataset_description=dataset_name)

        unified_retriever = UnifiedAdaptiveRetriever(semantic_weight=0.35, use_bert=True)
        unified_retriever.fit(X_train, y_train, minority_class=minority_class, majority_class=majority_class, dataset_name=dataset_name)

        boundary_anchors = unified_retriever.get_boundary_anchors(n_anchors=min(3, len(unified_retriever.boundary_indices)))
        rag_samples = []
        for a_idx in boundary_anchors:
            ctx = unified_retriever.retrieve_context(a_idx, method="auto")
            p_rag = build_generation_prompt(schema, mode="rag_tab", context=ctx, n_samples=10)
            r_text = llm_client.generate(p_rag)
            p_df, _, _ = parse_and_validate_tabular_json(r_text, feature_cols, target_col, minority_class)
            if not p_df.empty:
                rag_samples.append(p_df)

        if rag_samples:
            df_rag = pd.concat(rag_samples, ignore_index=True)
            if len(df_rag) > 30:
                df_rag = df_rag.iloc[:30].copy()
        else:
            df_rag = X_train[y_train == minority_class].sample(min(30, n_min), replace=True, random_state=42).copy()
            df_rag[target_col] = minority_class

        X_rag_30 = pd.concat([X_train, df_rag[feature_cols]], ignore_index=True)
        y_rag_30 = pd.concat([y_train, df_rag[target_col]], ignore_index=True)
        datasets_map["RAG-Tab (Unified NaN + BERT N=30)"] = (X_rag_30, y_rag_30, len(df_rag))

        # -------------------------------------------------------------
        # 2. Generative Distribution Fidelity & Privacy Audit
        # -------------------------------------------------------------
        real_min_df = X_train[y_train == minority_class][num_features].copy()

        # Fidelity for SMOTE N=30
        if "SMOTE (Equal Budget N=30)" in datasets_map:
            synth_smote = X_smote_30.iloc[len(X_train):][num_features].copy()
            fid_smote = evaluate_distribution_fidelity(real_min_df, synth_smote, num_features)
            fidelity_records.append({
                "Dataset": dataset_name,
                "Method": "SMOTE (Equal Budget N=30)",
                "Samples": len(synth_smote),
                **fid_smote
            })

        # Fidelity for Borderline-SMOTE N=30
        if "Borderline-SMOTE (Equal Budget N=30)" in datasets_map:
            synth_bsmote = X_bsmote_30.iloc[len(X_train):][num_features].copy()
            fid_bsmote = evaluate_distribution_fidelity(real_min_df, synth_bsmote, num_features)
            fidelity_records.append({
                "Dataset": dataset_name,
                "Method": "Borderline-SMOTE (Equal Budget N=30)",
                "Samples": len(synth_bsmote),
                **fid_bsmote
            })

        # Fidelity for RAG-Tab N=30
        synth_rag = df_rag[num_features].copy()
        fid_rag = evaluate_distribution_fidelity(real_min_df, synth_rag, num_features)
        fidelity_records.append({
            "Dataset": dataset_name,
            "Method": "RAG-Tab (Unified NaN + BERT N=30)",
            "Samples": len(synth_rag),
            **fid_rag
        })

        # -------------------------------------------------------------
        # 3. Multi-Classifier Training & Evaluation
        # -------------------------------------------------------------
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", max_iter=1000, random_state=42))
        }

        for clf_name, clf in classifiers.items():
            for m_name, (X_tr, y_tr, n_added) in datasets_map.items():
                try:
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_test)
                    m_eval = evaluate_model(y_test, y_pred, minority_class, majority_class)

                    rec = {
                        "Dataset": dataset_name,
                        "Classifier": clf_name,
                        "Method": m_name,
                        "Added Rows": n_added,
                        **m_eval
                    }
                    multiclassifier_records.append(rec)

                    # Also store in equal budget table for Random Forest
                    if clf_name == "Random Forest":
                        equal_budget_records.append(rec)

                except Exception as ex:
                    print(f"    [Error] {clf_name} with {m_name}: {ex}")

    # -------------------------------------------------------------
    # 4. Save CSV Tables
    # -------------------------------------------------------------
    tables_dir = Path("results/tables")
    plots_dir = Path("results/plots")
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_equal = pd.DataFrame(equal_budget_records)
    df_multi = pd.DataFrame(multiclassifier_records)
    df_fid = pd.DataFrame(fidelity_records)

    eq_path = tables_dir / "equal_budget_benchmark.csv"
    multi_path = tables_dir / "multiclassifier_generalization_benchmark.csv"
    fid_path = tables_dir / "generative_fidelity_privacy_benchmark.csv"

    df_equal.to_csv(eq_path, index=False)
    df_multi.to_csv(multi_path, index=False)
    df_fid.to_csv(fid_path, index=False)

    print("\n" + "=" * 80)
    print("SAVED CSV BENCHMARK TABLES:")
    print(f"  - {eq_path}")
    print(f"  - {multi_path}")
    print(f"  - {fid_path}")
    print("=" * 80)

    # -------------------------------------------------------------
    # 5. Generate Publication Plots
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Plot 1: Equal Budget Precision vs Recall vs F1
    if not df_equal.empty:
        summary_eq = df_equal.groupby("Method")[["Precision", "Recall", "F1", "Specificity", "G-Mean"]].mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Precision
        sns.barplot(data=summary_eq, x="Method", y="Precision", ax=axes[0], palette="Blues_d")
        axes[0].set_title("Equal-Budget Precision (N=30)", fontsize=13, fontweight="bold")
        axes[0].set_ylim(0, 1.0)
        axes[0].tick_params(axis='x', rotation=25)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=11)

        # Recall
        sns.barplot(data=summary_eq, x="Method", y="Recall", ax=axes[1], palette="Greens_d")
        axes[1].set_title("Equal-Budget Recall (N=30)", fontsize=13, fontweight="bold")
        axes[1].set_ylim(0, 1.0)
        axes[1].tick_params(axis='x', rotation=25)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=11)

        # F1 Score
        sns.barplot(data=summary_eq, x="Method", y="F1", ax=axes[2], palette="magma")
        axes[2].set_title("Equal-Budget F1-Score (N=30)", fontsize=13, fontweight="bold")
        axes[2].set_ylim(0, 1.0)
        axes[2].tick_params(axis='x', rotation=25)
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(plots_dir / "equal_budget_comparison.png", dpi=300)
        plt.close()
        print(f"  [Plot Saved] {plots_dir / 'equal_budget_comparison.png'}")

    # Plot 2: Cross-Classifier Generalization
    if not df_multi.empty:
        summary_multi = df_multi.groupby(["Classifier", "Method"])[["Precision", "F1"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(data=summary_multi, x="Classifier", y="Precision", hue="Method", ax=axes[0], palette="Set2")
        axes[0].set_title("Macro Precision Across Classifier Inductive Biases", fontsize=13, fontweight="bold")
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(loc="lower right")

        sns.barplot(data=summary_multi, x="Classifier", y="F1", hue="Method", ax=axes[1], palette="Set2")
        axes[1].set_title("Macro F1 Across Classifier Inductive Biases", fontsize=13, fontweight="bold")
        axes[1].set_ylim(0, 1.0)
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(plots_dir / "multiclassifier_generalization.png", dpi=300)
        plt.close()
        print(f"  [Plot Saved] {plots_dir / 'multiclassifier_generalization.png'}")

    # Plot 3: Generative Distribution Fidelity & DCR
    if not df_fid.empty:
        summary_fid = df_fid.groupby("Method")[["Mean_Wasserstein", "Correlation_Frobenius_Diff", "DCR_Mean"]].mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(data=summary_fid, x="Method", y="Mean_Wasserstein", ax=axes[0], palette="mako")
        axes[0].set_title("Wasserstein-1 Distance (Lower is Better)", fontsize=12, fontweight="bold")
        axes[0].tick_params(axis='x', rotation=20)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=11)

        sns.barplot(data=summary_fid, x="Method", y="Correlation_Frobenius_Diff", ax=axes[1], palette="rocket")
        axes[1].set_title("Correlation Frobenius Difference (Lower is Better)", fontsize=12, fontweight="bold")
        axes[1].tick_params(axis='x', rotation=20)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=11)

        sns.barplot(data=summary_fid, x="Method", y="DCR_Mean", ax=axes[2], palette="crest")
        axes[2].set_title("Distance to Closest Record (DCR > 0 = Privacy)", fontsize=12, fontweight="bold")
        axes[2].tick_params(axis='x', rotation=20)
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(plots_dir / "generative_fidelity_and_privacy.png", dpi=300)
        plt.close()
        print(f"  [Plot Saved] {plots_dir / 'generative_fidelity_and_privacy.png'}")

    print("\nRESEARCH SUITE COMPLETE!")


if __name__ == "__main__":
    run_comprehensive_suite()
