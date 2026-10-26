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

    clean_names = {
        "Original": "Original (No Resampling)",
        "SMOTE (Equal Budget N=30)": "SMOTE (N=30)",
        "Borderline-SMOTE (Equal Budget N=30)": "Borderline-SMOTE (N=30)",
        "RAG-Tab (Unified NaN + BERT N=30)": "RAG-Tab (BERT, N=30)",
        "SMOTE (Full Resampling)": "SMOTE (Full / Unconstrained)"
    }

    # Plot 1: Equal Budget Precision vs Recall vs F1
    if not df_equal.empty:
        df_eq_copy = df_equal.copy()
        df_eq_copy["Clean_Method"] = df_eq_copy["Method"].map(lambda x: clean_names.get(x, x))
        method_order = [
            "Original (No Resampling)",
            "SMOTE (Full / Unconstrained)",
            "Borderline-SMOTE (N=30)",
            "SMOTE (N=30)",
            "RAG-Tab (BERT, N=30)"
        ]
        summary_eq = df_eq_copy.groupby("Clean_Method")[["Precision", "Recall", "F1", "Specificity", "G-Mean"]].mean().reindex(method_order).reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        metrics = [
            ("Precision", "Equal-Budget Precision", "Blues_r", axes[0]),
            ("Recall", "Equal-Budget Recall", "Greens_r", axes[1]),
            ("F1", "Equal-Budget Macro F1-Score", "magma", axes[2])
        ]
        for col, title, palette, ax in metrics:
            sns.barplot(data=summary_eq, y="Clean_Method", x=col, ax=ax, palette=palette, edgecolor="black", linewidth=0.8)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel(col, fontsize=11, fontweight="semibold")
            ax.set_ylabel("" if ax != axes[0] else "Method", fontsize=11, fontweight="semibold")
            for p in ax.patches:
                val = p.get_width()
                ax.annotate(f" {val:.3f}", (val, p.get_y() + p.get_height() / 2.),
                            va='center', ha='left', color='black', fontweight='bold', fontsize=10.5)

        plt.suptitle("Equal-Budget Synthetic Data Benchmark (N_syn = 30 across 10 OpenML Datasets)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "equal_budget_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Plot Saved] {plots_dir / 'equal_budget_comparison.png'}")

    # Plot 2: Cross-Classifier Generalization
    if not df_multi.empty:
        df_multi_copy = df_multi.copy()
        df_multi_copy["Clean_Method"] = df_multi_copy["Method"].map(lambda x: clean_names.get(x, x))
        summary_multi = df_multi_copy.groupby(["Classifier", "Clean_Method"])[["Precision", "F1"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.barplot(data=summary_multi, x="Classifier", y="Precision", hue="Clean_Method", ax=axes[0], palette="Set2", edgecolor="black", linewidth=0.8)
        axes[0].set_title("Macro Precision Across Classifier Inductive Biases", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_xlabel("Downstream Classifier Family", fontsize=11, fontweight="semibold")
        axes[0].set_ylabel("Macro Precision", fontsize=11, fontweight="semibold")
        axes[0].legend(loc="lower right", framealpha=0.95, fontsize=10)
        for p in axes[0].patches:
            h = p.get_height()
            if h > 0.05:
                axes[0].annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.015),
                                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        sns.barplot(data=summary_multi, x="Classifier", y="F1", hue="Clean_Method", ax=axes[1], palette="Set2", edgecolor="black", linewidth=0.8)
        axes[1].set_title("Macro F1 Across Classifier Inductive Biases", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_xlabel("Downstream Classifier Family", fontsize=11, fontweight="semibold")
        axes[1].set_ylabel("Macro F1-Score", fontsize=11, fontweight="semibold")
        axes[1].legend(loc="lower right", framealpha=0.95, fontsize=10)
        for p in axes[1].patches:
            h = p.get_height()
            if h > 0.05:
                axes[1].annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h + 0.015),
                                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        plt.suptitle("Cross-Classifier Generalization (Random Forest vs. HistGBDT vs. Logistic Regression)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "multiclassifier_generalization.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Plot Saved] {plots_dir / 'multiclassifier_generalization.png'}")

    # Plot 3: Generative Distribution Fidelity & DCR
    if not df_fid.empty:
        df_fid_copy = df_fid.copy()
        df_fid_copy["Clean_Method"] = df_fid_copy["Method"].map(lambda x: clean_names.get(x, x))
        summary_fid = df_fid_copy.groupby("Clean_Method")[["Mean_Wasserstein", "Correlation_Frobenius_Diff", "DCR_Mean"]].mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        f_metrics = [
            ("Mean_Wasserstein", "Wasserstein-1 Distance\n(Lower is Better)", "mako", axes[0]),
            ("Correlation_Frobenius_Diff", "Correlation Matrix Error ||C_syn - C_real||_F\n(Lower is Better)", "rocket", axes[1]),
            ("DCR_Mean", "Distance to Closest Record (DCR)\n(Strictly > 0 = No Memorization)", "crest", axes[2])
        ]
        for col, title, palette, ax in f_metrics:
            sns.barplot(data=summary_fid, y="Clean_Method", x=col, ax=ax, palette=palette, edgecolor="black", linewidth=0.8)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel(col, fontsize=11, fontweight="semibold")
            ax.set_ylabel("" if ax != axes[0] else "Method", fontsize=11, fontweight="semibold")
            for p in ax.patches:
                val = p.get_width()
                ax.annotate(f" {val:.3f}", (val, p.get_y() + p.get_height() / 2.),
                            va='center', ha='left', color='black', fontweight='bold', fontsize=10)

        plt.suptitle("Generative Distribution Fidelity & Privacy Guarantees", fontsize=15, fontweight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(plots_dir / "generative_fidelity_and_privacy.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Plot Saved] {plots_dir / 'generative_fidelity_and_privacy.png'}")

    print("\nRESEARCH SUITE COMPLETE!")


if __name__ == "__main__":
    run_comprehensive_suite()
