"""Multiclass Imbalanced Learning Benchmark for RAG-Tab.

Supports datasets with 3 or more classes (e.g. dermatology [6 classes], vehicle [4 classes], segment [7 classes]).
Evaluates:
- Original (No Resampling)
- SMOTE (Standardized Multiclass)
- Borderline-SMOTE (Multiclass)
- RAG-Tab (Multiclass Adaptive NaN + BERT)
Across multiple classifiers (Random Forest, HistGradientBoosting, Logistic Regression).
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
from src.retrieval.unified_retriever import MulticlassAdaptiveRetriever, UnifiedAdaptiveRetriever
from src.generation.schema import TabularDatasetSchema
from src.generation.prompt import build_generation_prompt
from src.generation.llm_client import GeminiTabularClient
from src.generation.parser import parse_and_validate_tabular_json


MULTICLASS_DATASETS = [
    "dermatology",
    "vehicle",
    "segment"
]


def run_rag_tab_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    target_col: str,
    dataset_name: str,
    llm_client: GeminiTabularClient,
    samples_per_minority: int = 15
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Runs class-conditioned RAG-Tab generation across all minority classes."""
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    schema = TabularDatasetSchema(
        df=train_df,
        target_col=target_col,
        dataset_description=f"{dataset_name} Multiclass"
    )

    class_counts = {str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()}
    maj_class = max(class_counts, key=class_counts.get)
    n_maj = class_counts[maj_class]
    minority_classes = [c for c in class_counts if c != maj_class]

    retriever = MulticlassAdaptiveRetriever(semantic_weight=0.35, use_bert=True)
    retriever.fit(X_train, y_train, dataset_name=dataset_name)

    all_synth_dfs = []

    for c in minority_classes:
        n_c = class_counts[c]
        target_budget = min(samples_per_minority, max(5, int((n_maj - n_c) * 0.4)))
        anchors = retriever.get_boundary_anchors(target_class=c, n_anchors=min(3, n_c))

        class_seeds = []
        for a_idx in anchors:
            try:
                ctx = retriever.retrieve_context(target_class=c, anchor_idx=a_idx)
                prompt = build_generation_prompt(
                    schema=schema,
                    mode="rag_tab",
                    context=ctx,
                    target_class=c,
                    n_samples=max(2, target_budget // max(1, len(anchors)))
                )
                resp_text = llm_client.generate(prompt)
                p_df, _, _ = parse_and_validate_tabular_json(
                    resp_text,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    target_class=c
                )
                if not p_df.empty:
                    class_seeds.append(p_df)
            except Exception as e:
                pass

        if class_seeds:
            df_c = pd.concat(class_seeds, ignore_index=True)
        else:
            df_c = X_train[y_train.astype(str) == str(c)].sample(min(target_budget, n_c), replace=True, random_state=42).copy()
            df_c[target_col] = c

        all_synth_dfs.append(df_c)

    if all_synth_dfs:
        df_synthetic_all = pd.concat(all_synth_dfs, ignore_index=True)
    else:
        df_synthetic_all = pd.DataFrame(columns=feature_cols + [target_col])

    X_aug = pd.concat([X_train, df_synthetic_all[feature_cols]], ignore_index=True)
    y_aug = pd.concat([y_train.astype(str), df_synthetic_all[target_col].astype(str)], ignore_index=True)

    return X_aug, y_aug, len(df_synthetic_all)


def run_multiclass_benchmark():
    print("=" * 80)
    print("STARTING MULTICLASS IMBALANCED BENCHMARK SUITE")
    print(f"Datasets ({len(MULTICLASS_DATASETS)}): {', '.join(MULTICLASS_DATASETS)}")
    print("=" * 80)

    llm_client = GeminiTabularClient(model_name="gemini-3.8-flash", temperature=0.7)
    records = []

    for d_idx, d_name in enumerate(MULTICLASS_DATASETS, 1):
        print(f"\n[{d_idx}/{len(MULTICLASS_DATASETS)}] Processing Multiclass Dataset: {d_name}...")
        dataset_key = f"{d_name}_multiclass"

        try:
            X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
                dataset_name=dataset_key,
                splits_dir="data/splits"
            )
        except Exception as e:
            print(f"  [Error loading {dataset_key}]: {e}")
            continue

        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
        class_counts = {str(k): int(v) for k, v in y_train.value_counts().items()}
        n_classes = len(class_counts)
        print(f"  Samples: Train={len(X_train)}, Test={len(X_test)} | Classes={n_classes} {class_counts}")

        # -------------------------------------------------------------
        # 1. Prepare Resampled Datasets
        # -------------------------------------------------------------
        datasets_map = {
            "Original": (X_train.copy(), y_train.copy(), 0)
        }

        # SMOTE (Standardized Multiclass)
        try:
            X_sm, y_sm, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, random_state=42)
            datasets_map["SMOTE (Multiclass)"] = (X_sm, y_sm, len(X_sm) - len(X_train))
        except Exception as e:
            print(f"  [Warning] SMOTE failed: {e}")

        # Borderline-SMOTE (Multiclass)
        try:
            X_bsm, y_bsm, _ = apply_resampling(X_train, y_train, method="borderline_smote", scale_features=True, random_state=42)
            datasets_map["Borderline-SMOTE (Multiclass)"] = (X_bsm, y_bsm, len(X_bsm) - len(X_train))
        except Exception as e:
            print(f"  [Warning] Borderline-SMOTE failed: {e}")

        # GAMO (Generative Adversarial Minority Oversampling)
        try:
            X_gamo, y_gamo, _ = apply_resampling(X_train, y_train, method="gamo", random_state=42)
            datasets_map["GAMO (Multiclass)"] = (X_gamo, y_gamo, len(X_gamo) - len(X_train))
        except Exception as e:
            print(f"  [Warning] GAMO failed: {e}")

        # RAG-Tab (Multiclass Adaptive NaN + BERT)
        try:
            X_rag, y_rag, n_added = run_rag_tab_multiclass(
                X_train=X_train,
                y_train=y_train,
                feature_cols=feature_cols,
                target_col=target_col,
                dataset_name=d_name,
                llm_client=llm_client,
                samples_per_minority=20
            )
            datasets_map["RAG-Tab (Multiclass NaN + BERT)"] = (X_rag, y_rag, n_added)
        except Exception as e:
            print(f"  [Warning] RAG-Tab Multiclass failed: {e}")

        # -------------------------------------------------------------
        # 2. Evaluate Classifiers
        # -------------------------------------------------------------
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        }

        for clf_name, clf in classifiers.items():
            for m_name, (X_tr, y_tr, n_added) in datasets_map.items():
                try:
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_test)
                    eval_res = evaluate_model(y_test, y_pred)

                    rec = {
                        "Dataset": d_name,
                        "Classes": n_classes,
                        "Classifier": clf_name,
                        "Method": m_name,
                        "Added Rows": n_added,
                        "Precision": eval_res["Precision"],
                        "Recall": eval_res["Recall"],
                        "F1": eval_res["F1"],
                        "F1_Weighted": eval_res.get("F1_Weighted", eval_res["F1"]),
                        "Specificity": eval_res["Specificity"],
                        "G-Mean": eval_res["G-Mean"]
                    }
                    records.append(rec)
                    print(f"    [{clf_name}] {m_name:32s} -> Macro F1: {eval_res['F1']:.4f}, Macro Prec: {eval_res['Precision']:.4f}, Added: {n_added}")
                except Exception as ex:
                    print(f"    [Error] {clf_name} with {m_name}: {ex}")

    # -------------------------------------------------------------
    # 3. Save CSV Table & Generate Plot
    # -------------------------------------------------------------
    tables_dir = Path("results/tables")
    plots_dir = Path("results/plots")
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(records)
    out_csv = tables_dir / "multiclass_benchmark.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[Saved Table] {out_csv}")

    # Plot Multiclass Comparison
    if not df_results.empty:
        summary = df_results.groupby(["Classifier", "Method"])[["Precision", "Recall", "F1", "Specificity", "G-Mean"]].mean().reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=summary, x="Classifier", y="Precision", hue="Method", ax=axes[0], palette="Set2")
        axes[0].set_title("Multiclass Macro Precision Across Classifiers", fontsize=13, fontweight="bold")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc="lower right")

        sns.barplot(data=summary, x="Classifier", y="F1", hue="Method", ax=axes[1], palette="Set2")
        axes[1].set_title("Multiclass Macro F1-Score Across Classifiers", fontsize=13, fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plot_path = plots_dir / "multiclass_benchmark.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Saved Plot] {plot_path}")

    print("\nMULTICLASS BENCHMARK COMPLETE!")


if __name__ == "__main__":
    run_multiclass_benchmark()
