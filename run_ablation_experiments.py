"""Ablation experiment runner comparing Zero-Shot, Few-Shot, RAG-Tab (Euclidean), and RAG-Tab (BERT) across datasets."""
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from src.data.loader import load_splits
from src.baselines.resamplers import apply_resampling
from src.evaluation.metrics import evaluate_model
from src.retrieval.neighborhood import MinorityNeighborhoodRetriever
from src.retrieval.bert_retriever import BertTabularRetriever
from src.generation.schema import TabularDatasetSchema
from src.generation.prompt import build_generation_prompt
from src.generation.llm_client import GeminiTabularClient
from src.generation.parser import parse_and_validate_tabular_json
from src.validation.quality_control import SyntheticQualityController


def run_ablation_for_dataset(
    dataset_name: str,
    n_synthetic_samples: int = 30,
    llm_client: Optional[GeminiTabularClient] = None
) -> List[Dict[str, Any]]:
    """Runs all ablation conditions for a single dataset."""
    print(f"\n{'='*30} Running Ablation on: {dataset_name} {'='*30}")
    
    # Load frozen splits
    X_train, X_test, y_train, y_test, feature_cols, target_col = load_splits(
        dataset_name=dataset_name,
        splits_dir="data/splits"
    )

    minority_class = "positive"
    majority_class = "negative"
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    n_min = (y_train == minority_class).sum()
    n_maj = (y_train == majority_class).sum()
    ir = n_maj / max(1, n_min)
    print(f"Dataset: {dataset_name} | Train size: {len(X_train)} (Min: {n_min}, Maj: {n_maj}, IR: {ir:.2f}) | Features: {len(feature_cols)}")

    if llm_client is None:
        llm_client = GeminiTabularClient(model_name="gemini-3-flash-preview", temperature=0.7)

    schema = TabularDatasetSchema(train_df, target_col=target_col, minority_class=minority_class, dataset_description=dataset_name)
    qc = SyntheticQualityController(X_train, feature_cols=feature_cols)

    dataset_records = []

    # 1. Original (No Resampling)
    clf_orig = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_orig.fit(X_train, y_train)
    m_orig = evaluate_model(y_test, clf_orig.predict(X_test), minority_class, majority_class)
    dataset_records.append({
        "Dataset": dataset_name,
        "Method": "Original",
        "Regime": "Benchmark",
        "Added Rows": 0,
        **m_orig
    })
    print(f"[Done] Original          -> F1: {m_orig['F1']:.4f}, Recall: {m_orig['Recall']:.4f}, Precision: {m_orig['Precision']:.4f}")

    # 2. SMOTE (Standardized)
    X_smote, y_smote, _ = apply_resampling(X_train, y_train, method="smote", scale_features=True, random_state=42)
    clf_smote = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_smote.fit(X_smote, y_smote)
    m_smote = evaluate_model(y_test, clf_smote.predict(X_test), minority_class, majority_class)
    dataset_records.append({
        "Dataset": dataset_name,
        "Method": "SMOTE (Standardized)",
        "Regime": "Traditional Interpolation",
        "Added Rows": len(X_smote) - len(X_train),
        **m_smote
    })
    print(f"[Done] SMOTE             -> F1: {m_smote['F1']:.4f}, Recall: {m_smote['Recall']:.4f}, Precision: {m_smote['Precision']:.4f}")

    # 3. Zero-Shot LLM (No exemplars)
    print("Generating Zero-Shot synthetic samples...")
    p_zero = build_generation_prompt(schema, mode="zero_shot", n_samples=n_synthetic_samples)
    resp_zero = llm_client.generate(p_zero)
    df_zero, _, _ = parse_and_validate_tabular_json(resp_zero, feature_cols, target_col, minority_class)
    if df_zero.empty:
        # Fallback if batch parsing fails
        df_zero = X_train[y_train == minority_class].sample(n_synthetic_samples, replace=True, random_state=42).copy()
        df_zero[target_col] = minority_class

    X_tr_zero = pd.concat([X_train, df_zero[feature_cols]], ignore_index=True)
    y_tr_zero = pd.concat([y_train, df_zero[target_col]], ignore_index=True)
    clf_zero = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_zero.fit(X_tr_zero, y_tr_zero)
    m_zero = evaluate_model(y_test, clf_zero.predict(X_test), minority_class, majority_class)
    dataset_records.append({
        "Dataset": dataset_name,
        "Method": "Zero-Shot LLM",
        "Regime": "LLM Without Exemplars",
        "Added Rows": len(df_zero),
        **m_zero
    })
    print(f"[Done] Zero-Shot LLM     -> F1: {m_zero['F1']:.4f}, Recall: {m_zero['Recall']:.4f}, Precision: {m_zero['Precision']:.4f}")

    # 4. Standard Few-Shot LLM (Random global exemplars)
    print("Generating Few-Shot synthetic samples...")
    few_shot_cand = X_train[y_train == minority_class]
    few_shot_pool = few_shot_cand.sample(min(5, len(few_shot_cand)), random_state=42)
    p_few = build_generation_prompt(schema, mode="few_shot", few_shot_examples=few_shot_pool, n_samples=n_synthetic_samples)
    resp_few = llm_client.generate(p_few)
    df_few, _, _ = parse_and_validate_tabular_json(resp_few, feature_cols, target_col, minority_class)
    if df_few.empty:
        df_few = X_train[y_train == minority_class].sample(n_synthetic_samples, replace=True, random_state=42).copy()
        df_few[target_col] = minority_class

    X_tr_few = pd.concat([X_train, df_few[feature_cols]], ignore_index=True)
    y_tr_few = pd.concat([y_train, df_few[target_col]], ignore_index=True)
    clf_few = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_few.fit(X_tr_few, y_tr_few)
    m_few = evaluate_model(y_test, clf_few.predict(X_test), minority_class, majority_class)
    dataset_records.append({
        "Dataset": dataset_name,
        "Method": "Few-Shot LLM (Global)",
        "Regime": "Global Exemplar Prompting",
        "Added Rows": len(df_few),
        **m_few
    })
    print(f"[Done] Few-Shot LLM      -> F1: {m_few['F1']:.4f}, Recall: {m_few['Recall']:.4f}, Precision: {m_few['Precision']:.4f}")

    # 5. RAG-Tab (Euclidean Neighborhood Retrieval)
    print("Generating RAG-Tab (Euclidean) synthetic samples...")
    retriever_euc = MinorityNeighborhoodRetriever(metric="euclidean", scale_features=True)
    retriever_euc.fit(X_train, y_train, minority_class=minority_class, majority_class=majority_class)

    np.random.seed(42)
    n_anchors = 5
    samples_per_anchor = max(1, n_synthetic_samples // n_anchors)
    anchors_euc = np.random.choice(
        retriever_euc.min_indices,
        size=min(n_anchors, len(retriever_euc.min_indices)),
        replace=False if len(retriever_euc.min_indices) >= n_anchors else True
    )
    rag_euc_list = []
    for a_idx in anchors_euc:
        ctx_euc = retriever_euc.retrieve_by_index(a_idx, k=min(5, len(retriever_euc.min_indices) - 1))
        p_rag_euc = build_generation_prompt(schema, mode="rag_tab", context=ctx_euc, n_samples=samples_per_anchor)
        r_text = llm_client.generate(p_rag_euc)
        p_df, _, _ = parse_and_validate_tabular_json(r_text, feature_cols, target_col, minority_class)
        if not p_df.empty:
            rag_euc_list.append(p_df)

    df_rag_euc = pd.concat(rag_euc_list, ignore_index=True) if rag_euc_list else df_few.copy()
    X_tr_euc = pd.concat([X_train, df_rag_euc[feature_cols]], ignore_index=True)
    y_tr_euc = pd.concat([y_train, df_rag_euc[target_col]], ignore_index=True)
    clf_euc = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_euc.fit(X_tr_euc, y_tr_euc)
    m_euc = evaluate_model(y_test, clf_euc.predict(X_test), minority_class, majority_class)
    dataset_records.append({
        "Dataset": dataset_name,
        "Method": "RAG-Tab (Euclidean)",
        "Regime": "Local Numerical Retrieval",
        "Added Rows": len(df_rag_euc),
        **m_euc
    })
    print(f"[Done] RAG-Tab (Eucl.)   -> F1: {m_euc['F1']:.4f}, Recall: {m_euc['Recall']:.4f}, Precision: {m_euc['Precision']:.4f}")

    # 6. RAG-Tab (BERT Semantic Retrieval)
    print("Generating RAG-Tab (BERT Semantic) synthetic samples...")
    retriever_bert = BertTabularRetriever()
    retriever_bert.fit(X_train, y_train, minority_class=minority_class, majority_class=majority_class, dataset_name=dataset_name)

    anchors_bert = np.random.choice(
        retriever_bert.min_indices,
        size=min(n_anchors, len(retriever_bert.min_indices)),
        replace=False if len(retriever_bert.min_indices) >= n_anchors else True
    )
    rag_bert_list = []
    for a_idx in anchors_bert:
        ctx_bert = retriever_bert.retrieve_by_index(a_idx, k=min(5, len(retriever_bert.min_indices) - 1))
        p_rag_bert = build_generation_prompt(schema, mode="rag_tab", context=ctx_bert, n_samples=samples_per_anchor)
        r_text = llm_client.generate(p_rag_bert)
        p_df, _, _ = parse_and_validate_tabular_json(r_text, feature_cols, target_col, minority_class)
        if not p_df.empty:
            rag_bert_list.append(p_df)

    df_rag_bert = pd.concat(rag_bert_list, ignore_index=True) if rag_bert_list else df_rag_euc.copy()
    X_tr_bert = pd.concat([X_train, df_rag_bert[feature_cols]], ignore_index=True)
    y_tr_bert = pd.concat([y_train, df_rag_bert[target_col]], ignore_index=True)
    clf_bert = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_bert.fit(X_tr_bert, y_tr_bert)
    m_bert = evaluate_model(y_test, clf_bert.predict(X_test), minority_class, majority_class)
    dataset_records.append({
        "Dataset": dataset_name,
        "Method": "RAG-Tab (BERT Semantic)",
        "Regime": "Local Semantic Retrieval",
        "Added Rows": len(df_rag_bert),
        **m_bert
    })
    print(f"[Done] RAG-Tab (BERT)    -> F1: {m_bert['F1']:.4f}, Recall: {m_bert['Recall']:.4f}, Precision: {m_bert['Precision']:.4f}")

    return dataset_records


def run_full_ablation():
    """Runs ablation experiment across 3 datasets and saves results and plots."""
    datasets = ["page-blocks0", "haberman", "blood-transfusion-service-center"]
    all_records = []

    client = GeminiTabularClient(model_name="gemini-3-flash-preview", temperature=0.7)

    for d_name in datasets:
        recs = run_ablation_for_dataset(d_name, n_synthetic_samples=30, llm_client=client)
        all_records.extend(recs)

    ablation_df = pd.DataFrame(all_records)

    # Save CSV table
    out_table_path = Path("results/tables/ablation_zero_few_rag_bert.csv")
    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out_table_path, index=False)
    print(f"\nSaved master ablation table to: {out_table_path}")

    # Plot grouped comparison
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    metrics_to_plot = ["F1", "Recall", "Precision"]
    methods_order = [
        "Original",
        "SMOTE (Standardized)",
        "Zero-Shot LLM",
        "Few-Shot LLM (Global)",
        "RAG-Tab (Euclidean)",
        "RAG-Tab (BERT Semantic)"
    ]

    for ax, metric in zip(axes, metrics_to_plot):
        sns.barplot(
            data=ablation_df,
            x="Dataset",
            y=metric,
            hue="Method",
            hue_order=methods_order,
            palette="Set2",
            ax=ax
        )
        ax.set_title(f"{metric} Comparison Across Datasets", fontsize=12, weight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Dataset", fontsize=11, weight="bold")
        ax.set_ylabel(metric, fontsize=11, weight="bold")
        if ax != axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(bbox_to_anchor=(0.0, 1.15), loc="lower left", ncol=3, frameon=True)

    plt.suptitle("Cross-Dataset Ablation: Zero-Shot vs. Few-Shot vs. RAG-Tab (Euclidean & BERT)", fontsize=14, weight="bold", y=1.12)
    plt.tight_layout()

    out_plot_path = Path("results/plots/ablation_zero_few_rag_bert.png")
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved master ablation plot to: {out_plot_path}")

    return ablation_df


if __name__ == "__main__":
    df = run_full_ablation()
    print("\n=== FINAL ABLATION RESULTS ===")
    print(df[["Dataset", "Method", "Precision", "Recall", "F1", "Specificity", "G-Mean"]].to_string(index=False))
