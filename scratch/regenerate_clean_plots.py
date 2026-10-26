import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def regenerate_clean_plots():
    plots_dir = Path("results/plots")
    tables_dir = Path("results/tables")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set publication aesthetic theme
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.autolayout': False
    })

    # Method mapping to clean, professional names
    clean_names = {
        "Original": "Original (No Resampling)",
        "SMOTE (Equal Budget N=30)": "SMOTE (N=30)",
        "Borderline-SMOTE (Equal Budget N=30)": "Borderline-SMOTE (N=30)",
        "RAG-Tab (Unified NaN + BERT N=30)": "RAG-Tab (BERT, N=30)",
        "SMOTE (Full Resampling)": "SMOTE (Full / Unconstrained)",
        "SMOTE (Multiclass)": "SMOTE (Multiclass)",
        "Borderline-SMOTE (Multiclass)": "Borderline-SMOTE (Multiclass)",
        "RAG-Tab (Multiclass NaN + BERT)": "RAG-Tab (BERT Multiclass)"
    }

    # -------------------------------------------------------------------------
    # 1. Clean Equal Budget Comparison Plot (Horizontal Bars for Perfect Legibility)
    # -------------------------------------------------------------------------
    df_equal_path = tables_dir / "equal_budget_benchmark.csv"
    if df_equal_path.exists():
        df_equal = pd.read_csv(df_equal_path)
        df_equal["Clean_Method"] = df_equal["Method"].map(lambda x: clean_names.get(x, x))
        
        # Order methods logically
        method_order = [
            "Original (No Resampling)",
            "SMOTE (Full / Unconstrained)",
            "Borderline-SMOTE (N=30)",
            "SMOTE (N=30)",
            "RAG-Tab (BERT, N=30)"
        ]
        
        summary_eq = df_equal.groupby("Clean_Method")[["Precision", "Recall", "F1", "Specificity", "G-Mean"]].mean().reindex(method_order).reset_index()

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
            
            # Annotate bar values cleanly at the tip of each bar
            for p in ax.patches:
                val = p.get_width()
                ax.annotate(f" {val:.3f}", (val, p.get_y() + p.get_height() / 2.),
                            va='center', ha='left', color='black', fontweight='bold', fontsize=10.5)

        # Highlight RAG-Tab row
        plt.suptitle("Equal-Budget Synthetic Data Benchmark (N_syn = 30 across 10 OpenML Datasets)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "equal_budget_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Regenerated] {plots_dir / 'equal_budget_comparison.png'}")

    # -------------------------------------------------------------------------
    # 2. Clean Multiclassifier Generalization Plot
    # -------------------------------------------------------------------------
    df_multi_path = tables_dir / "multiclassifier_generalization_benchmark.csv"
    if df_multi_path.exists():
        df_multi = pd.read_csv(df_multi_path)
        df_multi["Clean_Method"] = df_multi["Method"].map(lambda x: clean_names.get(x, x))
        summary_multi = df_multi.groupby(["Classifier", "Clean_Method"])[["Precision", "F1"]].mean().reset_index()

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
                                 ha='center', va='bottom', fontsize=8.5, rotation=0, fontweight='bold')

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
                                 ha='center', va='bottom', fontsize=8.5, rotation=0, fontweight='bold')

        plt.suptitle("Cross-Classifier Generalization (Random Forest vs. HistGBDT vs. Logistic Regression)", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "multiclassifier_generalization.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Regenerated] {plots_dir / 'multiclassifier_generalization.png'}")

    # -------------------------------------------------------------------------
    # 3. Clean Generative Fidelity & Privacy Plot
    # -------------------------------------------------------------------------
    df_fid_path = tables_dir / "generative_fidelity_privacy_benchmark.csv"
    if df_fid_path.exists():
        df_fid = pd.read_csv(df_fid_path)
        df_fid["Clean_Method"] = df_fid["Method"].map(lambda x: clean_names.get(x, x))
        summary_fid = df_fid.groupby("Clean_Method")[["Mean_Wasserstein", "Correlation_Frobenius_Diff", "DCR_Mean"]].mean().reset_index()

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
        print(f"[Regenerated] {plots_dir / 'generative_fidelity_and_privacy.png'}")

    # -------------------------------------------------------------------------
    # 4. Clean Multiclass Benchmark Plot
    # -------------------------------------------------------------------------
    df_multi_bench_path = tables_dir / "multiclass_benchmark.csv"
    if df_multi_bench_path.exists():
        df_multi_bench = pd.read_csv(df_multi_bench_path)
        df_multi_bench["Clean_Method"] = df_multi_bench["Method"].map(lambda x: clean_names.get(x, x))
        summary_mb = df_multi_bench.groupby(["Dataset", "Classifier", "Clean_Method"])[["Precision", "F1"]].mean().reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(17, 6))

        # Focus on vehicle & dermatology
        sns.barplot(data=df_multi_bench[df_multi_bench["Dataset"] == "vehicle"],
                    x="Classifier", y="F1", hue="Clean_Method", ax=axes[0], palette="Set2", edgecolor="black", linewidth=0.8)
        axes[0].set_title("Vehicle Silhouettes (4 Classes) - Macro F1", fontsize=13, fontweight="bold", pad=12)
        axes[0].set_ylim(0.65, 0.88)
        axes[0].set_xlabel("Classifier", fontsize=11, fontweight="semibold")
        axes[0].set_ylabel("Macro F1-Score", fontsize=11, fontweight="semibold")
        axes[0].legend(loc="lower right", framealpha=0.95, fontsize=10)
        for p in axes[0].patches:
            h = p.get_height()
            if h > 0.1:
                axes[0].annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.005),
                                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        sns.barplot(data=df_multi_bench[df_multi_bench["Dataset"] == "dermatology"],
                    x="Classifier", y="F1", hue="Clean_Method", ax=axes[1], palette="Set2", edgecolor="black", linewidth=0.8)
        axes[1].set_title("Dermatology (6 Classes) - Macro F1", fontsize=13, fontweight="bold", pad=12)
        axes[1].set_ylim(0.92, 1.03)
        axes[1].set_xlabel("Classifier", fontsize=11, fontweight="semibold")
        axes[1].set_ylabel("Macro F1-Score", fontsize=11, fontweight="semibold")
        axes[1].legend(loc="lower right", framealpha=0.95, fontsize=10)
        for p in axes[1].patches:
            h = p.get_height()
            if h > 0.1:
                axes[1].annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.002),
                                 ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        plt.suptitle("Multiclass Imbalanced Benchmark: RAG-Tab vs. SMOTE Baselines", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "multiclass_benchmark.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Regenerated] {plots_dir / 'multiclass_benchmark.png'}")

if __name__ == "__main__":
    regenerate_clean_plots()
