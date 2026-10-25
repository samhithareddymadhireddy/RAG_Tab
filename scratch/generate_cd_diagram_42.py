"""Computes Friedman test and generates Critical Difference Diagrams across all 42 datasets."""
import sys
sys.path.insert(0, ".")
import csv
from collections import defaultdict
import pandas as pd
from src.evaluation.statistical_tests import compute_friedman_and_cd, plot_critical_difference_diagram

csv_path = "results/tables/ablation_zero_few_rag_bert.csv"

# Load data
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

prec_data = defaultdict(dict)
f1_data = defaultdict(dict)

for r in rows:
    d = r["Dataset"]
    m = r["Method"]
    # Clean method names for display
    clean_name = m.replace(" (Standardized)", "").replace(" (Global)", "").replace(" (BERT Semantic)", " (BERT)").replace(" (Euclidean)", " (Euclid)")
    prec_data[d][clean_name] = float(r["Precision"])
    f1_data[d][clean_name] = float(r["F1"])

df_prec = pd.DataFrame.from_dict(prec_data, orient="index").dropna()
df_f1 = pd.DataFrame.from_dict(f1_data, orient="index").dropna()

print(f"Loaded {len(df_prec)} complete datasets for Precision.")
print(f"Loaded {len(df_f1)} complete datasets for F1.")

# 1. Precision Critical Difference
ranks_p, cd_p, stat_p, pval_p = compute_friedman_and_cd(df_prec, higher_is_better=True)
print("\n--- PRECISION FRIEDMAN TEST ---")
print(f"Chi-square statistic: {stat_p:.4f}, p-value: {pval_p:.5e}")
print(f"Nemenyi CD (alpha=0.05): {cd_p:.4f}")
print("Mean Ranks (Lower is Better, 1 = Highest Precision):")
print(ranks_p)

plot_critical_difference_diagram(
    ranks_p,
    cd_p,
    title=f"Critical Difference Diagram: Precision (42 Datasets, CD={cd_p:.2f}, p={pval_p:.3e})",
    output_path="results/plots/critical_difference_precision.png"
)
print("Saved results/plots/critical_difference_precision.png")

# 2. F1 Critical Difference
ranks_f, cd_f, stat_f, pval_f = compute_friedman_and_cd(df_f1, higher_is_better=True)
print("\n--- F1-SCORE FRIEDMAN TEST ---")
print(f"Chi-square statistic: {stat_f:.4f}, p-value: {pval_f:.5e}")
print(f"Nemenyi CD (alpha=0.05): {cd_f:.4f}")
print("Mean Ranks (Lower is Better, 1 = Highest F1):")
print(ranks_f)

plot_critical_difference_diagram(
    ranks_f,
    cd_f,
    title=f"Critical Difference Diagram: F1-Score (42 Datasets, CD={cd_f:.2f}, p={pval_f:.3e})",
    output_path="results/plots/critical_difference_f1.png"
)
print("Saved results/plots/critical_difference_f1.png")
