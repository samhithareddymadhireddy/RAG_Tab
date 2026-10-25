import pandas as pd

df = pd.read_csv("results/tables/multiclassifier_generalization_benchmark.csv")
print("=== MULTI-CLASSIFIER BENCHMARK MACRO SUMMARY ===")
summary = df.groupby(["Classifier", "Method"])[["Precision", "Recall", "F1", "Specificity", "G-Mean"]].mean().reset_index()
print(summary.to_string(index=False))

print("\n=== OVERALL MACRO ACROSS ALL 3 CLASSIFIERS ===")
overall = df.groupby("Method")[["Precision", "Recall", "F1", "Specificity", "G-Mean"]].mean().reset_index()
print(overall.to_string(index=False))

print("\n=== HEAD-TO-HEAD WIN RATES (RAG-Tab vs SMOTE Equal Budget) ===")
methods = ["RAG-Tab (Unified NaN + BERT N=30)", "SMOTE (Equal Budget N=30)"]
sub = df[df["Method"].isin(methods)]
piv_f1 = sub.pivot_table(index=["Dataset", "Classifier"], columns="Method", values="F1")
piv_p = sub.pivot_table(index=["Dataset", "Classifier"], columns="Method", values="Precision")

f1_diff = piv_f1[methods[0]] - piv_f1[methods[1]]
p_diff = piv_p[methods[0]] - piv_p[methods[1]]

print(f"F1 Wins: {(f1_diff > 0.001).sum()}, Ties: {(f1_diff.abs() <= 0.001).sum()}, Losses: {(f1_diff < -0.001).sum()}")
print(f"Precision Wins: {(p_diff > 0.001).sum()}, Ties: {(p_diff.abs() <= 0.001).sum()}, Losses: {(p_diff < -0.001).sum()}")
