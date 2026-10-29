import pandas as pd
import numpy as np

df = pd.read_csv("results/seed_check/seed_noise_benchmark.csv")
datasets = df['Dataset'].unique()

print(f"{'Dataset':<32} | {'Condition':<15} | {'Precision (Mean ± Std)':<24} | {'F1 (Mean ± Std)':<24} | {'Added Rows':<10}")
print("-" * 115)

for ds in datasets:
    ds_df = df[df['Dataset'] == ds]
    for cond in ['Zero-Shot', 'Few-Shot', 'RAG-Euclidean', 'RAG-BERT']:
        sub = ds_df[ds_df['Condition'] == cond]
        p_mean = sub['Precision'].mean()
        p_std = sub['Precision'].std()
        f1_mean = sub['F1'].mean()
        f1_std = sub['F1'].std()
        rows_mean = sub['Added_Rows'].mean()
        print(f"{ds:<32} | {cond:<15} | {p_mean:.4f} ± {p_std:.4f}          | {f1_mean:.4f} ± {f1_std:.4f}     | {rows_mean:<10.1f}")
    print("-" * 115)

print("\n--- SUMMARY ACROSS DATASETS (MACRO AVERAGE) ---")
agg_per_ds = df.groupby(['Dataset', 'Condition'])[['Precision', 'F1', 'Added_Rows']].mean().reset_index()
for cond in ['Zero-Shot', 'Few-Shot', 'RAG-Euclidean', 'RAG-BERT']:
    sub = agg_per_ds[agg_per_ds['Condition'] == cond]
    std_seed = df[df['Condition'] == cond].groupby('Dataset')[['Precision', 'F1']].std().mean()
    print(f"{cond:<15}: Macro Prec = {sub['Precision'].mean():.4f} ± {sub['Precision'].std():.4f} | Macro F1 = {sub['F1'].mean():.4f} ± {sub['F1'].std():.4f} | Within-Dataset Seed Std: Prec = {std_seed['Precision']:.4f}, F1 = {std_seed['F1']:.4f}")
