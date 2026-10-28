import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

macro_csv = PROJECT_ROOT / "results" / "tables" / "all_datasets_macro_leaderboard.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(macro_csv)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(16, 8))

plot_df = df.melt(
    id_vars=["Method", "Family"],
    value_vars=["Macro F1 (Mean)", "Macro Precision", "Macro Recall", "Macro Specificity", "Macro G-Mean"],
    var_name="Metric",
    value_name="Score"
)
plot_df["Metric"] = plot_df["Metric"].str.replace("Macro ", "").str.replace(" (Mean)", "")

palette = {
    "F1": "#27ae60",
    "Precision": "#2980b9",
    "Recall": "#e74c3c",
    "Specificity": "#f39c12",
    "G-Mean": "#8e44ad"
}

sns.barplot(data=plot_df, x="Method", y="Score", hue="Metric", palette=palette, ax=ax)
ax.set_ylim(0.5, 1.0)
ax.set_title("Cross-Dataset Macro Benchmark Across All 43 Imbalanced Benchmark Datasets", fontsize=15, weight="bold", pad=15)
ax.set_xlabel("Method", fontsize=12, weight="bold")
ax.set_ylabel("Macro Average Score across 43 Datasets", fontsize=12, weight="bold")
plt.xticks(rotation=40, ha="right", fontsize=10)
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, title="Evaluation Metric")
plt.tight_layout()

out_png = PLOTS_DIR / "all_datasets_macro_comparison.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved cross-dataset comparison chart to: {out_png}")
