import sys
from pathlib import Path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TABLES_DIR / "all_methods_comparison_page_blocks0.csv")

# Set aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(18, 7))

plot_df = df.melt(
    id_vars=["Method"],
    value_vars=["Precision", "Recall", "F1", "Specificity", "G-Mean"],
    var_name="Metric",
    value_name="Score"
)

palette = {
    "Precision": "#2980b9",
    "Recall": "#e74c3c",
    "F1": "#27ae60",
    "Specificity": "#f39c12",
    "G-Mean": "#8e44ad"
}

sns.barplot(data=plot_df, x="Method", y="Score", hue="Metric", palette=palette, ax=ax)
ax.set_ylim(0.55, 1.02)
ax.set_title("Comprehensive Benchmark on Binary page-blocks0 (Random Forest Downstream)", fontsize=15, weight="bold", pad=15)
ax.set_xlabel("Imbalance Handling Method / Resampler", fontsize=12, weight="bold")
ax.set_ylabel("Metric Score", fontsize=12, weight="bold")
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, title="Evaluation Metric")
plt.tight_layout()

plot_path = PLOTS_DIR / "final_methods_comparison_page_blocks0.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Updated chart saved to: {plot_path}")
