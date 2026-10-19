"""Tabular dataset schema definitions and physical domain constraints."""
from typing import Dict, Any, List
import pandas as pd


class TabularDatasetSchema:
    """Extracts and manages feature types, statistical constraints, and schema definitions."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "Class",
        minority_class: str = "positive",
        dataset_description: str = "Page Blocks Document Classification"
    ):
        self.target_col = target_col
        self.minority_class = minority_class
        self.dataset_description = dataset_description
        self.feature_cols = [c for c in df.columns if c != target_col]

        # Extract empirical bounds and datatypes from training data only
        self.features_info: Dict[str, Dict[str, Any]] = {}
        for col in self.feature_cols:
            is_int = pd.api.types.is_integer_dtype(df[col]) or (df[col] % 1 == 0).all()
            self.features_info[col] = {
                "dtype": "integer" if is_int else "float",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Returns human-interpretable domain descriptions for page-blocks0 features."""
        return {
            "Height": "Height of the segmented block in pixels (integer >= 1).",
            "Lenght": "Length / width of the segmented block in pixels (integer >= 1).",
            "Area": "Total area of the block (Height * Lenght in pixels).",
            "Eccen": "Eccentricity of the block (Lenght / Height ratio).",
            "P_black": "Percentage of black pixels over the total block area [0.0, 1.0].",
            "P_and": "Percentage of black pixels after horizontal logical AND operation [0.0, 1.0].",
            "Mean_tr": "Mean number of black-to-white transitions per run in the block.",
            "Blackpix": "Total count of black pixels in the block.",
            "Blackand": "Total count of black pixels after AND operation.",
            "Wb_trans": "Total count of white-to-black transitions."
        }

    def to_prompt_schema(self) -> str:
        """Formats feature constraints into a clear markdown prompt schema."""
        lines = [f"Dataset: {self.dataset_description}", "Feature Specifications:"]
        desc_map = self.get_feature_descriptions()
        for col, info in self.features_info.items():
            desc = desc_map.get(col, "Numerical feature.")
            lines.append(
                f"- `{col}` ({info['dtype']}): {desc} "
                f"Valid domain range: [{info['min']:.2f}, {info['max']:.2f}], "
                f"training mean: {info['mean']:.2f}"
            )
        lines.append(f"- `{self.target_col}`: Must always be the minority class '{self.minority_class}'.")
        return "\n".join(lines)
