"""Prompt engineering module inspired by TMLR synthetic tabular generation research.

Supports 3 generation regimes:
1. Zero-Shot: Prompt contains executive domain report, variable definitions, and JSON schema (NO row exemplars).
2. Few-Shot: Prompt contains domain report, variable definitions, and random global minority exemplars.
3. RAG-Tab: Prompt contains domain report, variable definitions, minority anchor point, and local retrieved minority neighbors.
"""
from typing import Optional, List, Dict, Any
import pandas as pd
from .schema import TabularDatasetSchema
from ..retrieval.neighborhood import NeighborhoodContext


def build_data_description_prompt(
    dataset_name: str,
    sample_csv_text: str
) -> str:
    """Prompt 1: Generates high-level executive report on dataset domain and trends."""
    return f"""[Data description prompt]:
You're an expert data analyst at a large company. Your boss has asked you to analyze a dataset and provide a report on your findings.

You are provided with a sample of the dataset in CSV format with a header for dataset '{dataset_name}':
```csv
{sample_csv_text}
```

You'll need to write a report that provides insights into the data. Focus on the key points that will help your boss make informed decisions. Put a special emphasis on the relationships between the different variables in the dataset, the trends that you observe.

Make sure to try to extract the general meaning of the data, and not just the raw numbers. Your boss is looking for a high-level summary of the data, not a detailed analysis.

Your report should be clear, concise, and easy to understand. It should be written in plain English, and should not contain any technical jargon.
""".strip()


def build_variable_description_prompt(
    dataset_name: str,
    feature_names: List[str]
) -> str:
    """Prompt 2: Extracts variable names, descriptions, and possible distributions."""
    feat_list = ", ".join([f"'{f}'" for f in feature_names])
    return f"""[Variable description prompt]:
You're an expert data analyst at a large company. Your boss has asked you to analyze a dataset and provide a report on your findings.

The dataset '{dataset_name}' contains the following variables:
[{feat_list}]

Your task is to extract the names, descriptions and possible distributions of the variables in the dataset. Please note that categorical and string variables use the "none" distribution.

You should provide your answer as a machine-readable JSON object following this format:
{{
  "variables": [
    {{"name": "feature_name", "description": "plain english meaning", "type": "float|integer|categorical", "distribution": "normal|uniform|skewed|none"}}
  ]
}}
""".strip()


def build_generation_prompt(
    schema: TabularDatasetSchema,
    mode: str = "rag_tab",
    context: Optional[NeighborhoodContext] = None,
    few_shot_examples: Optional[pd.DataFrame] = None,
    domain_report: Optional[str] = None,
    n_samples: int = 1
) -> str:
    """Prompt 3: Constrained data generation prompt supporting Zero-Shot, Few-Shot, and RAG-Tab.

    Parameters
    ----------
    schema : TabularDatasetSchema
        Feature schema and physical domain bounds.
    mode : str
        One of 'zero_shot', 'few_shot', 'rag_tab'.
    context : NeighborhoodContext, optional
        Retrieved local neighborhood (required for 'rag_tab').
    few_shot_examples : pd.DataFrame, optional
        Global training minority instances (required for 'few_shot').
    domain_report : str, optional
        High-level domain context report (from Stage 1).
    n_samples : int, default=1
        Target number of synthetic minority samples ($N-OUT).
    """
    mode = mode.lower()
    schema_text = schema.to_prompt_schema()

    report_section = ""
    if domain_report:
        report_section = f"\n=== DOMAIN CONTEXT REPORT ===\n{domain_report}\n"

    # Assemble $DATA-EXAMPLE based on mode
    example_section = ""
    if mode == "zero_shot":
        example_section = """=== DATA EXEMPLARS ($DATA-EXAMPLE) ===
[ZERO-SHOT MODE: No row examples are provided. Synthesize minority rows purely based on the domain report and variable specifications below.]"""

    elif mode == "few_shot":
        if few_shot_examples is None or few_shot_examples.empty:
            raise ValueError("few_shot mode requires few_shot_examples DataFrame")
        table_str = few_shot_examples.to_string(index=False)
        example_section = f"""=== DATA EXEMPLARS ($DATA-EXAMPLE) ===
Here are {len(few_shot_examples)} random examples of genuine minority-class samples:
```
{table_str}
```
(These illustrate valid minority format and property values; generate diverse new rows that preserve these properties without copying.)"""

    elif mode == "rag_tab":
        if context is None:
            raise ValueError("rag_tab mode requires a NeighborhoodContext")
        anchor_str = pd.DataFrame([context.anchor_sample]).to_string(index=False)
        neighbors_str = context.neighbors.to_string(index=False)
        bounds_lines = [
            f"- {feat}: local range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}"
            for feat, stats in context.feature_ranges.items()
        ]
        example_section = f"""=== LOCAL RETRIEVAL CONTEXT ($DATA-EXAMPLE) ===
SELECTED MINORITY ANCHOR POINT:
{anchor_str}

RETRIEVED {context.k} LOCAL MINORITY NEIGHBORS:
{neighbors_str}

LOCAL TOPOLOGY METRICS:
- Mean Distance in Standardized Space : {context.mean_distance:.3f}
- Local Density Metric               : {context.local_density:.3f}
- Boundary Purity                    : {context.boundary_purity:.1%}

LOCAL FEATURE ENVELOPE:
{chr(10).join(bounds_lines)}
(Condition your generation on this specific local neighborhood manifold. Do not simply interpolate along a straight line; respect non-linear inter-variable relationships.)"""

    prompt = f"""[Data generation prompt]:
Your task is to control a synthetic data generator that creates synthetic data conforming to a report and variable schema given to you.
The sequence of operations must output exactly {n_samples} data points ($N-OUT={n_samples}).
{report_section}
{example_section}

=== VARIABLE SPECIFICATIONS ($VARIABLE-DESCRIPTION-TEXT) ===
{schema_text}

=== CONSTRAINED GENERATION RULES ($DATA-GENERATION-JSON-SCHEMA) ===
1. Target Label: Every generated row must have `"{schema.target_col}": "{schema.minority_class}"`.
2. Schema Adherence: Return ONLY a valid JSON array of {n_samples} objects with the exact variable names provided above.
3. Domain Validity:
   - Numerical values must strictly respect the valid domain ranges specified.
   - Non-negative counts and integer variables must be valid positive integers.
   - Bounded probability ratios must stay strictly in [0.0, 1.0].
4. Output Format:
   - Output ONLY the raw JSON array.
   - Do NOT include markdown commentary, explanations, or notes.
"""
    return prompt.strip()


# Backwards compatibility aliases
build_llm_only_prompt = lambda schema, n_samples=5: build_generation_prompt(schema, mode="zero_shot", n_samples=n_samples)
build_rag_tab_prompt = lambda schema, context, n_samples=1: build_generation_prompt(schema, mode="rag_tab", context=context, n_samples=n_samples)
