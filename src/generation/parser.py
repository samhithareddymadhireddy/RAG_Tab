"""JSON parser, schema validator, and error handling for LLM tabular generation."""
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


def extract_json_from_response(raw_text: str) -> str:
    """Extracts clean JSON substring from an LLM response even with code fences or commentary."""
    text = raw_text.strip()
    
    # Strip markdown code blocks
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]

    text = text.strip()
    
    # Look for outermost array [ ... ] or object { ... }
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket:last_bracket + 1]

    first_curly = text.find("{")
    last_curly = text.rfind("}")
    if first_curly != -1 and last_curly != -1 and last_curly > first_curly:
        return "[" + text[first_curly:last_curly + 1] + "]"

    return text


def parse_and_validate_tabular_json(
    raw_text: str,
    feature_cols: List[str],
    target_col: str = "Class",
    minority_class: str = "positive",
    target_class: Optional[str] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, int]]:
    """Parses JSON text into a validated pandas DataFrame, recording rejection reasons.

    Parameters
    ----------
    raw_text : str
        Raw string output from the LLM.
    feature_cols : list of str
        Expected feature names in exact order.
    target_col : str
        Target column name.
    minority_class : str
        Expected target value for synthesized samples (default 'positive').
    target_class : str, optional
        Target class override for multiclass synthesis.

    Returns
    -------
    valid_df : pd.DataFrame
        DataFrame of schema-validated synthetic samples.
    rejected : list of dict
        List of rejected samples with reasons.
    stats : dict
        Generation summary counts (total, valid, rejected).
    """
    clean_json = extract_json_from_response(raw_text)
    
    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        return (
            pd.DataFrame(columns=feature_cols + [target_col]),
            [{"raw_text": raw_text, "error": f"JSONDecodeError: {str(e)}"}],
            {"total_parsed": 0, "valid_count": 0, "rejected_count": 1}
        )

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        return (
            pd.DataFrame(columns=feature_cols + [target_col]),
            [{"raw_text": raw_text, "error": "Output is neither a list nor an object"}],
            {"total_parsed": 0, "valid_count": 0, "rejected_count": 1}
        )

    valid_rows = []
    rejected = []

    for item in data:
        if not isinstance(item, dict):
            rejected.append({"item": item, "reason": "Record is not a JSON object"})
            continue

        # Check required columns
        missing_cols = [c for c in feature_cols if c not in item]
        if missing_cols:
            rejected.append({"item": item, "reason": f"Missing columns: {missing_cols}"})
            continue

        # Validate numeric conversions
        row_vals = {}
        has_conversion_error = False
        for c in feature_cols:
            try:
                row_vals[c] = float(item[c])
            except (ValueError, TypeError):
                rejected.append({"item": item, "reason": f"Non-numeric value for column '{c}': {item.get(c)}"})
                has_conversion_error = True
                break

        if has_conversion_error:
            continue

        # Logical boundary checks for probabilities / non-negativity
        if "P_black" in row_vals and not (0.0 <= row_vals["P_black"] <= 1.0):
            rejected.append({"item": item, "reason": f"P_black out of [0, 1] range: {row_vals['P_black']}"})
            continue

        if "P_and" in row_vals and not (0.0 <= row_vals["P_and"] <= 1.0):
            rejected.append({"item": item, "reason": f"P_and out of [0, 1] range: {row_vals['P_and']}"})
            continue

        if "Height" in row_vals and row_vals["Height"] < 1:
            rejected.append({"item": item, "reason": f"Height must be >= 1, got {row_vals['Height']}"})
            continue

        # Target column
        chosen_class = target_class if target_class is not None else minority_class
        row_vals[target_col] = chosen_class
        valid_rows.append(row_vals)

    valid_df = pd.DataFrame(valid_rows)
    if not valid_df.empty:
        valid_df = valid_df[feature_cols + [target_col]]

    stats = {
        "total_parsed": len(data),
        "valid_count": len(valid_rows),
        "rejected_count": len(rejected)
    }

    return valid_df, rejected, stats
