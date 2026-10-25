
"""Evaluation utilities package."""
from .metrics import evaluate_model
from .distribution_metrics import (
    compute_marginal_fidelity,
    compute_correlation_difference,
    compute_distance_to_closest_record,
    evaluate_distribution_fidelity,
)

__all__ = [
    "evaluate_model",
    "compute_marginal_fidelity",
    "compute_correlation_difference",
    "compute_distance_to_closest_record",
    "evaluate_distribution_fidelity",
]
