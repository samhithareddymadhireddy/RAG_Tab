
"""Evaluation utilities package."""
from .metrics import evaluate_model
from .distribution_metrics import compute_marginal_fidelity, compute_correlation_difference

__all__ = ["evaluate_model", "compute_marginal_fidelity", "compute_correlation_difference"]
