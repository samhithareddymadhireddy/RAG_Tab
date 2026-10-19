"""Baselines package for traditional and advanced tabular oversampling."""
from .resamplers import get_resampler, apply_resampling

__all__ = ["get_resampler", "apply_resampling"]
