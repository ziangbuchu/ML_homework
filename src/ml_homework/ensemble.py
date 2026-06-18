"""Prediction ensemble helpers for improved forecasting models."""

from __future__ import annotations

import numpy as np


def combine_predictions(
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    left_weight: float = 0.5,
) -> np.ndarray:
    """Return a fixed-weight average of two prediction arrays."""

    if left_prediction.shape != right_prediction.shape:
        raise ValueError(
            f"prediction shape mismatch: {left_prediction.shape} vs {right_prediction.shape}"
        )
    if not 0.0 <= left_weight <= 1.0:
        raise ValueError(f"left_weight must be in [0, 1], got {left_weight}")
    return left_weight * left_prediction + (1.0 - left_weight) * right_prediction
