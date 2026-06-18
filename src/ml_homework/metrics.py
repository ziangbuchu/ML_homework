"""Shared metric functions for all forecasting models."""

from __future__ import annotations

import numpy as np


def mse_mae(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute mean squared error and mean absolute error with one shared definition."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")

    diff = prediction - target
    return {
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
    }
