from __future__ import annotations

import numpy as np

from ml_homework.metrics import mse_mae


def test_mse_mae_values():
    prediction = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [5.0, 4.0]])

    metrics = mse_mae(prediction, target)

    assert metrics["mse"] == 1.25
    assert metrics["mae"] == 0.75


def test_mse_mae_rejects_shape_mismatch():
    prediction = np.zeros((2, 2))
    target = np.zeros((2, 3))

    try:
        mse_mae(prediction, target)
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("shape mismatch should raise ValueError")
