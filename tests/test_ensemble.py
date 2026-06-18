from __future__ import annotations

import numpy as np

from ml_homework.ensemble import combine_predictions


def test_combine_predictions_equal_weight():
    left = np.array([[1.0, 3.0]])
    right = np.array([[3.0, 7.0]])

    prediction = combine_predictions(left, right)

    np.testing.assert_allclose(prediction, np.array([[2.0, 5.0]]))


def test_combine_predictions_custom_weight():
    left = np.array([[1.0, 3.0]])
    right = np.array([[3.0, 7.0]])

    prediction = combine_predictions(left, right, left_weight=0.25)

    np.testing.assert_allclose(prediction, np.array([[2.5, 6.0]]))


def test_combine_predictions_rejects_shape_mismatch():
    left = np.zeros((2, 3))
    right = np.zeros((2, 4))

    try:
        combine_predictions(left, right)
    except ValueError as error:
        assert "prediction shape mismatch" in str(error)
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def test_combine_predictions_rejects_invalid_weight():
    prediction = np.zeros((2, 3))

    try:
        combine_predictions(prediction, prediction, left_weight=1.2)
    except ValueError as error:
        assert "left_weight" in str(error)
    else:
        raise AssertionError("invalid weight should raise ValueError")
