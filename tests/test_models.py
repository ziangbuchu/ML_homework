from __future__ import annotations

import torch

from ml_homework.models import build_model


def test_lstm_forecaster_output_shape():
    model = build_model(
        {
            "name": "lstm",
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.0,
        },
        input_length=90,
        num_features=8,
        output_length=365,
    )
    inputs = torch.zeros(4, 90, 8)

    output = model(inputs)

    assert output.shape == (4, 365)


def test_transformer_forecaster_output_shape():
    model = build_model(
        {
            "name": "transformer",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
        },
        input_length=90,
        num_features=8,
        output_length=365,
    )
    inputs = torch.zeros(4, 90, 8)

    output = model(inputs)

    assert output.shape == (4, 365)


def test_tcn_transformer_forecaster_output_shape():
    model = build_model(
        {
            "name": "tcn_transformer",
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "conv_kernel_size": 5,
            "conv_layers": 1,
            "dropout": 0.1,
        },
        input_length=90,
        num_features=8,
        output_length=365,
    )
    inputs = torch.zeros(4, 90, 8)

    output = model(inputs)

    assert output.shape == (4, 365)
