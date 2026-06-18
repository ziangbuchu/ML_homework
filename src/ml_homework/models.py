"""Model registry used by the shared training framework."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class LinearForecaster(nn.Module):
    """Small smoke model that maps a flattened input window to the forecast horizon."""

    def __init__(self, input_length: int, num_features: int, output_length: int) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_features = num_features
        self.output_length = output_length
        self.projection = nn.Linear(input_length * num_features, output_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        return self.projection(inputs.reshape(batch_size, -1))


class LSTMForecaster(nn.Module):
    """Encoder-only LSTM baseline with a direct forecast projection head."""

    def __init__(
        self,
        input_length: int,
        num_features: int,
        output_length: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if output_length <= 0:
            raise ValueError(f"output_length must be positive, got {output_length}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, output_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(inputs)
        return self.projection(hidden[-1])


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed positional encoding for batch-first time series tensors."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.shape[1]]


class TransformerForecaster(nn.Module):
    """Transformer encoder baseline with pooled representation forecast head."""

    def __init__(
        self,
        input_length: int,
        num_features: int,
        output_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if output_length <= 0:
            raise ValueError(f"output_length must be positive, got {output_length}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got {d_model} and {nhead}")

        self.input_projection = nn.Linear(num_features, d_model)
        self.position = SinusoidalPositionalEncoding(input_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, output_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(inputs)
        encoded = self.position(encoded)
        encoded = self.encoder(encoded)
        pooled = encoded.mean(dim=1)
        return self.projection(pooled)


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block for local pattern extraction."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to preserve length, got {kernel_size}")
        padding = kernel_size // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = inputs.transpose(1, 2)
        hidden = self.conv(hidden).transpose(1, 2)
        hidden = self.dropout(self.activation(hidden))
        return self.norm(hidden + residual)


class TCNTransformerForecaster(nn.Module):
    """CNN/TCN + Transformer model for local fluctuation and context modeling."""

    def __init__(
        self,
        input_length: int,
        num_features: int,
        output_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        conv_kernel_size: int = 5,
        conv_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if conv_layers <= 0:
            raise ValueError(f"conv_layers must be positive, got {conv_layers}")

        self.input_projection = nn.Linear(num_features, d_model)
        self.conv_blocks = nn.ModuleList(
            TemporalConvBlock(d_model=d_model, kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(conv_layers)
        )
        self.position = SinusoidalPositionalEncoding(input_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(d_model, output_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        for block in self.conv_blocks:
            hidden = block(hidden)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)
        pooled = hidden.mean(dim=1)
        return self.projection(pooled)


def build_model(
    model_config: dict[str, Any],
    input_length: int,
    num_features: int,
    output_length: int,
) -> nn.Module:
    """Build a model by name; future tasks extend this registry."""

    model_name = str(model_config.get("name", "linear"))
    if model_name == "linear":
        return LinearForecaster(input_length, num_features, output_length)
    if model_name == "lstm":
        return LSTMForecaster(
            input_length=input_length,
            num_features=num_features,
            output_length=output_length,
            hidden_size=int(model_config.get("hidden_size", 64)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    if model_name == "transformer":
        return TransformerForecaster(
            input_length=input_length,
            num_features=num_features,
            output_length=output_length,
            d_model=int(model_config.get("d_model", 64)),
            nhead=int(model_config.get("nhead", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dim_feedforward=int(model_config.get("dim_feedforward", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if model_name == "tcn_transformer":
        return TCNTransformerForecaster(
            input_length=input_length,
            num_features=num_features,
            output_length=output_length,
            d_model=int(model_config.get("d_model", 64)),
            nhead=int(model_config.get("nhead", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dim_feedforward=int(model_config.get("dim_feedforward", 128)),
            conv_kernel_size=int(model_config.get("conv_kernel_size", 5)),
            conv_layers=int(model_config.get("conv_layers", 1)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown model name: {model_name}")
