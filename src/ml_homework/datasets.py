"""Dataset utilities for prepared power forecasting windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PowerWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset backed by a prepared `.npz` window file."""

    def __init__(self, path: str | Path, max_samples: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"window file not found: {self.path}")

        archive = np.load(self.path, allow_pickle=False)
        self.x = archive["x"].astype(np.float32)
        self.y = archive["y"].astype(np.float32)
        if self.x.ndim != 3:
            raise ValueError(f"x must have shape [samples, input_length, features], got {self.x.shape}")
        if self.y.ndim != 2:
            raise ValueError(f"y must have shape [samples, output_length], got {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"x/y sample mismatch: {self.x.shape[0]} vs {self.y.shape[0]}")

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be positive, got {max_samples}")
            self.x = self.x[:max_samples]
            self.y = self.y[:max_samples]

    @property
    def input_shape(self) -> tuple[int, int]:
        return int(self.x.shape[1]), int(self.x.shape[2])

    @property
    def output_length(self) -> int:
        return int(self.y.shape[1])

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])
