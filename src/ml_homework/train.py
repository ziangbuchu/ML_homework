"""Shared train/evaluate/save pipeline for forecasting experiments."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ml_homework.datasets import PowerWindowDataset
from ml_homework.metrics import mse_mae
from ml_homework.models import build_model


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run train, validation, test, and artifact writing from a config dict."""

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    run_dir = _run_dir(config)
    _prepare_run_dir(run_dir, bool(config.get("experiment", {}).get("overwrite", False)))

    dataset_config = config["data"]
    train_set = PowerWindowDataset(
        _resolve_path(dataset_config["train_path"]),
        _optional_int(dataset_config.get("max_train_samples")),
    )
    val_set = PowerWindowDataset(
        _resolve_path(dataset_config["val_path"]),
        _optional_int(dataset_config.get("max_val_samples")),
    )
    test_set = PowerWindowDataset(
        _resolve_path(dataset_config["test_path"]),
        _optional_int(dataset_config.get("max_test_samples")),
    )
    _assert_compatible_splits(train_set, val_set, test_set)

    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 32))
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
    }

    input_length, num_features = train_set.input_shape
    output_length = train_set.output_length
    model = build_model(config.get("model", {}), input_length, num_features, output_length)
    device = _select_device(str(config.get("experiment", {}).get("device", "auto")))
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()

    epochs = int(training_config.get("epochs", 20))
    patience = int(training_config.get("early_stopping_patience", 5))
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_eval = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_eval["metrics"]["mse"],
            "val_mae": val_eval["metrics"]["mae"],
        }
        history.append(row)

        if val_eval["metrics"]["mse"] < best_val:
            best_val = val_eval["metrics"]["mse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval = evaluate(model, loaders["train"], device)
    val_eval = evaluate(model, loaders["val"], device)
    test_eval = evaluate(model, loaders["test"], device)
    metrics = {
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train": train_eval["metrics"],
        "val": val_eval["metrics"],
        "test": test_eval["metrics"],
    }

    _write_artifacts(
        run_dir=run_dir,
        config=config,
        history=history,
        metrics=metrics,
        prediction=test_eval["prediction"],
        target=test_eval["target"],
        model=model,
        plot_count=int(config.get("artifacts", {}).get("plot_examples", 3)),
    )
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            output = model(inputs).detach().cpu().numpy()
            predictions.append(output)
            targets.append(target.numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    return {
        "prediction": prediction,
        "target": target,
        "metrics": mse_mae(prediction, target),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, target in loader:
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_size = int(inputs.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("empty training loader")
    return total_loss / total_samples


def _write_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    history: list[dict[str, float | int]],
    metrics: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
    model: nn.Module,
    plot_count: int,
) -> None:
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_mse", "val_mae"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history)

    np.savez_compressed(run_dir / "predictions_test.npz", prediction=prediction, target=target)
    torch.save(model.state_dict(), run_dir / "model.pt")

    for index in range(min(plot_count, prediction.shape[0])):
        _plot_prediction(
            prediction[index],
            target[index],
            run_dir / "figures" / f"prediction_{index:03d}.png",
        )


def _plot_prediction(prediction: np.ndarray, target: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(target, label="Ground Truth")
    plt.plot(prediction, label="Prediction")
    plt.xlabel("Forecast day")
    plt.ylabel("Standardized global active power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _run_dir(config: dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    output_root = _resolve_path(experiment.get("output_root", "results/runs"))
    name = str(experiment.get("name", "experiment"))
    seed = int(experiment.get("seed", 42))
    return output_root / name / f"seed_{seed}"


def _prepare_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=overwrite)


def _assert_compatible_splits(*datasets: PowerWindowDataset) -> None:
    input_shape = datasets[0].input_shape
    output_length = datasets[0].output_length
    for dataset in datasets[1:]:
        if dataset.input_shape != input_shape:
            raise ValueError(f"input shape mismatch: {dataset.input_shape} vs {input_shape}")
        if dataset.output_length != output_length:
            raise ValueError(f"output length mismatch: {dataset.output_length} vs {output_length}")


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device_name}")
    return torch.device(device_name)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
