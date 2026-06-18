"""Build official improved-model results from an LSTM/Transformer ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ml_homework.datasets import PowerWindowDataset
from ml_homework.ensemble import combine_predictions
from ml_homework.metrics import mse_mae
from ml_homework.models import build_model


SEEDS = [1, 2, 3, 4, 5]
HORIZONS = [90, 365]
LEFT_MODEL = "lstm"
RIGHT_MODEL = "transformer"
ENSEMBLE_MODEL = "lstm_transformer_ensemble"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default="results/official_runs")
    parser.add_argument("--output-root", default="results/official_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--left-weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    official_root = Path(args.official_root)
    output_root = Path(args.output_root)
    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for horizon in HORIZONS:
        for seed in SEEDS:
            run_dir = output_root / f"official_{ENSEMBLE_MODEL}_h{horizon}" / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            if metrics_path.is_file() and not args.force:
                skipped.append({"run_dir": str(run_dir)})
                print(f"skip_existing={run_dir}", flush=True)
                continue

            print(f"build_ensemble=h{horizon}/seed_{seed}", flush=True)
            result = build_one_ensemble(
                official_root=official_root,
                run_dir=run_dir,
                horizon=horizon,
                seed=seed,
                device_name=args.device,
                left_weight=args.left_weight,
            )
            completed.append(result)

    print(
        json.dumps(
            {
                "completed": completed,
                "skipped": skipped,
                "completed_count": len(completed),
                "skipped_count": len(skipped),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_one_ensemble(
    official_root: Path,
    run_dir: Path,
    horizon: int,
    seed: int,
    device_name: str,
    left_weight: float,
) -> dict[str, Any]:
    left_run = official_root / f"official_{LEFT_MODEL}_h{horizon}" / f"seed_{seed}"
    right_run = official_root / f"official_{RIGHT_MODEL}_h{horizon}" / f"seed_{seed}"
    left_config = read_yaml(left_run / "config.yaml")
    right_config = read_yaml(right_run / "config.yaml")

    device = select_device(device_name)
    splits = load_splits(left_config["data"])
    left_model = load_model(left_config, left_run, splits["train"], device)
    right_model = load_model(right_config, right_run, splits["train"], device)

    metrics: dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "ensemble": {
            "left_model": LEFT_MODEL,
            "right_model": RIGHT_MODEL,
            "left_weight": left_weight,
            "right_weight": 1.0 - left_weight,
        },
    }
    predictions_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, dataset in splits.items():
        left_eval = evaluate_model(left_model, dataset, device)
        right_eval = evaluate_model(right_model, dataset, device)
        if not np.allclose(left_eval["target"], right_eval["target"]):
            raise ValueError(f"target mismatch for h{horizon}/seed{seed}/{split}")
        prediction = combine_predictions(
            left_eval["prediction"],
            right_eval["prediction"],
            left_weight=left_weight,
        )
        target = left_eval["target"]
        metrics[split] = mse_mae(prediction, target)
        predictions_by_split[split] = {"prediction": prediction, "target": target}

    config = ensemble_config(
        left_config=left_config,
        right_config=right_config,
        horizon=horizon,
        seed=seed,
        output_root=run_dir.parents[1],
        device_name=device_name,
        left_weight=left_weight,
    )
    write_artifacts(run_dir, config, metrics, predictions_by_split["test"])
    return {
        "run_dir": str(run_dir),
        "horizon": horizon,
        "seed": seed,
        "test": metrics["test"],
    }


def load_splits(data_config: dict[str, str]) -> dict[str, PowerWindowDataset]:
    return {
        "train": PowerWindowDataset(resolve_path(data_config["train_path"])),
        "val": PowerWindowDataset(resolve_path(data_config["val_path"])),
        "test": PowerWindowDataset(resolve_path(data_config["test_path"])),
    }


def load_model(
    config: dict[str, Any],
    run_dir: Path,
    train_set: PowerWindowDataset,
    device: torch.device,
) -> torch.nn.Module:
    input_length, num_features = train_set.input_shape
    output_length = train_set.output_length
    model = build_model(config["model"], input_length, num_features, output_length)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    dataset: PowerWindowDataset,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, target in loader:
            output = model(inputs.to(device)).detach().cpu().numpy()
            predictions.append(output)
            targets.append(target.numpy())
    return {
        "prediction": np.concatenate(predictions, axis=0),
        "target": np.concatenate(targets, axis=0),
    }


def write_artifacts(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    test_prediction: dict[str, np.ndarray],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        run_dir / "predictions_test.npz",
        prediction=test_prediction["prediction"],
        target=test_prediction["target"],
    )
    for index in range(min(3, test_prediction["prediction"].shape[0])):
        plot_prediction(
            prediction=test_prediction["prediction"][index],
            target=test_prediction["target"][index],
            path=run_dir / "figures" / f"prediction_{index:03d}.png",
        )


def plot_prediction(prediction: np.ndarray, target: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(target, label="Ground Truth")
    plt.plot(prediction, label="Prediction")
    plt.xlabel("Forecast day")
    plt.ylabel("Standardized global active power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def ensemble_config(
    left_config: dict[str, Any],
    right_config: dict[str, Any],
    horizon: int,
    seed: int,
    output_root: Path,
    device_name: str,
    left_weight: float,
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": f"official_{ENSEMBLE_MODEL}_h{horizon}",
            "seed": seed,
            "device": device_name,
            "output_root": str(output_root),
            "overwrite": True,
        },
        "data": left_config["data"],
        "model": {
            "name": ENSEMBLE_MODEL,
            "left_model": left_config["model"],
            "right_model": right_config["model"],
            "left_weight": left_weight,
            "right_weight": 1.0 - left_weight,
        },
        "training": {
            "type": "posthoc_equal_weight_ensemble",
            "source_runs": [LEFT_MODEL, RIGHT_MODEL],
        },
        "artifacts": {"plot_examples": 3},
    }


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device_name}")
    return torch.device(device_name)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
