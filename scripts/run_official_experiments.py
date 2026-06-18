"""Run the official 3 model x 2 horizon x 5 seed experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ml_homework.train import run_experiment


SEEDS = [1, 2, 3, 4, 5]
HORIZONS = [90, 365]
MODELS: dict[str, dict[str, Any]] = {
    "lstm": {
        "name": "lstm",
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "transformer": {
        "name": "transformer",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "tcn_transformer": {
        "name": "tcn_transformer",
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "conv_kernel_size": 5,
        "conv_layers": 1,
        "dropout": 0.1,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/official_runs")
    parser.add_argument("--config-dir", default="configs/official")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model_key, model_config in MODELS.items():
        for horizon in HORIZONS:
            for seed in SEEDS:
                config = build_config(
                    model_key=model_key,
                    model_config=model_config,
                    horizon=horizon,
                    seed=seed,
                    output_root=args.output_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                config_path = config_dir / f"{model_key}_h{horizon}_seed{seed}.yaml"
                with config_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

                run_dir = run_dir_for(config)
                metrics_path = run_dir / "metrics.json"
                if metrics_path.is_file() and not args.force:
                    skipped.append({"config": str(config_path), "run_dir": str(run_dir)})
                    print(f"skip_existing={run_dir}", flush=True)
                    continue

                print(f"run={config_path}", flush=True)
                result = run_experiment(config)
                completed.append({"config": str(config_path), **result})

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


def build_config(
    model_key: str,
    model_config: dict[str, Any],
    horizon: int,
    seed: int,
    output_root: str,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": f"official_{model_key}_h{horizon}",
            "seed": seed,
            "device": device,
            "output_root": output_root,
            "overwrite": True,
        },
        "data": {
            "train_path": f"data/processed/uci_power/train_input90_output{horizon}.npz",
            "val_path": f"data/processed/uci_power/val_input90_output{horizon}.npz",
            "test_path": f"data/processed/uci_power/test_input90_output{horizon}.npz",
        },
        "model": model_config,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 3,
        },
        "artifacts": {
            "plot_examples": 3,
        },
    }


def run_dir_for(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    return Path(experiment["output_root"]) / experiment["name"] / f"seed_{experiment['seed']}"


if __name__ == "__main__":
    main()
