"""Evaluate validation-only weight selection for the LSTM/Transformer ensemble."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_ensemble_results import (
    HORIZONS,
    LEFT_MODEL,
    RIGHT_MODEL,
    SEEDS,
    evaluate_model,
    load_model,
    load_splits,
    read_yaml,
    select_device,
)
from ml_homework.ensemble import combine_predictions
from ml_homework.metrics import mse_mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default="results/official_runs")
    parser.add_argument("--output-dir", default="results/official_summary")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = parser.parse_args()

    official_root = Path(args.official_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.weights)

    sweep_rows, selection_rows = run_sweep(
        official_root=official_root,
        device_name=args.device,
        weights=weights,
    )
    write_rows(output_dir / "ensemble_weight_sweep.csv", sweep_rows)
    write_rows(output_dir / "ensemble_weight_selection.csv", selection_rows)
    summary = summarize_selection(selection_rows)
    (output_dir / "ensemble_weight_summary.md").write_text(summary, encoding="utf-8")

    print(
        json.dumps(
            {
                "sweep_rows": len(sweep_rows),
                "selection_rows": len(selection_rows),
                "summary": str(output_dir / "ensemble_weight_summary.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_weights(raw: str) -> list[float]:
    weights = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not weights:
        raise ValueError("at least one weight is required")
    invalid = [weight for weight in weights if not 0.0 <= weight <= 1.0]
    if invalid:
        raise ValueError(f"weights must be in [0, 1], got {invalid}")
    if not any(abs(weight - 0.5) < 1e-9 for weight in weights):
        weights.append(0.5)
    return sorted(set(weights))


def run_sweep(
    official_root: Path,
    device_name: str,
    weights: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = select_device(device_name)
    sweep_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        for seed in SEEDS:
            left_run = official_root / f"official_{LEFT_MODEL}_h{horizon}" / f"seed_{seed}"
            right_run = official_root / f"official_{RIGHT_MODEL}_h{horizon}" / f"seed_{seed}"
            left_config = read_yaml(left_run / "config.yaml")
            right_config = read_yaml(right_run / "config.yaml")
            splits = load_splits(left_config["data"])
            left_model = load_model(left_config, left_run, splits["train"], device)
            right_model = load_model(right_config, right_run, splits["train"], device)

            split_predictions: dict[str, dict[str, np.ndarray]] = {}
            for split in ["val", "test"]:
                left_eval = evaluate_model(left_model, splits[split], device)
                right_eval = evaluate_model(right_model, splits[split], device)
                if not np.allclose(left_eval["target"], right_eval["target"]):
                    raise ValueError(f"target mismatch for h{horizon}/seed{seed}/{split}")
                split_predictions[split] = {
                    "left_prediction": left_eval["prediction"],
                    "right_prediction": right_eval["prediction"],
                    "target": left_eval["target"],
                }

            run_rows: list[dict[str, Any]] = []
            for weight in weights:
                for split, arrays in split_predictions.items():
                    prediction = combine_predictions(
                        arrays["left_prediction"],
                        arrays["right_prediction"],
                        left_weight=weight,
                    )
                    metrics = mse_mae(prediction, arrays["target"])
                    row = {
                        "horizon": horizon,
                        "seed": seed,
                        "split": split,
                        "left_weight": weight,
                        "right_weight": 1.0 - weight,
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                    }
                    sweep_rows.append(row)
                    run_rows.append(row)

            val_rows = [row for row in run_rows if row["split"] == "val"]
            test_rows = [row for row in run_rows if row["split"] == "test"]
            selected = min(val_rows, key=lambda row: (float(row["mse"]), float(row["mae"])))
            selected_test = match_weight(test_rows, float(selected["left_weight"]))
            equal_test = match_weight(test_rows, 0.5)
            oracle_test = min(test_rows, key=lambda row: (float(row["mse"]), float(row["mae"])))
            selection_rows.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "selected_left_weight": selected["left_weight"],
                    "selected_val_mse": selected["mse"],
                    "selected_val_mae": selected["mae"],
                    "selected_test_mse": selected_test["mse"],
                    "selected_test_mae": selected_test["mae"],
                    "equal_test_mse": equal_test["mse"],
                    "equal_test_mae": equal_test["mae"],
                    "oracle_test_left_weight": oracle_test["left_weight"],
                    "oracle_test_mse": oracle_test["mse"],
                    "oracle_test_mae": oracle_test["mae"],
                }
            )

    return sweep_rows, selection_rows


def match_weight(rows: list[dict[str, Any]], weight: float) -> dict[str, Any]:
    for row in rows:
        if abs(float(row["left_weight"]) - weight) < 1e-9:
            return row
    raise ValueError(f"missing weight {weight}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_selection(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Ensemble Weight Sweep Summary",
        "",
        "Weights are selected on validation MSE for each horizon/seed. Test-oracle rows are diagnostic only and are not used for model selection.",
        "",
        "| Horizon | Rule | Test MSE mean±std | Test MAE mean±std | Selected weights |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        group = [row for row in rows if int(row["horizon"]) == horizon]
        lines.append(summary_line(horizon, "equal 0.5", group, "equal_test"))
        lines.append(summary_line(horizon, "val-selected", group, "selected_test"))
        lines.append(summary_line(horizon, "test-oracle diagnostic", group, "oracle_test"))
    return "\n".join(lines) + "\n"


def summary_line(horizon: int, rule: str, rows: list[dict[str, Any]], prefix: str) -> str:
    mse = np.asarray([float(row[f"{prefix}_mse"]) for row in rows], dtype=float)
    mae = np.asarray([float(row[f"{prefix}_mae"]) for row in rows], dtype=float)
    if prefix == "selected_test":
        weights = [str(row["selected_left_weight"]) for row in rows]
    elif prefix == "oracle_test":
        weights = [str(row["oracle_test_left_weight"]) for row in rows]
    else:
        weights = ["0.5" for _ in rows]
    return (
        f"| {horizon} | {rule} | {mse.mean():.6f}±{mse.std(ddof=1):.6f} | "
        f"{mae.mean():.6f}±{mae.std(ddof=1):.6f} | {', '.join(weights)} |"
    )


if __name__ == "__main__":
    main()
