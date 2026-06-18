"""Evaluate deterministic naive forecasting baselines for context."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml_homework.metrics import mse_mae


HORIZONS = [90, 365]
SPLITS = ["train", "val", "test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/uci_power")
    parser.add_argument("--output-dir", default="results/official_summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(Path(args.data_dir))
    write_rows(output_dir / "naive_baselines.csv", rows)
    summary = summarize_test_rows(rows)
    (output_dir / "naive_baseline_summary.md").write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary": str(output_dir / "naive_baseline_summary.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def collect_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for split in SPLITS:
            arrays = np.load(data_dir / f"{split}_input90_output{horizon}.npz")
            x = arrays["x"]
            y = arrays["y"]
            feature_columns = [str(value) for value in arrays["feature_columns"]]
            target_column = str(arrays["target_column"][0])
            target_index = feature_columns.index(target_column)
            target_history = x[:, :, target_index]
            predictions = {
                "last_value": np.repeat(target_history[:, -1, None], y.shape[1], axis=1),
                "input_mean": np.repeat(target_history.mean(axis=1, keepdims=True), y.shape[1], axis=1),
            }
            for model_name, prediction in predictions.items():
                metrics = mse_mae(prediction, y)
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "split": split,
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                    }
                )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_test_rows(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Naive Baseline Summary",
        "",
        "Deterministic baselines use only the target feature inside the 90-day input window. They are not trained and are reported as sanity checks, not as neural baselines.",
        "",
        "| Model | Horizon | Test MSE | Test MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    test_rows = [row for row in rows if row["split"] == "test"]
    for row in sorted(test_rows, key=lambda item: (int(item["horizon"]), str(item["model"]))):
        lines.append(
            f"| {row['model']} | {row['horizon']} | {float(row['mse']):.6f} | {float(row['mae']):.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
