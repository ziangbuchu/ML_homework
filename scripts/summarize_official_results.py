"""Summarize official experiment metrics and generate report-ready figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "tcn_transformer": "TCN+Transformer",
    "lstm_transformer_ensemble": "LSTM+TF Ensemble",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="results/official_runs")
    parser.add_argument("--output-dir", default="results/official_summary")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows = collect_rows(runs_root)
    if not rows:
        raise ValueError(f"no official runs found under {runs_root}")

    write_rows(output_dir / "metrics_by_run.csv", rows)
    summary_rows = summarize(rows)
    write_rows(output_dir / "metrics_summary.csv", summary_rows)
    write_markdown_table(output_dir / "report_table.md", summary_rows)
    manifest = write_comparison_figures(output_dir / "figures", rows)
    with (output_dir / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows, "figures": manifest}, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"runs": len(rows), "summary_rows": len(summary_rows)}, indent=2))


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("official_*_h*/seed_*/metrics.json")):
        run_dir = metrics_path.parent
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_name = str(config["model"]["name"])
        horizon = int(str(config["experiment"]["name"]).rsplit("_h", 1)[1])
        seed = int(config["experiment"]["seed"])
        for split in ["train", "val", "test"]:
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "seed": seed,
                    "split": split,
                    "mse": float(metrics[split]["mse"]),
                    "mae": float(metrics[split]["mae"]),
                    "run_dir": str(run_dir),
                    "config_path": str(run_dir / "config.yaml"),
                    "prediction_path": str(run_dir / "predictions_test.npz"),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    groups = sorted({(row["model"], row["horizon"], row["split"]) for row in rows})
    for model, horizon, split in groups:
        group_rows = [
            row for row in rows
            if row["model"] == model and row["horizon"] == horizon and row["split"] == split
        ]
        mse = np.asarray([row["mse"] for row in group_rows], dtype=float)
        mae = np.asarray([row["mae"] for row in group_rows], dtype=float)
        summary_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "split": split,
                "n": len(group_rows),
                "mse_mean": float(mse.mean()),
                "mse_std": float(mse.std(ddof=1)) if len(mse) > 1 else 0.0,
                "mae_mean": float(mae.mean()),
                "mae_std": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
            }
        )
    return summary_rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    lines = [
        "| Model | Horizon | Test MSE mean±std | Test MAE mean±std | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(test_rows, key=lambda item: (item["horizon"], item["model"])):
        lines.append(
            "| {model} | {horizon} | {mse_mean:.6f}±{mse_std:.6f} | "
            "{mae_mean:.6f}±{mae_std:.6f} | {n} |".format(
                **{**row, "model": model_label(str(row["model"]))}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_figures(figures_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for horizon in sorted({row["horizon"] for row in rows}):
        selected = [
            row for row in rows
            if row["horizon"] == horizon and row["split"] == "test" and row["seed"] == 1
        ]
        if not selected:
            continue

        plt.figure(figsize=(9, 4.8))
        ground_truth_plotted = False
        for row in sorted(selected, key=lambda item: item["model"]):
            arrays = np.load(row["prediction_path"])
            prediction = arrays["prediction"][0]
            target = arrays["target"][0]
            if not ground_truth_plotted:
                plt.plot(target, label="Ground Truth", linewidth=2.0)
                ground_truth_plotted = True
            plt.plot(prediction, label=model_label(str(row["model"])))

        plt.title(f"{horizon}-day forecast comparison (seed 1, first test window)")
        plt.xlabel("Forecast day")
        plt.ylabel("Standardized global active power")
        plt.legend()
        plt.tight_layout()
        path = figures_dir / f"comparison_h{horizon}_seed1.png"
        plt.savefig(path, dpi=180)
        plt.close()
        manifest.append({"horizon": str(horizon), "path": str(path)})
    return manifest


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


if __name__ == "__main__":
    main()
