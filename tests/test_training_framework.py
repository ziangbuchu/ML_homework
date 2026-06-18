from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_homework.train import run_experiment


def test_training_smoke_writes_artifacts(tmp_path):
    input_length = 4
    feature_count = 2
    output_length = 3
    for split, samples in {"train": 12, "val": 6, "test": 5}.items():
        x = np.random.default_rng(0).normal(size=(samples, input_length, feature_count)).astype("float32")
        y = x[:, -1, 0:1].repeat(output_length, axis=1).astype("float32")
        np.savez_compressed(tmp_path / f"{split}.npz", x=x, y=y)

    config = {
        "experiment": {
            "name": "unit_smoke",
            "seed": 7,
            "device": "cpu",
            "output_root": str(tmp_path / "runs"),
            "overwrite": True,
        },
        "data": {
            "train_path": str(tmp_path / "train.npz"),
            "val_path": str(tmp_path / "val.npz"),
            "test_path": str(tmp_path / "test.npz"),
        },
        "model": {"name": "linear"},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.01,
            "early_stopping_patience": 2,
        },
        "artifacts": {"plot_examples": 1},
    }

    result = run_experiment(config)
    run_dir = Path(result["run_dir"])

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "predictions_test.npz").is_file()
    assert (run_dir / "figures" / "prediction_000.png").is_file()
    assert "test" in result["metrics"]
