"""Run a configured forecasting experiment."""

from __future__ import annotations

import argparse
import json

from ml_homework.train import load_experiment_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    result = run_experiment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
