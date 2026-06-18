"""Prepare daily UCI household power windows for training."""

from __future__ import annotations

import argparse
import json

from ml_homework.data import load_config, prepare_power_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_uci.yaml")
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = prepare_power_data(config, nrows=args.nrows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
