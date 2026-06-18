"""Download the UCI household power dataset into data/raw."""

from __future__ import annotations

import argparse

from ml_homework.data import download_uci_power, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_uci.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"downloading_or_reusing={config.raw_dir}", flush=True)
    raw_file = download_uci_power(config.raw_dir, config.download_url, force=args.force)
    print(raw_file)


if __name__ == "__main__":
    main()
