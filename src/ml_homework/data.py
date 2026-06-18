"""Data pipeline for the UCI household power forecasting task."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_ZIP_NAME = "household_power_consumption.zip"
UCI_RAW_NAME = "household_power_consumption.txt"

ORIGINAL_COLUMNS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

COLUMN_RENAMES = {
    "Global_active_power": "global_active_power",
    "Global_reactive_power": "global_reactive_power",
    "Voltage": "voltage",
    "Global_intensity": "global_intensity",
    "Sub_metering_1": "sub_metering_1",
    "Sub_metering_2": "sub_metering_2",
    "Sub_metering_3": "sub_metering_3",
}

NUMERIC_COLUMNS = tuple(COLUMN_RENAMES.values())
SUM_COLUMNS = (
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
)
MEAN_COLUMNS = ("voltage", "global_intensity")


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved filesystem and modeling choices for data preparation."""

    source: str
    download_url: str
    raw_dir: Path
    raw_file: Path
    processed_dir: Path
    summary_path: Path
    scaler_path: Path
    target_column: str
    feature_columns: tuple[str, ...]
    input_length: int
    output_lengths: tuple[int, ...]
    train_ratio: float
    val_ratio: float
    split_policy: str
    missing_strategy: str
    weather_enabled: bool


def load_config(config_path: Path) -> PipelineConfig:
    """Load a YAML config and resolve relative paths from the repo root."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    repo_root = Path.cwd()
    split = raw_config.get("split", {})
    missing = raw_config.get("missing", {})
    weather = raw_config.get("weather", {})

    return PipelineConfig(
        source=str(raw_config.get("source", "uci_household_power")),
        download_url=str(raw_config.get("download_url", UCI_DOWNLOAD_URL)),
        raw_dir=_resolve_path(repo_root, raw_config["raw_dir"]),
        raw_file=_resolve_path(repo_root, raw_config["raw_file"]),
        processed_dir=_resolve_path(repo_root, raw_config["processed_dir"]),
        summary_path=_resolve_path(repo_root, raw_config["summary_path"]),
        scaler_path=_resolve_path(repo_root, raw_config["scaler_path"]),
        target_column=str(raw_config["target_column"]),
        feature_columns=tuple(raw_config["feature_columns"]),
        input_length=int(raw_config["input_length"]),
        output_lengths=tuple(int(value) for value in raw_config["output_lengths"]),
        train_ratio=float(split.get("train_ratio", 0.70)),
        val_ratio=float(split.get("val_ratio", 0.15)),
        split_policy=str(split.get("policy", "chronological_sliding_window_splits")),
        missing_strategy=str(
            missing.get("strategy", "minute_time_interpolate_then_ffill_bfill")
        ),
        weather_enabled=bool(weather.get("enabled", False)),
    )


def _resolve_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def download_uci_power(raw_dir: Path, url: str = UCI_DOWNLOAD_URL, force: bool = False) -> Path:
    """Download and extract the official UCI household power data file."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / UCI_RAW_NAME
    if raw_file.exists() and not force:
        return raw_file

    zip_path = raw_dir / UCI_ZIP_NAME
    if force or not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        members = set(archive.namelist())
        if UCI_RAW_NAME not in members:
            raise FileNotFoundError(f"{UCI_RAW_NAME} not found in {zip_path}")
        archive.extract(UCI_RAW_NAME, raw_dir)

    return raw_file


def read_uci_power_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the semicolon-delimited UCI text file into normalized columns."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"raw data file not found: {path}")

    frame = pd.read_csv(
        path,
        sep=";",
        na_values=["?", ""],
        usecols=ORIGINAL_COLUMNS,
        nrows=nrows,
        low_memory=False,
    )
    missing_columns = [column for column in ORIGINAL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing UCI columns: {missing_columns}")

    timestamp = pd.to_datetime(
        frame["Date"] + " " + frame["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    frame = frame.drop(columns=["Date", "Time"]).rename(columns=COLUMN_RENAMES)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.insert(0, "timestamp", timestamp)
    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def aggregate_daily_power(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill minute-level gaps and aggregate the power series to daily rows."""

    _require_columns(raw_frame, ("timestamp", *NUMERIC_COLUMNS))
    minute_frame = (
        raw_frame.set_index("timestamp")
        .sort_index()
        .asfreq("min")
    )

    missing_before = minute_frame[list(NUMERIC_COLUMNS)].isna().sum()
    missing_rows_before = int(minute_frame[list(NUMERIC_COLUMNS)].isna().any(axis=1).sum())
    minute_frame[list(NUMERIC_COLUMNS)] = (
        minute_frame[list(NUMERIC_COLUMNS)]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    missing_after = minute_frame[list(NUMERIC_COLUMNS)].isna().sum()

    minute_frame["sub_metering_remainder"] = (
        minute_frame["global_active_power"] * 1000.0 / 60.0
        - minute_frame["sub_metering_1"]
        - minute_frame["sub_metering_2"]
        - minute_frame["sub_metering_3"]
    )

    daily = minute_frame.resample("D").agg(
        {**{column: "sum" for column in SUM_COLUMNS}, **{column: "mean" for column in MEAN_COLUMNS}}
    )
    daily.index.name = "date"
    daily = daily.reset_index()

    summary = {
        "raw_minutes": int(len(raw_frame)),
        "calendar_minutes_after_reindex": int(len(minute_frame)),
        "missing_rows_before_fill": missing_rows_before,
        "missing_values_before_fill": {
            column: int(value) for column, value in missing_before.items()
        },
        "missing_values_after_fill": {
            column: int(value) for column, value in missing_after.items()
        },
        "daily_rows": int(len(daily)),
        "start_date": daily["date"].min().date().isoformat(),
        "end_date": daily["date"].max().date().isoformat(),
        "aggregation": {
            "sum": list(SUM_COLUMNS),
            "mean": list(MEAN_COLUMNS),
        },
        "missing_strategy": "minute_time_interpolate_then_ffill_bfill",
    }
    return daily, summary


def split_window_ranges(
    sample_count: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, tuple[int, int]]:
    """Split chronological window sample indices into train/val/test ranges."""

    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")

    train_end = int(sample_count * train_ratio)
    val_end = train_end + int(sample_count * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= sample_count:
        raise ValueError(f"split ratios produce empty split for {sample_count} windows")

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, sample_count),
    }


@dataclass(frozen=True)
class Standardizer:
    """Column-wise standardization state fitted on the training split only."""

    feature_columns: tuple[str, ...]
    mean: dict[str, float]
    scale: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "mean": self.mean,
            "scale": self.scale,
        }


def fit_standardizer(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> Standardizer:
    _require_columns(frame, feature_columns)
    mean = frame[list(feature_columns)].mean(axis=0)
    scale = frame[list(feature_columns)].std(axis=0, ddof=0).replace(0, 1.0)
    return Standardizer(
        feature_columns=feature_columns,
        mean={column: float(mean[column]) for column in feature_columns},
        scale={column: float(scale[column]) for column in feature_columns},
    )


def transform_features(frame: pd.DataFrame, standardizer: Standardizer) -> pd.DataFrame:
    _require_columns(frame, standardizer.feature_columns)
    transformed = frame.copy()
    for column in standardizer.feature_columns:
        transformed[column] = (transformed[column] - standardizer.mean[column]) / standardizer.scale[column]
    return transformed


def make_windows(
    daily: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    input_length: int,
    output_length: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Create sliding windows from a single chronological daily split."""

    _require_columns(daily, ("date", *feature_columns))
    if target_column not in feature_columns:
        raise ValueError(f"target_column must be included in feature_columns: {target_column}")

    window_count = len(daily) - input_length - output_length + 1
    feature_count = len(feature_columns)
    if window_count <= 0:
        return (
            np.empty((0, input_length, feature_count), dtype=np.float32),
            np.empty((0, output_length), dtype=np.float32),
            {
                "input_start": np.array([], dtype=str),
                "target_start": np.array([], dtype=str),
                "target_end": np.array([], dtype=str),
            },
        )

    feature_values = daily[list(feature_columns)].to_numpy(dtype=np.float32)
    target_values = daily[target_column].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(daily["date"]).dt.date.astype(str).to_numpy()

    x = np.empty((window_count, input_length, feature_count), dtype=np.float32)
    y = np.empty((window_count, output_length), dtype=np.float32)
    input_start: list[str] = []
    target_start: list[str] = []
    target_end: list[str] = []

    for index in range(window_count):
        input_end = index + input_length
        target_end_index = input_end + output_length
        x[index] = feature_values[index:input_end]
        y[index] = target_values[input_end:target_end_index]
        input_start.append(dates[index])
        target_start.append(dates[input_end])
        target_end.append(dates[target_end_index - 1])

    return (
        x,
        y,
        {
            "input_start": np.asarray(input_start),
            "target_start": np.asarray(target_start),
            "target_end": np.asarray(target_end),
        },
    )


def training_coverage_end_index(train_sample_count: int, input_length: int, output_length: int) -> int:
    """Return the exclusive daily-row end index covered by training windows."""

    if train_sample_count <= 0:
        raise ValueError(f"train_sample_count must be positive, got {train_sample_count}")
    return train_sample_count + input_length + output_length - 1


def prepare_power_data(config: PipelineConfig, nrows: int | None = None) -> dict[str, Any]:
    """Run the complete UCI data preparation pipeline."""

    raw_frame = read_uci_power_csv(config.raw_file, nrows=nrows)
    daily, summary = aggregate_daily_power(raw_frame)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    daily_path = config.processed_dir / "daily_power.csv"
    daily.to_csv(daily_path, index=False, lineterminator="\n")

    scaler_summary: dict[str, Any] = {}
    window_summary: dict[str, Any] = {}
    for output_length in config.output_lengths:
        total_samples = len(daily) - config.input_length - output_length + 1
        ranges = split_window_ranges(total_samples, config.train_ratio, config.val_ratio)
        train_start, train_end = ranges["train"]
        if train_start != 0:
            raise ValueError("training windows must start at the first chronological window")

        fit_end = training_coverage_end_index(train_end, config.input_length, output_length)
        fit_frame = daily.iloc[:fit_end].reset_index(drop=True)
        standardizer = fit_standardizer(fit_frame, config.feature_columns)
        transformed = transform_features(daily, standardizer)
        x, y, meta = make_windows(
            transformed,
            config.feature_columns,
            config.target_column,
            config.input_length,
            output_length,
        )
        if x.shape[0] != total_samples:
            raise RuntimeError(f"unexpected window count for output_length={output_length}")

        output_key = str(output_length)
        scaler_summary[output_key] = {
            **standardizer.to_json(),
            "fit_daily_rows": int(len(fit_frame)),
            "fit_start_date": fit_frame["date"].min().date().isoformat(),
            "fit_end_date": fit_frame["date"].max().date().isoformat(),
            "fit_rule": "all daily rows covered by training windows for this output length",
        }
        window_summary[output_key] = {
            "total_samples": int(total_samples),
            "input_length": config.input_length,
            "output_length": output_length,
            "split_ranges": {
                split_name: [int(start), int(end)]
                for split_name, (start, end) in ranges.items()
            },
            "scaler_key": output_key,
            "splits": {},
        }

        for split_name, (start, end) in ranges.items():
            window_path = config.processed_dir / (
                f"{split_name}_input{config.input_length}_output{output_length}.npz"
            )
            np.savez_compressed(
                window_path,
                x=x[start:end],
                y=y[start:end],
                feature_columns=np.asarray(config.feature_columns),
                target_column=np.asarray([config.target_column]),
                input_start=meta["input_start"][start:end],
                target_start=meta["target_start"][start:end],
                target_end=meta["target_end"][start:end],
            )
            split_meta = {
                key: values[start:end]
                for key, values in meta.items()
            }
            window_summary[output_key]["splits"][split_name] = {
                "path": str(window_path),
                "samples": int(end - start),
                "x_shape": list(x[start:end].shape),
                "y_shape": list(y[start:end].shape),
                "input_start": str(split_meta["input_start"][0]),
                "input_end": str(split_meta["input_start"][-1]),
                "target_start": str(split_meta["target_start"][0]),
                "target_end": str(split_meta["target_end"][-1]),
            }

    with config.scaler_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_policy": config.split_policy,
                "by_output_length": scaler_summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    full_summary = {
        "source": config.source,
        "raw_file": str(config.raw_file),
        "daily_path": str(daily_path),
        "scaler_path": str(config.scaler_path),
        "target_column": config.target_column,
        "feature_columns": list(config.feature_columns),
        "input_length": config.input_length,
        "output_lengths": list(config.output_lengths),
        "weather_enabled": config.weather_enabled,
        "split_policy": config.split_policy,
        **summary,
        "windows": window_summary,
    }
    with config.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(full_summary, handle, ensure_ascii=False, indent=2)
    return full_summary


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
