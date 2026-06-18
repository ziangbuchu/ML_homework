from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml_homework.data import (
    aggregate_daily_power,
    fit_standardizer,
    make_windows,
    read_uci_power_csv,
    split_window_ranges,
    training_coverage_end_index,
    transform_features,
)


def test_read_fill_and_daily_aggregation(tmp_path):
    raw_path = tmp_path / "household_power_consumption.txt"
    raw_path.write_text(
        "\n".join(
            [
                "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3",
                "1/1/2020;00:00:00;1.0;0.1;230.0;1.0;1.0;2.0;3.0",
                "1/1/2020;00:01:00;?;?;?;?;?;?;?",
                "1/1/2020;00:02:00;3.0;0.3;232.0;3.0;3.0;4.0;5.0",
            ]
        ),
        encoding="utf-8",
    )

    raw = read_uci_power_csv(raw_path)
    daily, summary = aggregate_daily_power(raw)

    assert summary["missing_rows_before_fill"] == 1
    assert summary["missing_values_after_fill"]["global_active_power"] == 0
    assert len(daily) == 1
    assert daily.loc[0, "global_active_power"] == 6.0
    assert daily.loc[0, "global_reactive_power"] == 0.6
    assert daily.loc[0, "voltage"] == 231.0
    assert math.isclose(daily.loc[0, "sub_metering_remainder"], 73.0)


def test_window_level_split_and_window_shapes():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "global_active_power": np.arange(20, dtype=float),
            "voltage": np.arange(100, 120, dtype=float),
        }
    )

    ranges = split_window_ranges(sample_count=16, train_ratio=0.5, val_ratio=0.25)
    assert ranges == {"train": (0, 8), "val": (8, 12), "test": (12, 16)}
    assert training_coverage_end_index(
        train_sample_count=8,
        input_length=3,
        output_length=2,
    ) == 12

    feature_columns = ("global_active_power", "voltage")
    standardizer = fit_standardizer(daily.iloc[:12], feature_columns)
    transformed = transform_features(daily, standardizer)
    x, y, meta = make_windows(
        transformed,
        feature_columns,
        target_column="global_active_power",
        input_length=3,
        output_length=2,
    )

    assert x.shape == (16, 3, 2)
    assert y.shape == (16, 2)
    assert meta["input_start"][0] == "2020-01-01"
    assert meta["target_start"][0] == "2020-01-04"
    np.testing.assert_allclose(
        y[0],
        transformed["global_active_power"].iloc[3:5].to_numpy(dtype=np.float32),
    )
