from datetime import datetime, timedelta, timezone

import pytest

import backtest_module as bm


def _coerce_dt(value):
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else value


def test_parse_batch_bars():
    assert bm._parse_batch_bars("200") == 200
    assert bm._parse_batch_bars("200bars") == 200
    assert bm._parse_batch_bars("200b") == 200
    assert bm._parse_batch_bars("") is None
    assert bm._parse_batch_bars("1M") is None


def test_build_bar_batches_basic():
    batches = bm._build_bar_batches(total_bars=10, batch_bars=4, step_bars=None)
    assert batches == [
        {"start_idx": 0, "end_idx": 4},
        {"start_idx": 4, "end_idx": 8},
        {"start_idx": 8, "end_idx": 10},
    ]


def test_build_bar_batches_rolling():
    batches = bm._build_bar_batches(total_bars=10, batch_bars=5, step_bars=2)
    assert batches[0] == {"start_idx": 0, "end_idx": 5}
    assert batches[1] == {"start_idx": 2, "end_idx": 7}
    assert batches[-1]["end_idx"] == 10


def test_build_time_batches_basic():
    pd = pytest.importorskip("pandas")
    timestamps = [
        datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(10)
    ]
    window = pd.tseries.frequencies.to_offset("3D")
    step = pd.tseries.frequencies.to_offset("3D")
    batches = bm._build_time_batches(timestamps, window, step)
    assert _coerce_dt(batches[0]["start_ts"]) == timestamps[0]
    assert _coerce_dt(batches[0]["end_ts"]) == timestamps[0] + timedelta(days=3)
    assert _coerce_dt(batches[-1]["end_ts"]) == timestamps[-1]
