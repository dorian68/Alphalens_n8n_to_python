from datetime import datetime, timezone, timedelta

import backtest_module as bm


def test_parse_date_only_start_end():
    start = bm._parse_date_arg("2025-08-02", is_end=False)
    end = bm._parse_date_arg("2025-08-02", is_end=True)
    assert start == datetime(2025, 8, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2025, 8, 2, 23, 59, 59, tzinfo=timezone.utc)


def test_parse_iso_with_z():
    dt = bm._parse_date_arg("2025-08-02T12:30:00Z", is_end=False)
    assert dt == datetime(2025, 8, 2, 12, 30, 0, tzinfo=timezone.utc)


def test_filter_bars_by_date():
    start_ts = datetime(2025, 8, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(5):
        ts = start_ts + timedelta(hours=i)
        bars.append(bm.Bar(ts=ts, open=1, high=1, low=1, close=1))

    start = datetime(2025, 8, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 8, 1, 3, 0, 0, tzinfo=timezone.utc)
    filtered = bm._filter_bars_by_date(bars, start, end)
    assert len(filtered) == 3
    assert filtered[0].ts == start
    assert filtered[-1].ts == end
