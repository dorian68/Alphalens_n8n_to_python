from datetime import datetime, timedelta, timezone
from typing import Optional

import backtest_module as bm


def _make_bars(count: int = 40, start: Optional[datetime] = None):
    start_ts = start or datetime(2020, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for i in range(count):
        ts = start_ts + timedelta(hours=i)
        drift = 0.2 if i % 2 == 0 else -0.1
        open_p = price
        high = price + 0.5
        low = price - 0.5
        close = price + drift
        bars.append(
            bm.Bar(
                ts=ts,
                open=open_p,
                high=high,
                low=low,
                close=close,
                volume=1.0,
            )
        )
        price = close
    return bars


def test_backtest_metrics_block_present(tmp_path):
    parser = bm.build_arg_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "TEST/USD",
            "--timeframe",
            "1h",
            "--gen-mode",
            "simple",
            "--forecast-mode",
            "off",
            "--horizons",
            "3",
            "--warmup-bars",
            "5",
            "--decision-every",
            "2",
            "--max-hold-bars",
            "3",
        ]
    )
    args.cache_dir = str(tmp_path / "cache")
    args.trades_json = ""
    args.trades_csv = ""
    args.record_runs = 0
    args.runs_dir = str(tmp_path / "runs")
    args.sizing_mode = "none"

    bars = _make_bars(40)
    trades, summary = bm._run_backtest_core(bars, args, write_outputs=False, record_runs=False)

    assert "metrics" in summary
    assert summary["metrics"]["n_trades"] == len(trades)
