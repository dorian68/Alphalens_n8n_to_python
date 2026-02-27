from pathlib import Path

import pandas as pd

from alphalens_forecast.visualization.trade_panorama import (
    load_backtest_json,
    load_trades_csv,
    load_trades_json,
    load_trades_dir,
    render_trade_panorama,
)


def _write_sample_csv(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "as_of": "2025-08-01T00:00:00Z",
                "entry_time": "2025-08-01T01:00:00Z",
                "exit_time": "2025-08-01T02:00:00Z",
                "direction": "long",
                "entry": 100.0,
                "stop": 99.0,
                "take_profit": 101.0,
                "exit_price": 101.0,
                "pnl": 1.0,
                "outcome": "tp",
                "regime_label": "RANGE",
            },
            {
                "as_of": "2025-08-01T03:00:00Z",
                "entry_time": "2025-08-01T03:30:00Z",
                "exit_time": "2025-08-01T04:00:00Z",
                "direction": "short",
                "entry": 102.0,
                "stop": 103.0,
                "take_profit": 100.5,
                "exit_price": 103.0,
                "pnl": -1.5,
                "outcome": "sl",
                "regime_label": None,
            },
        ]
    )
    df.to_csv(path, index=False)


def test_load_trades_csv(tmp_path):
    csv_path = tmp_path / "trades.csv"
    _write_sample_csv(csv_path)
    df = load_trades_csv(str(csv_path))
    assert "regime_label" in df.columns
    assert df["regime_label"].isna().sum() == 0


def test_render_trade_panorama(tmp_path):
    csv_path = tmp_path / "trades.csv"
    _write_sample_csv(csv_path)
    out_html = tmp_path / "out.html"
    result = render_trade_panorama(str(csv_path), out_html=str(out_html))
    assert Path(result).exists()
    assert Path(result).read_text(encoding="utf-8").strip().startswith("<")


def test_load_backtest_json(tmp_path):
    json_path = tmp_path / "backtest.json"
    json_path.write_text('{"metadata": {"symbol": "BTC/USD"}}', encoding="utf-8")
    data = load_backtest_json(str(json_path))
    assert data["metadata"]["symbol"] == "BTC/USD"


def test_load_trades_json(tmp_path):
    json_path = tmp_path / "trades.json"
    json_path.write_text('[{"entry_time":"2025-08-01T01:00:00Z","entry":100,"direction":"long"}]', encoding="utf-8")
    df = load_trades_json(str(json_path))
    assert len(df) == 1
    assert df["direction"].iloc[0] == "long"


def test_load_trades_dir(tmp_path):
    dir_path = tmp_path / "trades"
    dir_path.mkdir()
    (dir_path / "a.json").write_text('[{"entry_time":"2025-08-01T01:00:00Z","entry":100,"direction":"long"}]', encoding="utf-8")
    (dir_path / "b.json").write_text('[{"entry_time":"2025-08-01T02:00:00Z","entry":101,"direction":"short"}]', encoding="utf-8")
    df = load_trades_dir(str(dir_path))
    assert len(df) == 2
