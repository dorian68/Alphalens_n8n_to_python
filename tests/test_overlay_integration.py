from dataclasses import asdict
from datetime import datetime, timezone

import backtest_module as bm
from alphalens_forecast.trading.overlays.regime_risk_overlay import (
    OverlayConfig,
    RegimeRiskOverlay,
)


def _make_trade_result() -> bm.TradeResult:
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return bm.TradeResult(
        as_of=now,
        entry_time=now,
        exit_time=now,
        direction="long",
        entry=100.0,
        stop=99.0,
        take_profit=101.0,
        exit_price=101.0,
        pnl=1.0,
        r_multiple=1.0,
        outcome="tp",
        bars_held=1,
        confidence=0.8,
        position_size=1.0,
        notional=100.0,
        leverage=1.0,
    )


def test_overlay_keeps_schema_and_uses_skip_none():
    trade = _make_trade_result()
    before = asdict(trade)
    overlay = RegimeRiskOverlay(OverlayConfig(vol_ref=0.01, range_scale=0.7))
    ctx = {
        "regime_enabled": True,
        "regime_label": "RANGE",
        "regime_confidence": 0.2,
        "entry_model_vol": 0.02,
    }
    res = overlay.apply(trade, ctx)
    after = asdict(res)
    assert set(before.keys()) == set(after.keys())
    assert after["direction"] == before["direction"]
    assert round(after["position_size"], 6) != round(before["position_size"], 6)

    blocked = overlay.apply(
        _make_trade_result(),
        {"regime_enabled": True, "regime_label": "TREND_DOWN"},
    )
    assert blocked is None
