from alphalens_forecast.trading.overlays.regime_risk_overlay import (
    OverlayConfig,
    RegimeRiskOverlay,
)


def _base_trade(direction: str = "long"):
    return {
        "direction": direction,
        "position_size": 1.0,
        "notional": 100.0,
        "pnl": 10.0,
        "leverage": 1.0,
    }


def test_stress_chop_blocks_without_breakout():
    overlay = RegimeRiskOverlay(OverlayConfig())
    trade = _base_trade()
    ctx = {
        "regime_enabled": True,
        "regime_label": "STRESS_CHOP",
        "regime_confidence": 0.5,
        "model_confidence": 0.5,
    }
    assert overlay.apply(trade, ctx) is None


def test_stress_chop_allows_breakout_with_risk_off():
    overlay = RegimeRiskOverlay(OverlayConfig())
    trade = _base_trade()
    ctx = {
        "regime_enabled": True,
        "regime_label": "STRESS_CHOP",
        "regime_route": "breakout",
        "regime_confidence": 0.5,
        "model_confidence": 0.5,
    }
    res = overlay.apply(trade, ctx)
    assert res is trade
    assert round(trade["position_size"], 6) == 0.25
    assert round(trade["pnl"], 6) == 2.5


def test_trend_down_blocks_long_by_default():
    overlay = RegimeRiskOverlay(OverlayConfig())
    trade = _base_trade(direction="long")
    ctx = {"regime_enabled": True, "regime_label": "TREND_DOWN"}
    assert overlay.apply(trade, ctx) is None


def test_trend_up_blocks_short_by_default():
    overlay = RegimeRiskOverlay(OverlayConfig())
    trade = _base_trade(direction="short")
    ctx = {"regime_enabled": True, "regime_label": "TREND_UP"}
    assert overlay.apply(trade, ctx) is None


def test_range_scales_down_on_high_vol():
    overlay = RegimeRiskOverlay(OverlayConfig(vol_ref=0.01, range_scale=0.7))
    trade = _base_trade(direction="long")
    ctx = {
        "regime_enabled": True,
        "regime_label": "RANGE",
        "regime_confidence": 0.3,
        "entry_model_vol": 0.02,
    }
    res = overlay.apply(trade, ctx)
    assert res is trade
    assert round(trade["position_size"], 6) == 0.35


def test_noop_when_regime_disabled():
    overlay = RegimeRiskOverlay(OverlayConfig())
    trade = _base_trade()
    ctx = {"regime_enabled": False, "regime_label": "TREND_UP"}
    res = overlay.apply(trade, ctx)
    assert res is trade
    assert trade["position_size"] == 1.0
