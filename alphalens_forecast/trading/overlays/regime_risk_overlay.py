"""Regime-aware risk & eligibility overlay (post-processing)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_field(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


@dataclass
class OverlayConfig:
    enabled: bool = True
    stress_breakout_regime_conf: float = 0.75
    stress_breakout_model_conf: float = 0.7
    stress_chop_size_scale: float = 0.25
    range_scale: float = 0.7
    range_confidence_threshold: float = 0.65
    vol_ref: Optional[float] = None
    vol_scale_min: float = 0.5
    vol_scale_max: float = 1.2


class RegimeRiskOverlay:
    def __init__(self, config: Optional[OverlayConfig] = None) -> None:
        self.config = config or OverlayConfig()

    def apply(self, trade: Any, context: Dict[str, Any]) -> Any:
        if trade is None:
            return None
        if not self.config.enabled:
            return trade
        context = context or {}
        regime_enabled = bool(context.get("regime_enabled"))
        regime_label = context.get("regime_label")
        if not regime_enabled or not regime_label:
            return trade

        label = str(regime_label).strip().upper()
        direction = _get_field(trade, "direction")
        direction = str(direction).strip().lower() if direction is not None else ""

        if label == "STRESS_CHOP":
            if not self._breakout_allowed(context):
                return None
            self._scale_trade(trade, self.config.stress_chop_size_scale, context)
            return trade

        if label == "TREND_DOWN":
            if direction == "long" and not self._has_mean_reversion_tag(context):
                return None
            if direction == "short":
                self._scale_trade(trade, self._vol_scale(context), context)
            return trade

        if label == "TREND_UP":
            if direction == "short" and not self._has_mean_reversion_tag(context):
                return None
            if direction == "long":
                self._scale_trade(trade, self._vol_scale(context), context)
            return trade

        if label == "RANGE":
            scale = self.config.range_scale
            regime_conf = context.get("regime_confidence")
            vol_ref = self._resolve_vol_ref(context)
            entry_vol = context.get("entry_model_vol")
            if (
                _is_number(regime_conf)
                and float(regime_conf) >= self.config.range_confidence_threshold
                and vol_ref is not None
                and _is_number(entry_vol)
                and float(entry_vol) <= float(vol_ref)
            ):
                scale = 1.0
            else:
                scale = scale * self._vol_scale(context)
            self._scale_trade(trade, scale, context)
            return trade

        return trade

    def _resolve_vol_ref(self, context: Dict[str, Any]) -> Optional[float]:
        if _is_number(self.config.vol_ref):
            return float(self.config.vol_ref)
        vol_ref = context.get("vol_ref")
        if _is_number(vol_ref):
            return float(vol_ref)
        return None

    def _vol_scale(self, context: Dict[str, Any]) -> float:
        vol_ref = self._resolve_vol_ref(context)
        entry_vol = context.get("entry_model_vol")
        if vol_ref is None or not _is_number(entry_vol):
            return 1.0
        entry_vol_val = float(entry_vol)
        if entry_vol_val <= 0:
            return 1.0
        raw = float(vol_ref) / entry_vol_val
        return _clamp(raw, self.config.vol_scale_min, self.config.vol_scale_max)

    def _scale_trade(self, trade: Any, scale: float, context: Dict[str, Any]) -> None:
        if not _is_number(scale):
            return
        scale_val = float(scale)
        if scale_val <= 0:
            return
        if abs(scale_val - 1.0) < 1e-9:
            return

        for key in ("position_size", "notional", "pnl"):
            current = _get_field(trade, key)
            if _is_number(current):
                _set_field(trade, key, float(current) * scale_val)

        leverage = _get_field(trade, "leverage")
        if _is_number(leverage):
            new_leverage = float(leverage) * scale_val
            max_leverage = context.get("max_leverage")
            if _is_number(max_leverage):
                new_leverage = min(new_leverage, float(max_leverage))
            _set_field(trade, "leverage", new_leverage)

    def _breakout_allowed(self, context: Dict[str, Any]) -> bool:
        route = context.get("regime_route")
        if isinstance(route, str):
            route_norm = route.strip().lower()
            if route_norm in {"breakout", "breakout_vol_expansion"}:
                return True
        method = context.get("sl_tp_method")
        if isinstance(method, str) and "breakout" in method.lower():
            return True
        if context.get("breakout_confirmed") is True:
            return True
        regime_conf = context.get("regime_confidence")
        model_conf = context.get("model_confidence")
        if _is_number(regime_conf) and _is_number(model_conf):
            return (
                float(regime_conf) >= self.config.stress_breakout_regime_conf
                and float(model_conf) >= self.config.stress_breakout_model_conf
            )
        return False

    def _has_mean_reversion_tag(self, context: Dict[str, Any]) -> bool:
        route = context.get("regime_route")
        if isinstance(route, str) and route.strip().lower() in {"mean_reversion", "mr"}:
            return True
        tag = context.get("strategy_tag")
        if isinstance(tag, str) and tag.strip().lower() in {"mean_reversion", "mr"}:
            return True
        return False


__all__ = ["OverlayConfig", "RegimeRiskOverlay"]
