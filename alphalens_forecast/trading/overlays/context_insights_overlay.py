"""Context-insights overlay (pass-through by default).

This overlay is intentionally minimal so time-travel/local-engine imports
can succeed even when no insight signals are available. It provides a few
optional hooks if the context contains insight-derived hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


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
class ContextInsightsOverlayConfig:
    enabled: bool = True
    block_if_low_conf: bool = False
    min_confidence: float = 0.0


class ContextInsightsOverlay:
    """Optional overlay that reacts to context-provided insight hints."""

    def __init__(self, config: Optional[ContextInsightsOverlayConfig] = None) -> None:
        self.config = config or ContextInsightsOverlayConfig()

    def apply(self, trade: Any, context: Mapping[str, Any]) -> Any:
        if trade is None:
            return None
        if not self.config.enabled:
            return trade
        if not isinstance(context, Mapping):
            return trade

        # If the payload contains multiple horizons, apply per-horizon.
        if isinstance(trade, dict) and "horizons" in trade and isinstance(trade["horizons"], list):
            updated = dict(trade)
            updated_horizons = []
            for horizon in trade["horizons"]:
                if isinstance(horizon, dict):
                    updated_horizons.append(self._apply_trade(dict(horizon), context))
                else:
                    updated_horizons.append(horizon)
            updated["horizons"] = updated_horizons
            return updated

        if isinstance(trade, dict):
            return self._apply_trade(trade, context)

        return trade

    def _apply_trade(self, trade: Dict[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
        insights_enabled = bool(
            context.get("context_insights_enabled", False)
            or context.get("insights_enabled", False)
        )
        if not insights_enabled:
            return trade

        # Optional block if confidence is too low.
        if self.config.block_if_low_conf:
            conf = context.get("insights_confidence")
            if _is_number(conf) and float(conf) < float(self.config.min_confidence):
                return None

        # Optional size scaling if a hint is provided.
        scale = context.get("insights_size_scale")
        if _is_number(scale):
            scale_val = float(scale)
            if scale_val > 0 and abs(scale_val - 1.0) > 1e-9:
                for key in ("position_size", "notional", "pnl"):
                    current = _get_field(trade, key)
                    if _is_number(current):
                        _set_field(trade, key, float(current) * scale_val)

        # Optional tag for downstream visibility.
        if isinstance(trade, dict):
            trade.setdefault("overlay_tags", [])
            if isinstance(trade["overlay_tags"], list):
                trade["overlay_tags"].append("context_insights")

        return trade


__all__ = ["ContextInsightsOverlay", "ContextInsightsOverlayConfig"]
