"""
Backtest module for trade generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import asyncio
import importlib.util
import inspect
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except Exception:
    httpx = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    _load_dotenv = None

# -----------------------------
# Env loading
# -----------------------------

def _load_env(env_file: Optional[Path]) -> None:
    path = env_file or Path(".env")
    if _load_dotenv is not None:
        _load_dotenv(dotenv_path=path, override=False)
        return
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

@dataclass
class TradeSetup:
    direction: str  # "long" or "short"
    entry: float
    stop: float
    take_profits: List[float]
    confidence: Optional[float] = None
    horizon_bars: Optional[int] = None
    reason: Optional[str] = None

@dataclass
class TradeResult:
    as_of: datetime
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry: float
    stop: float
    take_profit: float
    exit_price: float
    pnl: float
    r_multiple: Optional[float]
    outcome: str  # "tp", "sl", "timeout"
    bars_held: int
    confidence: Optional[float] = None

# -----------------------------
# Utilities
# -----------------------------

TIMEFRAME_MINUTES = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}

def _parse_timeframe(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_MINUTES[tf]

def _parse_iso_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        raise ValueError(f"Invalid ISO datetime: {value}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _parse_horizons(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            raise ValueError(f"Invalid horizon: {p}")
    if not out:
        raise ValueError("Empty horizons list.")
    return out

def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, data: dict) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)

def _extract_first_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    text = text.strip()
    obj, _ = decoder.raw_decode(text)
    return obj

def _strip_json_fences(text: str) -> str:
    import re
    text = re.sub(r"```json\s*(\{.*?\})\s*```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"```\s*(\{.*?\})\s*```", r"\1", text, flags=re.DOTALL)
    return text.strip()

# -----------------------------
# Local app loading (optional)
# -----------------------------

_LOCAL_APP_CACHE: Dict[str, Any] = {}
_LOCAL_ENGINE_CACHE: Dict[str, Any] = {}

def _resolve_local_app_path(local_app_path: Optional[str]) -> Path:
    if local_app_path:
        p = Path(local_app_path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p.resolve()
        alt = Path(__file__).resolve().parent / local_app_path
        if alt.exists():
            return alt
    default_path = Path(__file__).resolve().parent / "alphalens_lambda" / "app.py"
    return default_path

def _load_local_app_module(local_app_path: Optional[str]) -> Any:
    key = local_app_path or ""
    if key in _LOCAL_APP_CACHE:
        return _LOCAL_APP_CACHE[key]
    path = _resolve_local_app_path(local_app_path)
    if not path.exists():
        raise RuntimeError(f"Local app.py not found at {path}")
    spec = importlib.util.spec_from_file_location("local_app_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOCAL_APP_CACHE[key] = module
    return module

def _resolve_local_engine_path(local_engine_path: Optional[str]) -> Path:
    if local_engine_path:
        p = Path(local_engine_path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p.resolve()
        alt = Path(__file__).resolve().parent / local_engine_path
        if alt.exists():
            return alt
    default_root = Path("C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation")
    if default_root.exists():
        return default_root
    return Path(__file__).resolve().parent

def _load_local_engine_module(local_engine_path: Optional[str]) -> Any:
    key = local_engine_path or ""
    if key in _LOCAL_ENGINE_CACHE:
        return _LOCAL_ENGINE_CACHE[key]
    base = _resolve_local_engine_path(local_engine_path)
    path = base
    if path.is_dir():
        path = path / "inference_api.py"
    if not path.exists():
        raise RuntimeError(f"Local engine inference_api.py not found at {path}")
    spec = importlib.util.spec_from_file_location("local_engine_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOCAL_ENGINE_CACHE[key] = module
    return module

def forecast_local_engine(
    symbol: str,
    timeframe: str,
    horizons: List[int],
    trade_mode: str = "forward",
    local_engine_path: Optional[str] = None,
) -> Dict[str, Any]:
    module = _load_local_engine_module(local_engine_path)
    handler = getattr(module, "handle_forecast", None)
    if handler is None:
        raise RuntimeError("handle_forecast not found in local engine module.")
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": horizons,
        "trade_mode": trade_mode,
        "use_montecarlo": True,
        "include_predictions": True,
        "include_metadata": True,
        "include_model_info": True,
        "paths": 1000,
    }
    start = time.time()
    status_code, response = handler(payload, request_id=str(uuid.uuid4()), debug=False)
    elapsed = time.time() - start
    if status_code >= 400:
        return {
            "status": "error",
            "error_type": "http_error",
            "http_status": status_code,
            "message": "Local engine returned error",
            "response_text": json.dumps(response, ensure_ascii=True),
            "elapsed_sec": round(elapsed, 2),
        }
    return {"status": "success", "elapsed_sec": round(elapsed, 2), "data": response}
def forecast_local_app(
    symbol: str,
    timeframe: str,
    horizons: List[int],
    trade_mode: str = "forward",
    local_app_path: Optional[str] = None,
) -> Dict[str, Any]:
    module = _load_local_app_module(local_app_path)
    tool_obj = getattr(module, "forecast_aws_api", None)
    if tool_obj is None:
        raise RuntimeError("forecast_aws_api not found in local app module.")
    kwargs = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": horizons,
        "trade_mode": trade_mode,
    }
    if hasattr(tool_obj, "coroutine") and callable(getattr(tool_obj, "coroutine")):
        return asyncio.run(tool_obj.coroutine(**kwargs))
    if hasattr(tool_obj, "ainvoke") and callable(getattr(tool_obj, "ainvoke")):
        return asyncio.run(tool_obj.ainvoke(kwargs))
    if inspect.iscoroutinefunction(tool_obj):
        return asyncio.run(tool_obj(**kwargs))
    if callable(tool_obj):
        return tool_obj(**kwargs)
    raise RuntimeError("Unsupported forecast_aws_api tool interface in local app.")

# -----------------------------
# Data loading
# -----------------------------

def _parse_ts(value: str) -> datetime:
    raw = value.strip()
    if raw.isdigit():
        num = int(raw)
        if num > 10_000_000_000:
            num = num / 1000.0
        return datetime.fromtimestamp(num, tz=timezone.utc)
    return _parse_iso_dt(raw)

def load_bars_from_csv(path: Path) -> List[Bar]:
    bars: List[Bar] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp") or row.get("time") or row.get("date")
            if not ts:
                raise ValueError("CSV must include timestamp/time/date column.")
            bar = Bar(
                ts=_parse_ts(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if row.get("volume") else None,
            )
            bars.append(bar)
    bars.sort(key=lambda b: b.ts)
    return bars

def fetch_twelvedata_bars(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    api_key: str,
    outputsize: int = 5000,
) -> List[Bar]:
    if httpx is None:
        raise RuntimeError("httpx is required for Twelve Data fetch.")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": timeframe,
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "apikey": api_key,
        "format": "JSON",
        "outputsize": outputsize,
    }
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    values = data.get("values") or []
    bars: List[Bar] = []
    for v in values:
        ts = _parse_iso_dt(v["datetime"])
        bars.append(
            Bar(
                ts=ts,
                open=float(v["open"]),
                high=float(v["high"]),
                low=float(v["low"]),
                close=float(v["close"]),
                volume=float(v["volume"]) if v.get("volume") else None,
            )
        )
    bars.sort(key=lambda b: b.ts)
    return bars

# -----------------------------
# Forecast + Surface APIs
# -----------------------------

def forecast_aws_api(
    symbol: str,
    timeframe: str,
    horizons: List[int],
    trade_mode: str = "forward",
    forecast_url: Optional[str] = None,
) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is required for forecast_aws_api.")
    url = forecast_url or "https://jqrlegdulnnrpiixiecf.supabase.co/functions/v1/forecast-proxy"

    if not symbol or not isinstance(symbol, str):
        return {"status": "error", "error_type": "validation_error", "message": "Invalid symbol"}
    if not timeframe or not isinstance(timeframe, str):
        return {"status": "error", "error_type": "validation_error", "message": "Invalid timeframe"}
    if not isinstance(horizons, list) or not horizons:
        return {"status": "error", "error_type": "validation_error", "message": "Invalid horizons"}

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": horizons,
        "trade_mode": trade_mode,
        "use_montecarlo": True,
        "include_predictions": True,
        "include_metadata": True,
        "include_model_info": True,
        "paths": 1000,
    }

    start = time.time()
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(url, json=payload)
        elapsed = time.time() - start
        if response.status_code >= 400:
            return {
                "status": "error",
                "error_type": "http_error",
                "http_status": response.status_code,
                "message": "Forecast API returned error",
                "response_text": response.text,
                "elapsed_sec": round(elapsed, 2),
            }
        try:
            data = response.json()
        except Exception:
            return {
                "status": "error",
                "error_type": "invalid_json",
                "message": "Forecast API returned non-JSON response",
                "raw_response": response.text,
            }
        return {"status": "success", "elapsed_sec": round(elapsed, 2), "data": data}
    except Exception as e:
        return {
            "status": "error",
            "error_type": "network_error",
            "message": str(e),
        }

def _forecast_call(
    req: Dict[str, Any],
    backend: str,
    local_app_path: Optional[str],
    local_engine_path: Optional[str],
) -> Dict[str, Any]:
    if backend == "local_engine":
        return forecast_local_engine(
            symbol=req["symbol"],
            timeframe=req["timeframe"],
            horizons=req["horizons"],
            trade_mode=req.get("trade_mode", "forward"),
            local_engine_path=local_engine_path,
        )
    if backend == "local_app":
        return forecast_local_app(
            symbol=req["symbol"],
            timeframe=req["timeframe"],
            horizons=req["horizons"],
            trade_mode=req.get("trade_mode", "forward"),
            local_app_path=local_app_path,
        )
    return forecast_aws_api(**req)

def forecast_aws_api_batch(
    requests: List[Dict[str, Any]],
    batch_size: int,
    backend: str,
    local_app_path: Optional[str],
    local_engine_path: Optional[str],
) -> List[Dict[str, Any]]:
    if not requests:
        return []
    if batch_size <= 1:
        return [_forecast_call(req, backend, local_app_path, local_engine_path) for req in requests]
    workers = min(batch_size, len(requests))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda r: _forecast_call(r, backend, local_app_path, local_engine_path), requests))

def surface_probability_aws_api(
    symbol: str,
    timeframe: str,
    direction: str,
    methodology: str,
    target_prob_min: float,
    target_prob_max: float,
    target_prob_steps: int,
    sl_sigma_min: float,
    sl_sigma_max: float,
    sl_sigma_steps: int,
    horizon_hours: Optional[float] = None,
    steps: Optional[int] = None,
    entry_price: Optional[float] = None,
    paths: int = 3000,
    dof: float = 3.0,
    surface_url: Optional[str] = None,
) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is required for surface_probability_aws_api.")
    if horizon_hours is not None and steps is not None:
        return {"status": "error", "error_type": "invalid_request", "message": "Provide horizon_hours OR steps."}
    if not (0.0 < target_prob_min < target_prob_max < 1.0):
        return {"status": "error", "error_type": "invalid_target_prob", "message": "target_prob must be between 0 and 1"}
    if not (sl_sigma_min > 0 and sl_sigma_max > sl_sigma_min):
        return {"status": "error", "error_type": "invalid_sl_sigma", "message": "sl_sigma must be positive"}

    url = surface_url or "https://jqrlegdulnnrpiixiecf.supabase.co/functions/v1/surface-proxy"
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "methodology": methodology,
        "paths": paths,
        "dof": dof,
        "entry_price": entry_price,
        "horizon_hours": horizon_hours,
        "steps": steps,
        "target_prob": {"min": target_prob_min, "max": target_prob_max, "steps": target_prob_steps},
        "sl_sigma": {"min": sl_sigma_min, "max": sl_sigma_max, "steps": sl_sigma_steps},
    }

    start = time.time()
    try:
        with httpx.Client(timeout=20) as client:
            r = client.post(url, json=payload)
        elapsed = time.time() - start
        if r.status_code >= 400:
            return {
                "status": "error",
                "error_type": "http_error",
                "http_status": r.status_code,
                "message": "Surface API returned error",
                "response_text": r.text,
                "elapsed_sec": round(elapsed, 2),
            }
        return {"status": "ok", "elapsed_sec": round(elapsed, 2), "surface": r.json()}
    except Exception as e:
        return {"status": "error", "error_type": "network_error", "message": str(e)}

# -----------------------------
# Macro / News
# -----------------------------

def fetch_finnhub_news_last_30d(token: str, as_of: datetime) -> List[Dict[str, Any]]:
    if httpx is None:
        raise RuntimeError("httpx is required for Finnhub fetch.")
    url = "https://finnhub.io/api/v1/news"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params={"category": "general", "token": token})
        r.raise_for_status()
        data = r.json()
    cutoff = int((as_of - timedelta(days=30)).timestamp())
    out: List[Dict[str, Any]] = []
    for a in data:
        ts = a.get("datetime")
        if not isinstance(ts, int):
            continue
        if ts < cutoff or ts > int(as_of.timestamp()):
            continue
        out.append(
            {
                "headline": a.get("headline"),
                "summary": a.get("summary"),
                "source": a.get("source", "Unknown"),
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat(),
            }
        )
    return out

def fetch_finnhub_econ_calendar(token: str, as_of: datetime, countries: List[str]) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is required for Finnhub econ calendar.")
    url = "https://finnhub.io/api/v1/calendar/economic"
    from_d = (as_of - timedelta(days=7)).strftime("%Y-%m-%d")
    to_d = (as_of + timedelta(days=7)).strftime("%Y-%m-%d")
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params={"token": token, "from": from_d, "to": to_d})
        r.raise_for_status()
        calendar = r.json().get("economicCalendar", [])
    filtered = [
        {
            "event": ev.get("event", "Unavailable"),
            "country": ev.get("country", "Unavailable"),
            "time": ev.get("time", "Unavailable"),
            "actual": ev.get("actual"),
            "consensus": ev.get("estimate"),
            "previous": ev.get("prev"),
            "impact": (ev.get("impact") or "Unavailable").lower(),
        }
        for ev in calendar
        if ev.get("country") in countries
    ][:20]
    return {"economic_events": filtered or "Unavailable", "meta": {"from": from_d, "to": to_d}}

def detect_countries_from_symbol(symbol: str) -> List[str]:
    ccy_to_country = {
        "USD": "US", "EUR": "EU", "GBP": "UK", "JPY": "JP",
        "CHF": "CH", "AUD": "AU", "CAD": "CA", "NZD": "NZ", "CNY": "CN",
    }
    out: List[str] = []
    if "/" in symbol:
        a, b = symbol.split("/", 1)
        if a in ccy_to_country:
            out.append(ccy_to_country[a])
        if b in ccy_to_country:
            out.append(ccy_to_country[b])
    if not out:
        out = ["US", "EU"]
    return list(dict.fromkeys(out))

def fetch_abcg_research(query: str, abcg_url: Optional[str]) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is required for ABCG research.")
    url = abcg_url or "https://sfceyst3pu6ib35hqlh4xplbdy0repmb.lambda-url.us-east-2.on.aws/rag/query"
    payload = {"query": query, "topk": 1, "alpha": 0.2, "beta": 0.0, "gamma": 0.8, "tau_days": 14}
    with httpx.Client(timeout=45) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# -----------------------------
# Trade generation
# -----------------------------

def compute_market_features(bars: List[Bar], window: int = 50) -> Dict[str, Any]:
    if not bars:
        return {}
    window = min(window, len(bars))
    recent = bars[-window:]
    closes = [b.close for b in recent]
    highs = [b.high for b in recent]
    lows = [b.low for b in recent]
    last = recent[-1]
    sma = sum(closes) / len(closes)
    atr = 0.0
    for i in range(1, len(recent)):
        tr = max(
            recent[i].high - recent[i].low,
            abs(recent[i].high - recent[i - 1].close),
            abs(recent[i].low - recent[i - 1].close),
        )
        atr += tr
    atr = atr / max(1, len(recent) - 1)
    return {
        "last_close": last.close,
        "last_open": last.open,
        "last_high": last.high,
        "last_low": last.low,
        "sma": sma,
        "atr": atr,
        "recent_high": max(highs),
        "recent_low": min(lows),
    }

def infer_direction_from_forecast(forecast_data: Dict[str, Any]) -> Optional[str]:
    if not forecast_data:
        return None
    data = forecast_data.get("data") or forecast_data
    for key in ("direction", "bias", "signal"):
        val = data.get(key)
        if isinstance(val, str):
            v = val.lower()
            if "long" in v or "bull" in v:
                return "long"
            if "short" in v or "bear" in v:
                return "short"
    return None

def simple_trade_from_market(bars: List[Bar], forecast_data: Optional[Dict[str, Any]]) -> Optional[TradeSetup]:
    if not bars:
        return None
    feat = compute_market_features(bars)
    last = feat.get("last_close")
    atr = feat.get("atr") or 0.0
    if atr <= 0:
        return None
    direction = infer_direction_from_forecast(forecast_data or {}) or ("long" if last >= feat.get("sma", last) else "short")
    if direction == "long":
        entry = last
        stop = entry - 1.5 * atr
        tp = entry + 3.0 * atr
    else:
        entry = last
        stop = entry + 1.5 * atr
        tp = entry - 3.0 * atr
    return TradeSetup(direction=direction, entry=entry, stop=stop, take_profits=[tp], confidence=None)

def _normalize_risk_appetite(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip().lower()
    if "aggr" in v or v in {"high", "aggressive"}:
        return "aggressive"
    if "mod" in v or v in {"medium", "moderate"}:
        return "moderate"
    if "cons" in v or v in {"low", "conservative"}:
        return "conservative"
    return None

RISK_PROFILE_MULTIPLIERS = {
    "conservative": {"sl_mult": 1.0, "tp_mult": 1.5, "prob": 0.80},
    "moderate": {"sl_mult": 2.25, "tp_mult": 2.5, "prob": 0.62},
    "aggressive": {"sl_mult": 5.0, "tp_mult": 3.5, "prob": 0.40},
}

def _extract_forecast_payload(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    if not forecast_data:
        return {}
    root = forecast_data
    if isinstance(root.get("data"), dict):
        root = root["data"]
    if isinstance(root.get("data"), dict) and isinstance(root["data"].get("payload"), dict):
        return root["data"]["payload"]
    if isinstance(root.get("payload"), dict):
        return root["payload"]
    return root if isinstance(root, dict) else {}

def _extract_forecast_horizons(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    for key in ("horizons", "trades", "trade_setups", "tradeSetups"):
        val = payload.get(key)
        if isinstance(val, list):
            return [v for v in val if isinstance(v, dict)]
        if isinstance(val, dict):
            out: List[Dict[str, Any]] = []
            for ra_key, item in val.items():
                if isinstance(item, dict):
                    h = dict(item)
                    h.setdefault("risk_appetite", ra_key)
                    out.append(h)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            h = dict(sub)
                            h.setdefault("risk_appetite", ra_key)
                            out.append(h)
            if out:
                return out
    return []

def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

def _extract_entry_fallback(forecast_data: Dict[str, Any]) -> Optional[float]:
    payload = _extract_forecast_payload(forecast_data)
    for key in ("entry_price", "entryPrice", "execution_price", "executionPrice", "last_price", "lastPrice"):
        val = _to_float(payload.get(key))
        if val is not None:
            return val
    data = forecast_data.get("data") if isinstance(forecast_data, dict) else None
    if isinstance(data, dict):
        meta = data.get("metadata") or {}
        if isinstance(meta, dict):
            for key in ("execution_price", "executionPrice", "last_price", "lastPrice"):
                val = _to_float(meta.get(key))
                if val is not None:
                    return val
    return None

def _extract_sigma_from_forecast(forecast_data: Dict[str, Any]) -> Optional[float]:
    if not isinstance(forecast_data, dict):
        return None
    data = forecast_data.get("data") or {}
    if isinstance(data, dict):
        meta = data.get("metadata") or {}
        if isinstance(meta, dict):
            vol = meta.get("vol_model") or {}
            if isinstance(vol, dict):
                sigma = _to_float(vol.get("sigma_last") or vol.get("sigma"))
                if sigma:
                    return sigma
            sigma = _to_float(meta.get("sigma_path_max") or meta.get("sigma_path_min"))
            if sigma:
                return sigma
            sigma = _to_float(meta.get("residual_std"))
            if sigma:
                return sigma
    payload = _extract_forecast_payload(forecast_data)
    sigma = _to_float(payload.get("sigma_h") or payload.get("sigma") or payload.get("vol"))
    return sigma

def _asset_class_from_symbol(symbol: str) -> str:
    s = (symbol or "").upper()
    if "/" in s:
        base, quote = s.split("/", 1)
        if base in {"XAU", "XAG", "WTI", "BRENT"}:
            return "commodity"
        if base in {"BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "BNB"} or quote in {"BTC", "ETH"}:
            return "crypto"
        if len(base) == 3 and len(quote) == 3:
            return "fx"
    return "other"

def _friction_sigma(timeframe: str, symbol: str) -> float:
    minutes = _parse_timeframe(timeframe)
    asset = _asset_class_from_symbol(symbol)
    base_by_asset = {
        "fx": 1.0,
        "crypto": 1.5,
        "commodity": 1.0,
        "other": 1.0,
    }
    base = base_by_asset.get(asset, 1.0)
    scale = (15.0 / max(1.0, float(minutes))) ** 0.5
    return base * scale

def _parse_horizon_bars(value: Any, timeframe: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw.isdigit():
            return int(raw)
        if raw.endswith("h"):
            try:
                hours = float(raw[:-1])
            except Exception:
                return None
            minutes = _parse_timeframe(timeframe)
            return int(round(hours * 60.0 / minutes))
        if raw.endswith("d"):
            try:
                days = float(raw[:-1])
            except Exception:
                return None
            minutes = _parse_timeframe(timeframe)
            return int(round(days * 24.0 * 60.0 / minutes))
    return None

def _extract_trade_candidates(
    forecast_data: Dict[str, Any],
    timeframe: str,
) -> List[Dict[str, Any]]:
    payload = _extract_forecast_payload(forecast_data)
    horizons = _extract_forecast_horizons(payload)
    entry_fallback = _extract_entry_fallback(forecast_data)
    payload_risk = _normalize_risk_appetite(
        payload.get("risk_appetite")
        or payload.get("riskAppetite")
        or payload.get("risk_profile")
        or payload.get("riskProfile")
    )
    candidates: List[Dict[str, Any]] = []
    for h in horizons:
        direction = (h.get("direction") or h.get("bias") or h.get("signal") or "").lower()
        if direction not in ("long", "short"):
            continue
        entry = _to_float(h.get("entry_price") or h.get("entry") or h.get("entryPrice")) or entry_fallback
        if entry is None:
            continue
        confidence = _to_float(h.get("confidence") or h.get("prob_hit_tp_before_sl"))
        horizon_bars = _parse_horizon_bars(h.get("h") or h.get("horizon"), timeframe)
        risk_appetite = _normalize_risk_appetite(
            h.get("risk_appetite")
            or h.get("riskAppetite")
            or h.get("risk_profile")
            or h.get("riskProfile")
            or payload_risk
        )
        candidates.append(
            {
                "direction": direction,
                "entry": entry,
                "risk_appetite": risk_appetite,
                "horizon_bars": horizon_bars,
                "confidence": confidence,
            }
        )
    return candidates

def _select_candidate_by_horizon(
    candidates: List[Dict[str, Any]],
    risk_appetite: str,
) -> Optional[Dict[str, Any]]:
    with_h = [c for c in candidates if c.get("horizon_bars") is not None]
    if not with_h:
        return None
    ordered = sorted(with_h, key=lambda c: c["horizon_bars"])
    if risk_appetite == "aggressive":
        return ordered[0]
    if risk_appetite == "conservative":
        return ordered[-1]
    # moderate -> middle / upper-middle
    return ordered[len(ordered) // 2]

def _select_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    def score(c: Dict[str, Any]) -> float:
        conf = c.get("confidence")
        if conf is not None:
            return float(conf)
        hb = c.get("horizon_bars")
        return -float(hb) if hb is not None else 0.0
    return max(candidates, key=score)

def _find_limit_entry_index(
    bars: List[Bar],
    start_idx: int,
    entry_price: float,
    expiry_bars: int,
) -> Optional[int]:
    if expiry_bars < 1:
        expiry_bars = 1
    end_idx = min(start_idx + expiry_bars, len(bars) - 1)
    for idx in range(start_idx + 1, end_idx + 1):
        b = bars[idx]
        if b.low <= entry_price <= b.high:
            return idx
    return None

def _build_trade_from_candidate(
    candidate: Dict[str, Any],
    symbol: str,
    timeframe: str,
    risk_appetite: str,
    market_features: Dict[str, Any],
    forecast_data: Dict[str, Any],
) -> Optional[TradeSetup]:
    direction = candidate.get("direction")
    entry = candidate.get("entry")
    if direction not in ("long", "short") or entry is None:
        return None

    atr = _to_float((market_features or {}).get("atr"))
    method = "atr"
    unit = atr if atr and atr > 0 else None
    if unit is None:
        sigma = _extract_sigma_from_forecast(forecast_data)
        if sigma is None or sigma <= 0:
            return None
        method = "sigma"
        unit = float(entry) * float(sigma)

    ra = _normalize_risk_appetite(risk_appetite) or "moderate"
    profile = RISK_PROFILE_MULTIPLIERS.get(ra) or RISK_PROFILE_MULTIPLIERS["moderate"]

    friction = _friction_sigma(timeframe, symbol)
    sl_mult = profile["sl_mult"] + 0.5 * friction
    tp_mult = profile["tp_mult"] + 0.5 * friction

    if direction == "long":
        stop = entry - sl_mult * unit
        tp = entry + tp_mult * unit
    else:
        stop = entry + sl_mult * unit
        tp = entry - tp_mult * unit
    if stop <= 0 or tp <= 0:
        return None

    reason = f"risk_appetite={ra}; method={method}; friction_sigma={round(friction,4)}"
    return TradeSetup(
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        take_profits=[float(tp)],
        confidence=candidate.get("confidence"),
        horizon_bars=candidate.get("horizon_bars"),
        reason=reason,
    )

def trade_from_forecast(
    forecast_data: Dict[str, Any],
    symbol: str,
    timeframe: str,
    risk_appetite: str,
    market_features: Dict[str, Any],
) -> Optional[TradeSetup]:
    candidates = _extract_trade_candidates(forecast_data, timeframe)
    if not candidates:
        return None
    if isinstance(risk_appetite, str) and risk_appetite.strip().lower() == "auto":
        base = _select_best_candidate(candidates)
        if base is None:
            return None
        ra = _normalize_risk_appetite(base.get("risk_appetite")) or "moderate"
        return _build_trade_from_candidate(base, symbol, timeframe, ra, market_features, forecast_data)
    ra = _normalize_risk_appetite(risk_appetite) or "moderate"
    matched = [c for c in candidates if c.get("risk_appetite") == ra]
    if matched:
        base = _select_best_candidate(matched)
        return _build_trade_from_candidate(base, symbol, timeframe, ra, market_features, forecast_data)
    picked = _select_candidate_by_horizon(candidates, ra)
    if picked:
        return _build_trade_from_candidate(picked, symbol, timeframe, ra, market_features, forecast_data)
    base = _select_best_candidate(candidates)
    return _build_trade_from_candidate(base, symbol, timeframe, ra, market_features, forecast_data)

def llm_generate_trade(
    symbol: str,
    timeframe: str,
    as_of: datetime,
    forecast_data: Optional[Dict[str, Any]],
    market_features: Dict[str, Any],
    macro_context: Optional[Dict[str, Any]],
    model: str,
    temperature: float,
) -> Optional[TradeSetup]:
    if ChatOpenAI is None or SystemMessage is None:
        raise RuntimeError("langchain_openai is required for LLM trade generation.")
    system = """
You are an institutional trade setup generator.
Return ONLY valid JSON with the schema:
{
  "direction": "long" | "short",
  "entry": number,
  "stop": number,
  "take_profits": [number, ...],
  "confidence": number,
  "horizon_bars": number,
  "reason": string
}
Rules:
- No markdown. No text outside JSON.
- If you cannot build a valid trade, return {"skip": true, "reason": "..."}.
"""
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "as_of": as_of.isoformat(),
        "forecast_data": forecast_data or {},
        "market_features": market_features or {},
        "macro_context": macro_context or {},
    }
    user = f"Context:\\n{json.dumps(payload, ensure_ascii=True)}"
    llm = ChatOpenAI(model=model, temperature=temperature)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = _strip_json_fences(resp.content or "")
    if not text:
        return None
    try:
        obj = _extract_first_json(text)
    except Exception:
        return None
    if obj.get("skip"):
        return None
    try:
        direction = str(obj["direction"]).lower()
        entry = float(obj["entry"])
        stop = float(obj["stop"])
        tps = [float(x) for x in obj.get("take_profits", []) if x is not None]
        conf = obj.get("confidence")
        conf_val = float(conf) if conf is not None else None
        horizon = obj.get("horizon_bars")
        horizon_val = int(horizon) if horizon is not None else None
        reason = obj.get("reason")
    except Exception:
        return None
    if direction not in ("long", "short") or not tps:
        return None
    return TradeSetup(
        direction=direction,
        entry=entry,
        stop=stop,
        take_profits=tps,
        confidence=conf_val,
        horizon_bars=horizon_val,
        reason=reason,
    )

# -----------------------------
# Backtest core
# -----------------------------

def simulate_trade(
    bars: List[Bar],
    entry_idx: int,
    trade: TradeSetup,
    max_hold_bars: int,
    fill_policy: str,
    entry_mode: str = "next_open",
    entry_expiry_bars: int = 1,
    management: Optional[Dict[str, float]] = None,
) -> Optional[TradeResult]:
    if entry_idx + 1 >= len(bars):
        return None
    entry_bar = bars[entry_idx + 1]
    entry_fill_idx = entry_idx + 1
    if entry_mode in ("limit_next_bar", "limit_window"):
        if trade.entry is None:
            return None
        if entry_mode == "limit_next_bar":
            if not (entry_bar.low <= trade.entry <= entry_bar.high):
                return None
            entry_price = trade.entry
        else:
            fill_idx = _find_limit_entry_index(bars, entry_idx, trade.entry, entry_expiry_bars)
            if fill_idx is None:
                return None
            entry_fill_idx = fill_idx
            entry_bar = bars[entry_fill_idx]
            entry_price = trade.entry
    else:
        entry_price = entry_bar.open
    stop = trade.stop
    direction = trade.direction
    take_profit = trade.take_profits[0]
    r_dist = abs(entry_price - stop) if stop is not None else 0.0
    best_price = entry_price
    for i in range(entry_fill_idx, min(entry_fill_idx + max_hold_bars, len(bars))):
        b = bars[i]
        if direction == "long":
            hit_sl = b.low <= stop
            hit_tp = b.high >= take_profit
            if hit_sl and hit_tp:
                if fill_policy == "tp_first":
                    exit_price = take_profit
                    outcome = "tp"
                else:
                    exit_price = stop
                    outcome = "sl"
                pnl = exit_price - entry_price
                r_mult = pnl / (entry_price - stop) if entry_price != stop else None
                return TradeResult(
                    as_of=bars[entry_idx].ts,
                    entry_time=entry_bar.ts,
                    exit_time=b.ts,
                    direction=direction,
                    entry=entry_price,
                    stop=stop,
                    take_profit=take_profit,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_mult,
                    outcome=outcome,
                    bars_held=i - entry_fill_idx + 1,
                    confidence=trade.confidence,
                )
            if hit_tp:
                exit_price = take_profit
                pnl = exit_price - entry_price
                r_mult = pnl / (entry_price - stop) if entry_price != stop else None
                return TradeResult(
                    as_of=bars[entry_idx].ts,
                    entry_time=entry_bar.ts,
                    exit_time=b.ts,
                    direction=direction,
                    entry=entry_price,
                    stop=stop,
                    take_profit=take_profit,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_mult,
                    outcome="tp",
                    bars_held=i - entry_fill_idx + 1,
                    confidence=trade.confidence,
                )
            if hit_sl:
                exit_price = stop
                pnl = exit_price - entry_price
                r_mult = pnl / (entry_price - stop) if entry_price != stop else None
                return TradeResult(
                    as_of=bars[entry_idx].ts,
                    entry_time=entry_bar.ts,
                    exit_time=b.ts,
                    direction=direction,
                    entry=entry_price,
                    stop=stop,
                    take_profit=take_profit,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_mult,
                    outcome="sl",
                    bars_held=i - entry_fill_idx + 1,
                    confidence=trade.confidence,
                )
        else:
            hit_sl = b.high >= stop
            hit_tp = b.low <= take_profit
            if hit_sl and hit_tp:
                if fill_policy == "tp_first":
                    exit_price = take_profit
                    outcome = "tp"
                else:
                    exit_price = stop
                    outcome = "sl"
                pnl = entry_price - exit_price
                r_mult = pnl / (stop - entry_price) if entry_price != stop else None
                return TradeResult(
                    as_of=bars[entry_idx].ts,
                    entry_time=entry_bar.ts,
                    exit_time=b.ts,
                    direction=direction,
                    entry=entry_price,
                    stop=stop,
                    take_profit=take_profit,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_mult,
                    outcome=outcome,
                    bars_held=i - entry_fill_idx + 1,
                    confidence=trade.confidence,
                )
            if hit_tp:
                exit_price = take_profit
                pnl = entry_price - exit_price
                r_mult = pnl / (stop - entry_price) if entry_price != stop else None
                return TradeResult(
                    as_of=bars[entry_idx].ts,
                    entry_time=entry_bar.ts,
                    exit_time=b.ts,
                    direction=direction,
                    entry=entry_price,
                    stop=stop,
                    take_profit=take_profit,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_mult,
                    outcome="tp",
                    bars_held=i - entry_fill_idx + 1,
                    confidence=trade.confidence,
                )
            if hit_sl:
                exit_price = stop
                pnl = entry_price - exit_price
                r_mult = pnl / (stop - entry_price) if entry_price != stop else None
                return TradeResult(
                    as_of=bars[entry_idx].ts,
                    entry_time=entry_bar.ts,
                    exit_time=b.ts,
                    direction=direction,
                    entry=entry_price,
                    stop=stop,
                    take_profit=take_profit,
                    exit_price=exit_price,
                    pnl=pnl,
                    r_multiple=r_mult,
                    outcome="sl",
                    bars_held=i - entry_fill_idx + 1,
                    confidence=trade.confidence,
                )

        if management and r_dist > 0:
            breakeven_r = management.get("breakeven_r")
            trail_r = management.get("trail_r")
            if direction == "long":
                best_price = max(best_price, b.high)
                if breakeven_r is not None and (best_price - entry_price) >= breakeven_r * r_dist:
                    stop = max(stop, entry_price)
                if trail_r is not None:
                    trail_stop = best_price - trail_r * r_dist
                    if trail_stop > stop:
                        stop = trail_stop
            else:
                best_price = min(best_price, b.low)
                if breakeven_r is not None and (entry_price - best_price) >= breakeven_r * r_dist:
                    stop = min(stop, entry_price)
                if trail_r is not None:
                    trail_stop = best_price + trail_r * r_dist
                    if trail_stop < stop:
                        stop = trail_stop

    last_idx = min(entry_fill_idx + max_hold_bars, len(bars) - 1)
    last_bar = bars[last_idx]
    exit_price = last_bar.close
    pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    r_mult = None
    if direction == "long":
        r_mult = pnl / (entry_price - stop) if entry_price != stop else None
    else:
        r_mult = pnl / (stop - entry_price) if entry_price != stop else None
    return TradeResult(
        as_of=bars[entry_idx].ts,
        entry_time=entry_bar.ts,
        exit_time=last_bar.ts,
        direction=direction,
        entry=entry_price,
        stop=stop,
        take_profit=take_profit,
        exit_price=exit_price,
        pnl=pnl,
        r_multiple=r_mult,
        outcome="timeout",
        bars_held=last_idx - entry_fill_idx + 1,
        confidence=trade.confidence,
    )

def _management_for_risk_appetite(risk_appetite: str) -> Dict[str, float]:
    ra = _normalize_risk_appetite(risk_appetite) or "moderate"
    if ra == "aggressive":
        return {"breakeven_r": 0.5, "trail_r": 0.75}
    if ra == "conservative":
        return {"breakeven_r": 1.5, "trail_r": 1.5}
    return {"breakeven_r": 1.0, "trail_r": 1.0}

# -----------------------------
# Metrics
# -----------------------------

def summarize_results(trades: List[TradeResult]) -> Dict[str, Any]:
    if not trades:
        return {"trades": 0, "net_pnl": 0.0}
    net = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_r = None
    r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
    if r_vals:
        avg_r = sum(r_vals) / len(r_vals)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades),
        "net_pnl": net,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4) if avg_r is not None else None,
        "max_drawdown": round(max_dd, 4),
    }

def _serialize_trade_result(trade: TradeResult) -> Dict[str, Any]:
    data = asdict(trade)
    for key in ("as_of", "entry_time", "exit_time"):
        val = data.get(key)
        if isinstance(val, datetime):
            data[key] = val.isoformat()
    return data

# -----------------------------
# Cache helpers
# -----------------------------

def cache_path(cache_dir: Path, category: str, symbol: str, timeframe: str, as_of: datetime) -> Path:
    safe = _safe_symbol(symbol)
    ts = as_of.strftime("%Y%m%dT%H%M%SZ")
    return cache_dir / category / safe / timeframe / f"{ts}.json"

def _collect_decision_indices(
    start_idx: int,
    end_idx: int,
    warmup: int,
    decision_every: int,
    limit: int,
) -> List[int]:
    out: List[int] = []
    j = start_idx
    while j < end_idx and len(out) < limit:
        if (j - warmup) % decision_every == 0:
            out.append(j)
        j += 1
    return out

def _prefetch_forecasts(
    start_idx: int,
    end_idx: int,
    bars: List[Bar],
    warmup: int,
    decision_every: int,
    batch_size: int,
    symbol: str,
    timeframe: str,
    horizons: List[int],
    trade_mode: str,
    forecast_url: Optional[str],
    forecast_backend: str,
    local_app_path: Optional[str],
    local_engine_path: Optional[str],
    cache_dir: Path,
    mem_cache: Dict[int, Dict[str, Any]],
) -> None:
    if batch_size <= 1:
        return
    indices = _collect_decision_indices(start_idx, end_idx, warmup, decision_every, batch_size)
    if not indices:
        return
    to_fetch: List[int] = []
    requests: List[Dict[str, Any]] = []
    for idx in indices:
        if idx in mem_cache:
            continue
        as_of = bars[idx].ts
        f_cache = cache_path(cache_dir, "forecast", symbol, timeframe, as_of)
        if f_cache.exists():
            mem_cache[idx] = _read_json(f_cache)
            continue
        to_fetch.append(idx)
        requests.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "horizons": horizons,
                "trade_mode": trade_mode,
                "forecast_url": forecast_url,
            }
        )
    if not requests:
        return
    results = forecast_aws_api_batch(
        requests,
        batch_size=batch_size,
        backend=forecast_backend,
        local_app_path=local_app_path,
        local_engine_path=local_engine_path,
    )
    for idx, res in zip(to_fetch, results):
        mem_cache[idx] = res
        as_of = bars[idx].ts
        f_cache = cache_path(cache_dir, "forecast", symbol, timeframe, as_of)
        _write_json(f_cache, res)

# -----------------------------
# Main backtest
# -----------------------------

def run_backtest(args: argparse.Namespace) -> None:
    _load_env(Path(args.env_file) if args.env_file else None)
    symbol = args.symbol
    timeframe = args.timeframe
    _parse_timeframe(timeframe)

    cache_dir = Path(args.cache_dir)
    _ensure_dir(cache_dir)

    if args.price_csv:
        bars = load_bars_from_csv(Path(args.price_csv))
    else:
        api_key = args.twelve_data_key or os.environ.get("TWELVE_DATA_API_KEY")
        if not api_key:
            raise RuntimeError("TWELVE_DATA_API_KEY is required when price source is Twelve Data.")
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.lookback)
        bars = fetch_twelvedata_bars(symbol, timeframe, start, end, api_key)

    if len(bars) < 10:
        raise RuntimeError("Not enough bars for backtest.")

    decision_every = max(1, args.decision_every)
    warmup = max(5, args.warmup_bars)
    horizons = _parse_horizons(args.horizons)
    max_hold_bars = args.max_hold_bars or max(horizons)
    forecast_batch_size = max(1, args.forecast_batch_size)
    forecast_mem: Dict[int, Dict[str, Any]] = {}
    if args.gen_mode == "forecast" and args.forecast_mode == "off":
        raise RuntimeError("gen-mode=forecast requires forecast_mode=live or replay.")
    management = _management_for_risk_appetite(args.risk_appetite)
    entry_expiry_bars = max(1, int(args.entry_expiry_bars))
    forecast_backend = args.forecast_backend
    local_app_path = args.local_app_path
    local_engine_path = args.local_engine_path

    trades: List[TradeResult] = []
    i = warmup
    end_idx = len(bars) - 2
    while i < end_idx:
        if (i - warmup) % decision_every != 0:
            i += 1
            continue

        as_of = bars[i].ts

        forecast_data = None
        if args.forecast_mode != "off":
            f_cache = cache_path(cache_dir, "forecast", symbol, timeframe, as_of)
            if args.forecast_mode == "replay":
                forecast_data = _read_json(f_cache)
            else:
                _prefetch_forecasts(
                    start_idx=i,
                    end_idx=end_idx,
                    bars=bars,
                    warmup=warmup,
                    decision_every=decision_every,
                    batch_size=forecast_batch_size,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizons=horizons,
                    trade_mode=args.trade_mode,
                    forecast_url=args.forecast_url,
                    forecast_backend=forecast_backend,
                    local_app_path=local_app_path,
                    local_engine_path=local_engine_path,
                    cache_dir=cache_dir,
                    mem_cache=forecast_mem,
                )
                forecast_data = forecast_mem.pop(i, None)
                if forecast_data is None:
                    if forecast_backend == "local_engine":
                        forecast_data = forecast_local_engine(
                            symbol=symbol,
                            timeframe=timeframe,
                            horizons=horizons,
                            trade_mode=args.trade_mode,
                            local_engine_path=local_engine_path,
                        )
                    elif forecast_backend == "local_app":
                        forecast_data = forecast_local_app(
                            symbol=symbol,
                            timeframe=timeframe,
                            horizons=horizons,
                            trade_mode=args.trade_mode,
                            local_app_path=local_app_path,
                        )
                    else:
                        forecast_data = forecast_aws_api(
                            symbol=symbol,
                            timeframe=timeframe,
                            horizons=horizons,
                            trade_mode=args.trade_mode,
                            forecast_url=args.forecast_url,
                        )
                    _write_json(f_cache, forecast_data)

        macro_context = None
        if args.macro_mode != "off":
            m_cache = cache_path(cache_dir, "macro", symbol, timeframe, as_of)
            if args.macro_mode == "replay":
                macro_context = _read_json(m_cache)
            else:
                finnhub_token = args.finnhub_token or os.environ.get("FINNHUB_TOKEN")
                if not finnhub_token:
                    raise RuntimeError("FINNHUB_TOKEN is required for macro_mode=live.")
                countries = detect_countries_from_symbol(symbol)
                macro_context = {
                    "news": fetch_finnhub_news_last_30d(finnhub_token, as_of),
                    "economic_calendar": fetch_finnhub_econ_calendar(finnhub_token, as_of, countries),
                }
                if args.use_abcg:
                    macro_context["abcg_research"] = fetch_abcg_research(args.question, args.abcg_url)
                _write_json(m_cache, macro_context)

        market_features = compute_market_features(bars[: i + 1], window=args.feature_window)

        trade: Optional[TradeSetup] = None
        if args.gen_mode == "forecast":
            if forecast_data:
                trade = trade_from_forecast(
                    forecast_data=forecast_data,
                    symbol=symbol,
                    timeframe=timeframe,
                    risk_appetite=args.risk_appetite,
                    market_features=market_features,
                )
            else:
                trade = None
        elif args.gen_mode == "llm":
            trade = llm_generate_trade(
                symbol=symbol,
                timeframe=timeframe,
                as_of=as_of,
                forecast_data=forecast_data,
                market_features=market_features,
                macro_context=macro_context,
                model=args.llm_model,
                temperature=args.llm_temperature,
            )
        else:
            trade = simple_trade_from_market(bars[: i + 1], forecast_data)

        if trade is None:
            i += 1
            continue

        effective_max_hold = max_hold_bars
        if args.gen_mode == "forecast" and trade.horizon_bars and trade.horizon_bars > 0:
            effective_max_hold = min(max_hold_bars, trade.horizon_bars)

        result = simulate_trade(
            bars=bars,
            entry_idx=i,
            trade=trade,
            max_hold_bars=effective_max_hold,
            fill_policy=args.fill_policy,
            entry_mode="limit_window" if args.gen_mode == "forecast" else "next_open",
            entry_expiry_bars=(
                min(entry_expiry_bars, trade.horizon_bars)
                if args.gen_mode == "forecast" and trade.horizon_bars
                else entry_expiry_bars
            ),
            management=management if args.gen_mode == "forecast" else None,
        )

        if result:
            trades.append(result)
            if args.position_mode == "one_at_time":
                exit_idx = next(
                    (idx for idx, b in enumerate(bars) if b.ts == result.exit_time),
                    i + 1,
                )
                i = max(i + 1, exit_idx)
                continue
        i += 1

    summary = summarize_results(trades)

    print("\\nBacktest summary")
    print(json.dumps(summary, indent=2))

    if args.trades_json:
        out = Path(args.trades_json)
        _ensure_dir(out.parent)
        with out.open("w", encoding="utf-8") as f:
            json.dump([_serialize_trade_result(t) for t in trades], f, ensure_ascii=True, indent=2)
        print(f"Trades saved: {out}")

    if args.trades_csv:
        out = Path(args.trades_csv)
        _ensure_dir(out.parent)
        with out.open("w", encoding="utf-8", newline="") as f:
            if trades:
                writer = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
                writer.writeheader()
                for t in trades:
                    writer.writerow(asdict(t))
        print(f"Trades CSV saved: {out}")

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest module for trade generation.")
    p.add_argument("--env-file", default=".env", help="Path to .env file.")
    p.add_argument("--symbol", required=True, help="Instrument symbol, e.g. BTC/USD")
    p.add_argument("--timeframe", default="1h", help="Timeframe: 5m,15m,30m,1h,4h,1d,1w")
    p.add_argument("--lookback", type=int, default=120, help="Lookback days for price data (if not using CSV).")
    p.add_argument("--price-csv", default="", help="Optional CSV with OHLCV data.")
    p.add_argument("--twelve-data-key", default="", help="Override Twelve Data API key.")
    p.add_argument("--forecast-mode", choices=["live", "replay", "off"], default="live")
    p.add_argument(
        "--forecast-backend",
        choices=["remote", "local_app", "local_engine"],
        default="remote",
        help="Forecast backend for live mode: remote uses HTTP, local_app imports app.py tool, local_engine calls local inference_api.",
    )
    p.add_argument(
        "--local-app-path",
        default="",
        help="Path to local app.py (used with --forecast-backend local_app).",
    )
    p.add_argument(
        "--local-engine-path",
        default="",
        help="Path to local forecast engine repo or inference_api.py (used with --forecast-backend local_engine).",
    )
    p.add_argument("--forecast-url", default="", help="Override Forecast API URL.")
    p.add_argument("--trade-mode", default="forward", help="Forecast trade_mode value.")
    p.add_argument("--forecast-batch-size", type=int, default=1, help="Concurrent forecast requests per batch (live mode).")
    p.add_argument("--macro-mode", choices=["off", "live", "replay"], default="off")
    p.add_argument("--finnhub-token", default="", help="Override Finnhub token.")
    p.add_argument("--use-abcg", action="store_true", help="Include ABCG research in macro context.")
    p.add_argument("--abcg-url", default="", help="Override ABCG research URL.")
    p.add_argument("--question", default="Generate trade setup", help="Query for ABCG (if enabled).")
    p.add_argument("--gen-mode", choices=["llm", "simple", "forecast"], default="llm")
    p.add_argument(
        "--risk-appetite",
        choices=["auto", "aggressive", "moderate", "conservative"],
        default="auto",
        help="Risk appetite for selecting forecast trade levels.",
    )
    p.add_argument("--llm-model", default="gpt-4.1-mini", help="LLM model name.")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--horizons", default="12,24", help="Comma-separated horizons (integers).")
    p.add_argument("--max-hold-bars", type=int, default=0, help="Max bars to hold (0 uses max horizon).")
    p.add_argument(
        "--entry-expiry-bars",
        type=int,
        default=1,
        help="Bars to wait for limit entry in forecast mode (capped by trade horizon).",
    )
    p.add_argument("--decision-every", type=int, default=1, help="Generate a trade every N bars.")
    p.add_argument("--warmup-bars", type=int, default=50, help="Warmup bars before first decision.")
    p.add_argument("--feature-window", type=int, default=50, help="Market feature window.")
    p.add_argument("--position-mode", choices=["one_at_time", "overlap"], default="one_at_time")
    p.add_argument("--fill-policy", choices=["sl_first", "tp_first"], default="sl_first")
    p.add_argument("--cache-dir", default="backtest_cache", help="Cache directory for replay.")
    p.add_argument("--trades-json", default="backtest_trades.json", help="Output JSON file for trades.")
    p.add_argument("--trades-csv", default="", help="Output CSV file for trades.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    ns = parser.parse_args()
    try:
        run_backtest(ns)
    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        sys.exit(1)
