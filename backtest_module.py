"""
Backtest module for trade generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import asyncio
import importlib.util
import inspect
import uuid
import tempfile
import re
from bisect import bisect_left, bisect_right
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    _load_dotenv = None

try:
    from alphalens_forecast.trading.overlays.regime_risk_overlay import (
        RegimeRiskOverlay,
        OverlayConfig,
    )
except Exception:
    RegimeRiskOverlay = None
    OverlayConfig = None

# -----------------------------
# Env loading
# -----------------------------

def _load_env(env_file: Optional[Path]) -> None:
    path = env_file or Path(".env")
    if _load_dotenv is not None:
        _load_dotenv(dotenv_path=path, override=False)
        return

# -----------------------------
# Date helpers
# -----------------------------

def _parse_date_arg(value: Optional[str], *, is_end: bool = False) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    # Date-only format: YYYY-MM-DD
    if "T" not in raw and " " not in raw and len(raw) <= 10:
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {value}") from exc
        if is_end:
            dt = datetime(dt.year, dt.month, dt.day, 23, 59, 59)
        else:
            dt = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    else:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid datetime format: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _filter_bars_by_date(
    bars: List["Bar"],
    start: Optional[datetime],
    end: Optional[datetime],
) -> List["Bar"]:
    if start is None and end is None:
        return bars
    start_ts = start
    end_ts = end
    if start_ts is not None and start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    if end_ts is not None and end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)
    filtered: List[Bar] = []
    for b in bars:
        ts = b.ts if b.ts.tzinfo is not None else b.ts.replace(tzinfo=timezone.utc)
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        filtered.append(b)
    return filtered
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
    entry_model: Optional[Dict[str, Any]] = None
    sl_tp_model: Optional[Dict[str, Any]] = None
    regime: Optional[Dict[str, Any]] = None

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
    position_size: Optional[float] = None
    notional: Optional[float] = None
    leverage: Optional[float] = None
    entry_model: Optional[Dict[str, Any]] = None
    sl_tp_model: Optional[Dict[str, Any]] = None
    regime: Optional[Dict[str, Any]] = None

# -----------------------------
# Utilities
# -----------------------------

TIMEFRAME_ALIASES = {
    "1m": "1min",
    "1min": "1min",
    "5m": "5min",
    "5min": "5min",
    "15m": "15min",
    "15min": "15min",
    "30m": "30min",
    "30min": "30min",
    "45m": "45min",
    "45min": "45min",
    "1h": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1day",
    "1day": "1day",
    "1w": "1week",
    "1week": "1week",
    "1wk": "1week",
}

TIMEFRAME_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "45min": 45,
    "1h": 60,
    "2h": 120,
    "3h": 180,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1day": 1440,
    "1week": 10080,
}

def _normalize_timeframe(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    normalized = TIMEFRAME_ALIASES.get(tf)
    if not normalized:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return normalized

def _parse_timeframe(timeframe: str) -> int:
    normalized = _normalize_timeframe(timeframe)
    return TIMEFRAME_MINUTES[normalized]

def _parse_iso_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        raise ValueError(f"Invalid ISO datetime: {value}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _parse_horizons(value: str) -> List[int]:
    if value is None:
        return []
    raw = str(value).strip().lower()
    if raw in {"", "auto", "default"}:
        return []
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            raise ValueError(f"Invalid horizon: {p}")
    return out

def _default_horizons_for_timeframe(timeframe: str) -> List[int]:
    tf = _normalize_timeframe(timeframe)
    mapping = {
        "15min": [12],
        "30min": [24],
        "1h": [24],
        "4h": [30],
    }
    return mapping.get(tf, [12, 24])

def _resolve_horizons(value: str, timeframe: str) -> List[int]:
    parsed = _parse_horizons(value)
    if parsed:
        return parsed
    return _default_horizons_for_timeframe(timeframe)

def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")

def _slugify_filename(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if not raw:
        return ""
    raw = raw.replace("/", "_").replace("\\", "_")
    raw = raw.replace(":", "")
    raw = raw.replace(" ", "")
    raw = re.sub(r"[^a-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw

def _default_trade_data_dir() -> Path:
    here = Path.cwd()
    direct = here / "trade_data"
    if direct.exists():
        return direct
    nested = here / "alphalens_lambda" / "trade_data"
    if nested.exists():
        return nested
    return direct

def _auto_trades_json_path(args: argparse.Namespace, base_dir: Optional[Path] = None) -> Path:
    base = base_dir or _default_trade_data_dir()
    parts: List[str] = ["trades"]
    parts.append(_slugify_filename(getattr(args, "symbol", "")))
    parts.append(_slugify_filename(getattr(args, "timeframe", "")))

    date_from = getattr(args, "date_from", None)
    date_to = getattr(args, "date_to", None)
    lookback = getattr(args, "lookback", None)
    if date_from or date_to:
        if date_from:
            parts.append(_slugify_filename(date_from))
        if date_to:
            parts.append(_slugify_filename(date_to))
    elif lookback:
        parts.append(f"lb{_slugify_filename(lookback)}")

    gen_mode = getattr(args, "gen_mode", None)
    forecast_mode = getattr(args, "forecast_mode", None)
    forecast_backend = getattr(args, "forecast_backend", None)
    risk_appetite = getattr(args, "risk_appetite", None)
    decision_every = getattr(args, "decision_every", None)
    if gen_mode:
        parts.append(_slugify_filename(gen_mode))
    if forecast_mode:
        parts.append(_slugify_filename(forecast_mode))
    if forecast_backend:
        parts.append(_slugify_filename(forecast_backend))
    if risk_appetite:
        parts.append(_slugify_filename(risk_appetite))
    if decision_every:
        parts.append(f"de{_slugify_filename(decision_every)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts.append(stamp)
    filename = "_".join([p for p in parts if p]) + ".json"
    return base / filename

def _resolve_trades_json_path(args: argparse.Namespace) -> Optional[Path]:
    raw = getattr(args, "trades_json", "")
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    if raw_str.lower() == "auto":
        return _auto_trades_json_path(args, None)
    path = Path(raw_str)
    if raw_str.endswith(("/", "\\")) or (path.exists() and path.is_dir()):
        return _auto_trades_json_path(args, path)
    return path

def _trades_json_flag_provided(argv: Optional[List[str]] = None) -> bool:
    args = argv if argv is not None else sys.argv
    for item in args:
        if item == "--trades-json" or item.startswith("--trades-json="):
            return True
    return False

def _maybe_autoname_trades_json(args: argparse.Namespace) -> None:
    raw = getattr(args, "trades_json", "")
    raw_str = "" if raw is None else str(raw).strip()
    if not _trades_json_flag_provided() and raw_str in {"", "backtest_trades.json"}:
        args.trades_json = str(_auto_trades_json_path(args, None))
        return
    resolved = _resolve_trades_json_path(args)
    args.trades_json = "" if resolved is None else str(resolved)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    for attempt in range(3):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                return None
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            _tqdm_write(f"[cache] invalid JSON in {path}: {exc}; treating as missing.")
            return None
        except OSError as exc:
            _tqdm_write(f"[cache] failed reading {path}: {exc}; treating as missing.")
            return None

def _write_json(path: Path, data: dict) -> None:
    _ensure_dir(path.parent)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            suffix=".tmp",
        ) as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
            tmp = Path(f.name)
        os.replace(tmp, path)
    finally:
        if tmp and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

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
    as_of: Optional[str] = None,
    asof_field: str = "as_of",
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
    if as_of:
        field = "asOf" if asof_field == "asOf" else "as_of"
        payload[field] = as_of
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
    as_of: Optional[str] = None,
    asof_field: str = "as_of",
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
    if as_of:
        kwargs["as_of"] = as_of
        kwargs["asof_field"] = asof_field
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

def _format_asof(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def _parse_forecast_asof(raw: str) -> tuple[str, Optional[datetime]]:
    if not raw:
        return "none", None
    token = raw.strip().lower()
    if token in {"bar", "bars", "auto"}:
        return "bar", None
    return "fixed", _parse_ts(raw)

def _resolve_forecast_asof(
    mode: str,
    bar_ts: datetime,
    fixed: Optional[datetime],
) -> Optional[str]:
    if mode == "bar":
        return _format_asof(bar_ts)
    if mode == "fixed" and fixed is not None:
        return _format_asof(fixed)
    return None

def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    return default

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default

def _maybe_tqdm(iterable, total: Optional[int], desc: str):
    if _tqdm is None:
        return iterable
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        mininterval=0.2,
        position=0,
    )

def _tqdm_write(message: str) -> None:
    if _tqdm is None:
        print(message)
    else:
        _tqdm.write(message)

def _parse_batch_bars(value: str) -> Optional[int]:
    if not value:
        return None
    raw = value.strip().lower()
    if raw.isdigit():
        return int(raw)
    for suffix in ("bars", "bar", "b"):
        if raw.endswith(suffix):
            num = raw[: -len(suffix)].strip()
            if num.isdigit():
                return int(num)
    return None

def _parse_batch_offset(value: str):
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas is required for date-based batch sizing.") from exc
    raw = value.strip()
    if not raw:
        return None
    try:
        return pd.tseries.frequencies.to_offset(raw)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid batch size '{value}'. Use e.g. 1M, 3M, 1Y, or N bars.") from exc

def _timestamp_list(bars: List[Bar]) -> List[datetime]:
    out: List[datetime] = []
    for b in bars:
        ts = b.ts if b.ts.tzinfo is not None else b.ts.replace(tzinfo=timezone.utc)
        out.append(ts)
    return out

def _build_bar_batches(
    total_bars: int,
    batch_bars: int,
    step_bars: Optional[int],
) -> List[Dict[str, int]]:
    batches: List[Dict[str, int]] = []
    if batch_bars <= 0:
        return batches
    step = step_bars or batch_bars
    if step <= 0:
        raise ValueError("batch step must be positive.")
    start = 0
    while start < total_bars:
        end = min(total_bars, start + batch_bars)
        if end > start:
            batches.append({"start_idx": start, "end_idx": end})
        start += step
    return batches

def _build_time_batches(
    timestamps: List[datetime],
    window_offset,
    step_offset,
) -> List[Dict[str, datetime]]:
    if not timestamps:
        return []
    start_ts = timestamps[0]
    end_ts = timestamps[-1]
    step = step_offset or window_offset
    if step is None:
        raise ValueError("batch step is required for rolling time batches.")
    batches: List[Dict[str, datetime]] = []
    current = start_ts
    guard = 0
    while current <= end_ts:
        window_end = current + window_offset
        if window_end < current:
            raise ValueError("batch window must be positive.")
        batch_end = min(window_end, end_ts)
        batches.append({"start_ts": current, "end_ts": batch_end})
        current = current + step
        guard += 1
        if guard > 100000:
            raise RuntimeError("Batch generation exceeded safety limit.")
    return batches

def _coerce_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        return _parse_ts(str(int(value)))
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        return _parse_ts(raw)
    return None

def _extract_forecast_asof_dt(forecast_data: Dict[str, Any]) -> Optional[datetime]:
    if not isinstance(forecast_data, dict):
        return None
    def _get(path: List[str]) -> Any:
        cur: Any = forecast_data
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur
    candidates = [
        ["data", "data", "as_of"],
        ["data", "data", "asOf"],
        ["data", "payload", "as_of"],
        ["data", "payload", "asOf"],
        ["data", "as_of"],
        ["data", "asOf"],
        ["payload", "as_of"],
        ["payload", "asOf"],
        ["as_of"],
        ["asOf"],
    ]
    for path in candidates:
        val = _get(path)
        dt = _coerce_dt(val)
        if dt is not None:
            return dt
    return None

def _extract_forecast_metadata(forecast_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(forecast_data, dict):
        return None
    candidates = [
        ["data", "metadata"],
        ["data", "data", "metadata"],
        ["metadata"],
    ]
    cur: Any
    for path in candidates:
        cur = forecast_data
        for key in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
        if isinstance(cur, dict):
            return cur
    return None


def _augment_regime_info(
    regime: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(regime, dict):
        return regime if isinstance(regime, dict) else None
    updated = dict(regime)
    mode = updated.get("mode") or updated.get("regime_mode")
    mode_used = updated.get("mode_used") or updated.get("regime_mode_used")
    detector = updated.get("detector") or updated.get("model")
    if not mode and isinstance(metadata, dict):
        mode = metadata.get("regime_mode") or metadata.get("regime_mode_used")
    if not mode_used and isinstance(metadata, dict):
        mode_used = metadata.get("regime_mode_used") or metadata.get("mode_used")
    if not mode:
        mode = os.environ.get("ALPHALENS_REGIME_MODE")
    if mode:
        updated["mode"] = mode
    if mode_used:
        updated["mode_used"] = mode_used
    if detector:
        updated["detector"] = detector
    elif updated.get("enabled"):
        updated["detector"] = "deterministic"
    return updated

def _get_regime_missing_log_path() -> Optional[Path]:
    raw = os.environ.get("ALPHALENS_REGIME_MISSING_LOG")
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.lower() in {"0", "false", "off", "none", "disable", "disabled"}:
        return None
    return Path(value)

def _append_regime_missing_log(record: Dict[str, Any]) -> None:
    path = _get_regime_missing_log_path()
    if path is None:
        return
    try:
        _ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to write regime missing log: {exc}")

def _configure_regime_missing_log(args: argparse.Namespace) -> None:
    raw = os.environ.get("ALPHALENS_REGIME_MISSING_LOG")
    if raw is not None:
        return
    runs_dir = Path(getattr(args, "runs_dir", "backtests_runs"))
    os.environ["ALPHALENS_REGIME_MISSING_LOG"] = str(runs_dir / "regime_missing.jsonl")

def _log_missing_regime_label(
    *,
    regime: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    forecast_data: Optional[Dict[str, Any]],
    symbol: str,
    timeframe: str,
) -> None:
    if not isinstance(regime, dict):
        regime_dict = None
    else:
        regime_dict = regime
    enabled = regime_dict.get("enabled") if isinstance(regime_dict, dict) else None
    label = regime_dict.get("label") if isinstance(regime_dict, dict) else None
    if enabled is False:
        return
    if isinstance(label, str) and label.strip():
        return
    expected = False
    if isinstance(regime_dict, dict):
        expected = True
    if isinstance(metadata, dict) and (
        "regime" in metadata
        or "regime_mode" in metadata
        or "regime_mode_used" in metadata
        or "regime_detector" in metadata
    ):
        expected = True
    if not expected:
        return
    as_of = _extract_forecast_asof_dt(forecast_data) if isinstance(forecast_data, dict) else None
    if as_of is not None:
        ts = as_of.astimezone(timezone.utc).isoformat()
    else:
        ts = "unknown"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "as_of": ts,
        "enabled": enabled,
        "label": label,
    }
    if isinstance(regime_dict, dict):
        record["route"] = regime_dict.get("route")
        record["mode"] = regime_dict.get("mode") or regime_dict.get("regime_mode")
        record["detector"] = regime_dict.get("detector")
    _append_regime_missing_log(record)

def _forecast_is_time_travel(forecast_data: Dict[str, Any]) -> bool:
    meta = _extract_forecast_metadata(forecast_data)
    if not isinstance(meta, dict):
        return False
    return bool(meta.get("backtest_time_travel"))

def _forecast_asof_matches_bar(
    forecast_data: Dict[str, Any],
    bar_ts: datetime,
    timeframe: str,
) -> bool:
    asof_dt = _extract_forecast_asof_dt(forecast_data)
    if asof_dt is None:
        return False
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
    tf_minutes = _parse_timeframe(timeframe)
    tolerance_sec = max(60, int(tf_minutes * 60 * 0.1))
    return abs((asof_dt - bar_ts).total_seconds()) <= tolerance_sec

def _bars_to_price_frame(bars: List[Bar]):
    try:
        import pandas as pd
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pandas and numpy are required for time-travel forecasts.") from exc
    if not bars:
        raise ValueError("No bars available for forecast.")
    rows = []
    for b in bars:
        ts = b.ts if b.ts.tzinfo is not None else b.ts.replace(tzinfo=timezone.utc)
        rows.append(
            {
                "datetime": ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume) if b.volume is not None else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        raise ValueError("No valid close prices after cleaning.")
    if (df["close"] <= 0).any():
        df = df[df["close"] > 0]
    df["log_price"] = np.log(df["close"])
    df["log_return"] = df["log_price"].diff().fillna(0.0)
    df["return"] = df["close"].pct_change().fillna(0.0)
    return df

def _train_time_travel_mean_model(
    price_frame,
    timeframe: str,
    local_engine_path: Optional[str],
):
    module = _load_local_engine_module(local_engine_path)
    import importlib
    selection = importlib.import_module("alphalens_forecast.models.selection")
    core = importlib.import_module("alphalens_forecast.core")
    config = module.get_config()
    model_type = selection.select_model_type(timeframe)
    device = selection.resolve_device(getattr(config, "torch_device", None), model_type)
    mean_model = selection.instantiate_model(model_type, device=device)
    if hasattr(mean_model, "set_dataloader_config") and getattr(config, "training", None) is not None:
        try:
            mean_model.set_dataloader_config(config.training)
        except Exception:
            pass
    features = core.prepare_features(price_frame)
    mean_model.fit(features.target, features.regressors)
    return mean_model, model_type, device

def _update_time_travel_model_series(
    mean_model: Any,
    price_frame,
    timeframe: str,
) -> bool:
    model_name = mean_model.__class__.__name__
    if model_name != "NHiTSForecaster":
        return False
    try:
        import importlib
        import numpy as np
        core = importlib.import_module("alphalens_forecast.core")
        features = core.prepare_features(price_frame)
        target = features.target
        if hasattr(mean_model, "_build_target_series") and hasattr(mean_model, "_scaler"):
            series = mean_model._build_target_series(target)
            mean_model._series = series.astype(np.float32)
            mean_model._scaled_series = mean_model._scaler.transform(series).astype(np.float32)
            return True
    except Exception:
        return False
    return False

def _time_travel_min_bars(timeframe: str, local_engine_path: Optional[str]) -> int:
    try:
        module = _load_local_engine_module(local_engine_path)
        import importlib
        selection = importlib.import_module("alphalens_forecast.models.selection")
        config = module.get_config()
        model_type = selection.select_model_type(timeframe)
        device = selection.resolve_device(getattr(config, "torch_device", None), model_type)
        model = selection.instantiate_model(model_type, device=device)
        input_len = getattr(model, "_input_chunk_length", None)
        output_len = getattr(model, "_output_chunk_length", None)
        if isinstance(input_len, int) and isinstance(output_len, int) and input_len > 0 and output_len > 0:
            return int(input_len + output_len)
    except Exception:
        return 0
    return 0

def forecast_local_engine_time_travel(
    symbol: str,
    timeframe: str,
    horizons: List[int],
    trade_mode: str,
    local_engine_path: Optional[str],
    execution_price: Optional[float],
    bars: Optional[List[Bar]] = None,
    price_frame: Optional[Any] = None,
    model_cache_dir: Optional[Path] = None,
    mean_model_override: Optional[Any] = None,
    model_type_override: Optional[str] = None,
    device_override: Optional[str] = None,
    use_montecarlo: Optional[bool] = None,
    paths: Optional[int] = None,
) -> Dict[str, Any]:
    module = _load_local_engine_module(local_engine_path)
    start = time.time()
    if price_frame is None:
        if bars is None:
            return {
                "status": "error",
                "error_type": "price_frame_error",
                "message": "No bars provided for time-travel forecast.",
            }
        try:
            price_frame = _bars_to_price_frame(bars)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error_type": "price_frame_error",
                "message": str(exc),
            }
    try:
        config = module.get_config()
        resolved_use_mc = use_montecarlo if use_montecarlo is not None else config.monte_carlo.use_montecarlo
        resolved_paths = paths if paths is not None else config.monte_carlo.paths
        if mean_model_override is None:
            mean_model, model_type, device = _train_time_travel_mean_model(
                price_frame,
                timeframe,
                local_engine_path,
            )
            model_status = {
                "model_type": model_type,
                "device": device,
                "mean": {"loaded": True, "reason": "time_travel_fit"},
                "vol": {"loaded": False, "reason": "time_travel_fit"},
            }
        else:
            mean_model = mean_model_override
            model_type = model_type_override or "unknown"
            device = device_override or getattr(config, "torch_device", "cpu")
            model_status = {
                "model_type": model_type,
                "device": device,
                "mean": {"loaded": True, "reason": "time_travel_cache"},
                "vol": {"loaded": False, "reason": "time_travel_cache"},
            }
        model_dir = model_cache_dir or Path(tempfile.mkdtemp(prefix="alphalens_time_travel_models_"))
        data_provider = module.DataProvider(config.twelve_data, cache_dir=None, auto_refresh=False)
        engine = module.ForecastEngine(config, data_provider, module.ModelRouter(model_dir))
        result = engine.forecast(
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
            paths=resolved_paths,
            use_montecarlo=resolved_use_mc,
            trade_mode=trade_mode,
            show_progress=False,
            mean_model_override=mean_model,
            vol_model_override=None,
            force_retrain=False,
            refresh_data=False,
            execution_price=execution_price,
            execution_price_source="client" if execution_price is not None else None,
            price_frame=price_frame,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_type": "engine_error",
            "message": str(exc),
        }
    elapsed = time.time() - start
    request_context = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": horizons,
        "use_montecarlo": True,
        "paths": 1000,
        "trade_mode": trade_mode,
        "include_predictions": True,
        "include_metadata": True,
        "include_model_info": True,
    }
    if execution_price is not None:
        request_context["execution_price"] = execution_price
    meta = result.metadata or {}
    if not isinstance(meta, dict):
        meta = {}
    meta["backtest_time_travel"] = True
    meta["backtest_time_travel_model"] = model_status.get("model_type")
    payload_out: Dict[str, Any] = {
        "ok": True,
        "status": "ok",
        "message": "forecast completed",
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "request": request_context,
        "warnings": [],
        "data": {
            "payload": result.payload,
            "as_of": result.as_of,
            "data_hash": result.data_hash,
            "durations": result.durations,
        },
        "metadata": meta,
        "model_status": model_status,
        "total_seconds": round(elapsed, 3),
    }
    return {"status": "success", "elapsed_sec": round(elapsed, 2), "data": payload_out}

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

def save_bars_to_csv(bars: List[Bar], path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
        )
        writer.writeheader()
        for b in bars:
            ts = b.ts if b.ts.tzinfo is not None else b.ts.replace(tzinfo=timezone.utc)
            writer.writerow(
                {
                    "timestamp": ts.astimezone(timezone.utc).isoformat(),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": "" if b.volume is None else float(b.volume),
                }
            )

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
    interval = _normalize_timeframe(timeframe)
    params = {
        "symbol": symbol,
        "interval": interval,
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
    as_of: Optional[str] = None,
    asof_field: str = "as_of",
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
        "timeframe": _normalize_timeframe(timeframe),
        "horizons": horizons,
        "trade_mode": trade_mode,
        "use_montecarlo": True,
        "include_predictions": True,
        "include_metadata": True,
        "include_model_info": True,
        "paths": 1000,
    }
    if as_of:
        field = "asOf" if asof_field == "asOf" else "as_of"
        payload[field] = as_of

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
    as_of = req.get("as_of")
    asof_field = req.get("asof_field", "as_of")
    req = dict(req)
    if "timeframe" in req and isinstance(req["timeframe"], str):
        req["timeframe"] = _normalize_timeframe(req["timeframe"])
    if backend == "local_engine":
        return forecast_local_engine(
            symbol=req["symbol"],
            timeframe=req["timeframe"],
            horizons=req["horizons"],
            trade_mode=req.get("trade_mode", "forward"),
            local_engine_path=local_engine_path,
            as_of=as_of,
            asof_field=asof_field,
        )
    if backend == "local_app":
        return forecast_local_app(
            symbol=req["symbol"],
            timeframe=req["timeframe"],
            horizons=req["horizons"],
            trade_mode=req.get("trade_mode", "forward"),
            local_app_path=local_app_path,
            as_of=as_of,
            asof_field=asof_field,
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

def _extract_forecast_metadata(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    if not forecast_data or not isinstance(forecast_data, dict):
        return {}
    candidates = [
        forecast_data.get("metadata"),
        (forecast_data.get("data") or {}).get("metadata"),
        ((forecast_data.get("data") or {}).get("data") or {}).get("metadata"),
    ]
    for cand in candidates:
        if isinstance(cand, dict):
            return cand
    return {}

def _fallback_entry_model_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None
    mean_model = metadata.get("mean_model")
    if not isinstance(mean_model, dict):
        return None
    raw_name = mean_model.get("name") or mean_model.get("class")
    if not raw_name:
        return None
    mean_name = str(raw_name)
    lower = mean_name.lower()
    if "arima" in lower:
        mean_name = "ARIMA"
    elif "kalman" in lower:
        mean_name = "KALMAN"
    elif "momentum" in lower:
        mean_name = "MOMENTUM"
    elif "meanreversion" in lower or "mean_reversion" in lower:
        mean_name = "MEAN_REVERSION"
    elif "flat" in lower:
        mean_name = "FLAT"
    elif "ets" in lower:
        mean_name = "ETS"
    elif lower == "ou" or " ou" in lower or "ou_" in lower or "_ou" in lower:
        mean_name = "OU"
    elif "nhits" in lower:
        mean_name = "NHITS"
    elif "neuralprophet" in lower:
        mean_name = "NeuralProphet"
    elif "prophet" in lower and "neural" not in lower:
        mean_name = "Prophet"
    elif "tft" in lower:
        mean_name = "TFT"
    return {"mean": mean_name}

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
        model_info = h.get("model") if isinstance(h.get("model"), dict) else None
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
                "model": model_info,
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

def _select_candidate_by_target_horizons(
    candidates: List[Dict[str, Any]],
    target_horizons: List[int],
) -> Optional[Dict[str, Any]]:
    if not candidates or not target_horizons:
        return None
    with_h = [c for c in candidates if isinstance(c.get("horizon_bars"), int)]
    if not with_h:
        return None

    def distance(c: Dict[str, Any]) -> int:
        hb = int(c["horizon_bars"])
        return min(abs(hb - t) for t in target_horizons)

    min_dist = min(distance(c) for c in with_h)
    closest = [c for c in with_h if distance(c) == min_dist]
    if len(closest) == 1:
        return closest[0]
    return _select_best_candidate(closest)

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
    regime: Optional[Dict[str, Any]] = None,
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
    entry_model = candidate.get("model") if isinstance(candidate.get("model"), dict) else None
    sl_tp_model = {
        "method": method,
        "risk_appetite": ra,
        "friction_sigma": round(friction, 4),
    }
    if unit is not None:
        sl_tp_model["unit"] = float(unit)
    return TradeSetup(
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        take_profits=[float(tp)],
        confidence=candidate.get("confidence"),
        horizon_bars=candidate.get("horizon_bars"),
        reason=reason,
        entry_model=entry_model,
        sl_tp_model=sl_tp_model,
        regime=regime if isinstance(regime, dict) else None,
    )

def trade_from_forecast(
    forecast_data: Dict[str, Any],
    symbol: str,
    timeframe: str,
    risk_appetite: str,
    market_features: Dict[str, Any],
    skip_range_regime: bool = False,
) -> Optional[TradeSetup]:
    candidates = _extract_trade_candidates(forecast_data, timeframe)
    if not candidates:
        return None
    metadata = _extract_forecast_metadata(forecast_data)
    fallback_model = _fallback_entry_model_from_metadata(metadata if isinstance(metadata, dict) else None)
    regime = metadata.get("regime") if isinstance(metadata, dict) else None
    regime = _augment_regime_info(regime, metadata)
    _log_missing_regime_label(
        regime=regime,
        metadata=metadata if isinstance(metadata, dict) else None,
        forecast_data=forecast_data,
        symbol=symbol,
        timeframe=timeframe,
    )
    if skip_range_regime and isinstance(regime, dict):
        enabled = regime.get("enabled", True)
        label = regime.get("label")
        if enabled and isinstance(label, str) and label.strip().upper() == "RANGE":
            return None
    target_horizons = _default_horizons_for_timeframe(timeframe)
    base = _select_candidate_by_target_horizons(candidates, target_horizons) or _select_best_candidate(candidates)
    if base is None:
        return None
    if fallback_model is not None:
        fallback_mean = fallback_model.get("mean")
        current_model = base.get("model")
        if not isinstance(current_model, dict):
            if fallback_mean:
                base = dict(base)
                base["model"] = {"mean": fallback_mean}
        else:
            merged = dict(current_model)
            if fallback_mean:
                current_mean = merged.get("mean")
                if current_mean is None or not str(current_mean).strip():
                    merged["mean"] = fallback_mean
                else:
                    if str(current_mean).strip().lower() != str(fallback_mean).strip().lower():
                        merged["mean"] = fallback_mean
            base = dict(base)
            base["model"] = merged
    if isinstance(risk_appetite, str) and risk_appetite.strip().lower() == "auto":
        ra = _normalize_risk_appetite(base.get("risk_appetite")) or "moderate"
    else:
        ra = _normalize_risk_appetite(risk_appetite) or "moderate"
    return _build_trade_from_candidate(base, symbol, timeframe, ra, market_features, forecast_data, regime=regime)

def _build_overlay_context(
    trade: Optional[TradeSetup],
    result: Optional[TradeResult],
    *,
    market_features: Optional[Dict[str, Any]] = None,
    forecast_data: Optional[Dict[str, Any]] = None,
    max_leverage: Optional[float] = None,
    vol_ref: Optional[float] = None,
) -> Dict[str, Any]:
    regime = None
    if trade is not None:
        regime = trade.regime
    if regime is None and result is not None:
        regime = result.regime
    if not isinstance(regime, dict):
        regime = {}
    sl_tp_model = None
    if trade is not None and isinstance(trade.sl_tp_model, dict):
        sl_tp_model = trade.sl_tp_model
    if sl_tp_model is None and result is not None and isinstance(result.sl_tp_model, dict):
        sl_tp_model = result.sl_tp_model
    entry_model_vol = _extract_sigma_from_forecast(forecast_data) if forecast_data else None
    context: Dict[str, Any] = {
        "regime_enabled": regime.get("enabled", False),
        "regime_label": regime.get("label"),
        "regime_confidence": regime.get("confidence"),
        "regime_route": regime.get("route"),
        "model_confidence": getattr(trade, "confidence", None) if trade is not None else getattr(result, "confidence", None),
        "entry_model_vol": entry_model_vol,
        "atr": _to_float((market_features or {}).get("atr")),
        "sl_tp_method": (sl_tp_model or {}).get("method") if isinstance(sl_tp_model, dict) else None,
        "max_leverage": max_leverage,
        "vol_ref": vol_ref,
    }
    return context

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
# Position sizing
# -----------------------------

def _warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)

def _positive_float(value: Any) -> Optional[float]:
    val = _to_float(value)
    if val is None or not math.isfinite(val) or val <= 0:
        return None
    return val

def _normalize_sizing_mode(value: Optional[str]) -> str:
    v = (value or "none").strip().lower()
    if v in {"none", "off", "disabled"}:
        return "none"
    if v in {"fixed", "fixed_fractional", "fixed-fractional", "fractional", "ff"}:
        return "fixed_fractional"
    if v in {"vol_target", "volatility_target", "vol_targeting", "volatility_targeting", "vol"}:
        return "vol_target"
    return "none"

def _apply_size_caps(
    size: Optional[float],
    *,
    entry_price: float,
    capital: Optional[float],
    min_size: Optional[float],
    max_size: Optional[float],
    max_leverage: Optional[float],
) -> Optional[float]:
    val = _to_float(size)
    if val is None or not math.isfinite(val) or val <= 0:
        return None
    if min_size is not None and val < min_size:
        val = min_size
    if max_size is not None and val > max_size:
        val = max_size
    if (
        max_leverage is not None
        and capital is not None
        and capital > 0
        and entry_price > 0
    ):
        cap = (capital * max_leverage) / entry_price
        if cap > 0:
            val = min(val, cap)
    if val <= 0 or not math.isfinite(val):
        return None
    return val

def _compute_realized_vol(bars: List[Bar], window: int = 20) -> Optional[float]:
    if not bars or len(bars) < 2:
        return None
    window = max(2, min(window, len(bars)))
    start_idx = len(bars) - window
    returns: List[float] = []
    for i in range(start_idx + 1, len(bars)):
        prev = bars[i - 1].close
        cur = bars[i].close
        if prev is None or prev <= 0:
            continue
        returns.append((cur - prev) / prev)
    if len(returns) < 2:
        return None
    return statistics.pstdev(returns)

def compute_position_size(
    *,
    sizing_mode: str,
    entry_price: float,
    stop_price: Optional[float],
    capital: Optional[float],
    risk_per_trade_pct: Optional[float],
    min_size: Optional[float],
    max_size: Optional[float],
    max_leverage: Optional[float],
    market_features: Optional[Dict[str, Any]] = None,
    bars: Optional[List[Bar]] = None,
    forecast_data: Optional[Dict[str, Any]] = None,
    target_vol: Optional[float] = None,
    vol_lookback: int = 20,
) -> Dict[str, Any]:
    mode = _normalize_sizing_mode(sizing_mode)
    entry = _to_float(entry_price)
    if entry is None or not math.isfinite(entry) or entry <= 0:
        return {"size": None, "reason": "invalid_entry", "skip": True, "method": mode}

    cap = _positive_float(capital)
    risk_pct = _positive_float(risk_per_trade_pct)
    min_pos = _positive_float(min_size)
    max_pos = _positive_float(max_size)
    max_lev = _positive_float(max_leverage)

    size: Optional[float] = None
    reason: Optional[str] = None

    if mode == "fixed_fractional":
        stop_dist: Optional[float] = None
        stop_val = _to_float(stop_price)
        if stop_val is not None:
            stop_dist = abs(entry - stop_val)
        if stop_dist is None or stop_dist <= 0:
            atr = _to_float((market_features or {}).get("atr"))
            if atr and atr > 0:
                stop_dist = atr
        if cap is None:
            reason = "missing_capital"
        elif risk_pct is None:
            reason = "missing_risk_pct"
        elif stop_dist is None or stop_dist <= 0:
            reason = "missing_stop_distance"
        else:
            risk_amount = cap * (risk_pct / 100.0)
            size = risk_amount / stop_dist
    elif mode == "vol_target":
        realized_vol = _compute_realized_vol(bars or [], window=vol_lookback)
        forecast_vol = _extract_sigma_from_forecast(forecast_data or {})
        vol_target = _positive_float(target_vol)
        if cap is None:
            reason = "missing_capital"
        else:
            base_size = cap / entry
            scale: Optional[float] = None
            if vol_target is not None:
                vol = forecast_vol or realized_vol
                if vol and vol > 0:
                    scale = vol_target / vol
                else:
                    reason = "missing_vol"
            else:
                if realized_vol and forecast_vol and realized_vol > 0 and forecast_vol > 0:
                    scale = forecast_vol / realized_vol
                elif realized_vol and realized_vol > 0:
                    scale = 1.0 / realized_vol
                else:
                    reason = "missing_vol"
            if scale is not None:
                size = base_size * scale
    elif mode == "none":
        return {"size": None, "reason": None, "skip": True, "method": mode}
    else:
        return {"size": None, "reason": "unsupported_mode", "skip": True, "method": mode}

    size = _apply_size_caps(
        size,
        entry_price=entry,
        capital=cap,
        min_size=min_pos,
        max_size=max_pos,
        max_leverage=max_lev,
    )

    fallback = False
    if size is None:
        if min_pos is not None:
            size = _apply_size_caps(
                min_pos,
                entry_price=entry,
                capital=cap,
                min_size=None,
                max_size=max_pos,
                max_leverage=max_lev,
            )
            fallback = True
        if size is None:
            return {"size": None, "reason": reason or "size_unavailable", "skip": True, "method": mode}

    notional = entry * size
    leverage = (notional / cap) if cap is not None and cap > 0 else None
    return {
        "size": size,
        "notional": notional,
        "leverage": leverage,
        "reason": reason,
        "fallback": fallback,
        "method": mode,
    }

# -----------------------------
# Metrics
# -----------------------------

def _safe_round(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        if not math.isfinite(value):
            return None
        return round(value, ndigits)
    except Exception:
        return None

_TREND_MODEL_NAMES = {"NHITS", "NEURALPROPHET", "PROPHET", "TFT", "NBEATS"}
_BASELINE_MODEL_NAMES = {"ARIMA", "KALMAN", "MOMENTUM", "MEAN_REVERSION", "FLAT", "ETS", "OU"}

def _normalize_entry_model_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lower = raw.lower()
    if "neuralprophet" in lower:
        return "NEURALPROPHET"
    if "prophet" in lower and "neural" not in lower:
        return "PROPHET"
    if "nhits" in lower:
        return "NHITS"
    if "nbeats" in lower:
        return "NBEATS"
    if "tft" in lower:
        return "TFT"
    if "arima" in lower:
        return "ARIMA"
    if "kalman" in lower:
        return "KALMAN"
    if "momentum" in lower:
        return "MOMENTUM"
    if "meanreversion" in lower or "mean_reversion" in lower:
        return "MEAN_REVERSION"
    if "flat" in lower:
        return "FLAT"
    if "ets" in lower:
        return "ETS"
    if lower == "ou" or " ou" in lower or "ou_" in lower or "_ou" in lower:
        return "OU"
    return raw

def _route_from_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    upper = str(label).strip().upper()
    if upper in {"TREND_UP", "TREND_DOWN"}:
        return "trend"
    if upper in {"RANGE", "BREAKOUT_VOL_EXPANSION", "STRESS_CHOP"}:
        return "alternate"
    return None

def _build_routing_report(trades: List[TradeResult]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "total_trades": len(trades),
        "by_regime": {},
        "mismatches": {
            "trend_route_baseline_model": 0,
            "alternate_route_trend_model": 0,
            "unknown_model": 0,
            "unknown_regime": 0,
        },
        "examples": {
            "trend_route_baseline_model": [],
            "alternate_route_trend_model": [],
        },
    }
    for t in trades:
        regime = t.regime if isinstance(t.regime, dict) else {}
        label_raw = regime.get("label")
        label = str(label_raw).strip().upper() if isinstance(label_raw, str) and label_raw.strip() else "UNKNOWN"
        route = regime.get("route") if isinstance(regime, dict) else None
        if isinstance(route, str):
            route_norm = route.strip().lower()
            if route_norm not in {"trend", "alternate"}:
                route_norm = None
        else:
            route_norm = None
        if route_norm is None:
            route_norm = _route_from_label(label)
        if not route_norm:
            report["mismatches"]["unknown_regime"] += 1

        model = None
        if isinstance(t.entry_model, dict):
            model = t.entry_model.get("mean") or t.entry_model.get("name")
        model_norm = _normalize_entry_model_name(model)
        if not model_norm:
            report["mismatches"]["unknown_model"] += 1
            model_norm = "UNKNOWN"

        entry = report["by_regime"].setdefault(
            label,
            {"total": 0, "route": route_norm, "models": {}},
        )
        entry["total"] += 1
        entry["route"] = entry.get("route") or route_norm
        models = entry["models"]
        models[model_norm] = models.get(model_norm, 0) + 1

        if route_norm == "trend" and model_norm in _BASELINE_MODEL_NAMES:
            report["mismatches"]["trend_route_baseline_model"] += 1
            if len(report["examples"]["trend_route_baseline_model"]) < 5:
                report["examples"]["trend_route_baseline_model"].append(
                    {
                        "as_of": t.as_of.isoformat(),
                        "regime_label": label,
                        "model": model_norm,
                    }
                )
        if route_norm == "alternate" and model_norm in _TREND_MODEL_NAMES:
            report["mismatches"]["alternate_route_trend_model"] += 1
            if len(report["examples"]["alternate_route_trend_model"]) < 5:
                report["examples"]["alternate_route_trend_model"].append(
                    {
                        "as_of": t.as_of.isoformat(),
                        "regime_label": label,
                        "model": model_norm,
                    }
                )
    return report

def _finite_values(values: List[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        val = _to_float(v)
        if val is None or not math.isfinite(val):
            continue
        out.append(val)
    return out

def _max_consecutive(flags: List[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best

def _merge_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def _filter_trades_for_metrics(
    trades: List[TradeResult],
    exclude_timeouts: bool,
) -> tuple[List[TradeResult], int]:
    if not exclude_timeouts:
        return trades, 0
    filtered = [t for t in trades if t.outcome != "timeout"]
    return filtered, (len(trades) - len(filtered))

def compute_backtest_metrics(
    trades: List[TradeResult],
    *,
    initial_capital: Optional[float] = None,
    backtest_start: Optional[datetime] = None,
    backtest_end: Optional[datetime] = None,
    risk_per_trade_pct: Optional[float] = None,
    sizing_mode: Optional[str] = None,
) -> Dict[str, Any]:
    n_trades = len(trades)
    pnl_vals = _finite_values([t.pnl for t in trades])
    if len(pnl_vals) != n_trades and n_trades > 0:
        _warn("Non-finite pnl detected; metrics computed on finite subset.")

    wins = [p for p in pnl_vals if p > 0]
    losses = [p for p in pnl_vals if p < 0]
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = -sum(losses) if losses else 0.0
    net_pnl = sum(pnl_vals) if pnl_vals else 0.0
    gross_pnl = gross_profit - gross_loss

    avg_trade = (sum(pnl_vals) / len(pnl_vals)) if pnl_vals else None
    median_trade = statistics.median(pnl_vals) if pnl_vals else None
    win_rate = (len(wins) / len(pnl_vals)) if pnl_vals else None
    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None
    avg_win_loss_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        avg_win_loss_ratio = avg_win / abs(avg_loss)
    profit_factor = None
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    expectancy = avg_trade if avg_trade is not None else None

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_vals:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = None
    cap = _positive_float(initial_capital)
    if cap is not None:
        equity = cap
        peak = equity
        max_dd = 0.0
        for p in pnl_vals:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        if peak > 0:
            max_dd_pct = max_dd / peak

    r_vals = _finite_values([t.r_multiple for t in trades if t.r_multiple is not None])
    avg_r = (sum(r_vals) / len(r_vals)) if r_vals else None
    median_r = statistics.median(r_vals) if r_vals else None
    r_win_rate = (len([r for r in r_vals if r > 0]) / len(r_vals)) if r_vals else None
    max_r = max(r_vals) if r_vals else None
    min_r = min(r_vals) if r_vals else None
    pct_r_gt_2 = (len([r for r in r_vals if r > 2]) / len(r_vals)) if r_vals else None
    pct_r_gt_1 = (len([r for r in r_vals if r > 1]) / len(r_vals)) if r_vals else None
    pct_r_lt_minus_1 = (len([r for r in r_vals if r < -1]) / len(r_vals)) if r_vals else None
    pct_r_lt_minus_2 = (len([r for r in r_vals if r < -2]) / len(r_vals)) if r_vals else None

    pnl_flags = [p > 0 for p in pnl_vals]
    loss_flags = [p < 0 for p in pnl_vals]
    max_consecutive_wins = _max_consecutive(pnl_flags) if pnl_vals else 0
    max_consecutive_losses = _max_consecutive(loss_flags) if pnl_vals else 0

    trade_pnl_std = statistics.stdev(pnl_vals) if len(pnl_vals) > 1 else None
    largest_win = max(wins) if wins else None
    largest_loss = min(losses) if losses else None

    holding_secs: List[float] = []
    intervals: List[tuple[datetime, datetime]] = []
    for t in trades:
        if not t.entry_time or not t.exit_time:
            continue
        start = t.entry_time
        end = t.exit_time
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end < start:
            continue
        holding_secs.append((end - start).total_seconds())
        intervals.append((start, end))

    avg_holding = (sum(holding_secs) / len(holding_secs)) if holding_secs else None
    median_holding = statistics.median(holding_secs) if holding_secs else None

    if backtest_start is None and intervals:
        backtest_start = min(s for s, _ in intervals)
    if backtest_end is None and intervals:
        backtest_end = max(e for _, e in intervals)

    exposure_time_pct = None
    if backtest_start and backtest_end and backtest_end > backtest_start and intervals:
        merged = _merge_intervals(intervals)
        exposure = sum((e - s).total_seconds() for s, e in merged)
        total = (backtest_end - backtest_start).total_seconds()
        if total > 0:
            exposure_time_pct = exposure / total

    returns: List[float] = []
    if cap is not None and cap > 0 and pnl_vals:
        equity = cap
        for p in pnl_vals:
            if equity <= 0:
                break
            returns.append(p / equity)
            equity += p

    returns_vol = statistics.stdev(returns) if len(returns) > 1 else None
    sharpe = None
    sortino = None
    skew = None
    kurtosis = None
    if returns and returns_vol is not None and returns_vol > 0:
        mean_ret = sum(returns) / len(returns)
        sharpe = mean_ret / returns_vol
        neg = [r for r in returns if r < 0]
        if len(neg) > 1:
            neg_std = statistics.stdev(neg)
            if neg_std > 0:
                sortino = mean_ret / neg_std
        n = len(returns)
        if n >= 3:
            mean = mean_ret
            m2 = sum((r - mean) ** 2 for r in returns) / n
            if m2 > 0:
                m3 = sum((r - mean) ** 3 for r in returns) / n
                skew = m3 / (m2 ** 1.5)
        if n >= 4:
            mean = mean_ret
            m2 = sum((r - mean) ** 2 for r in returns) / n
            if m2 > 0:
                m4 = sum((r - mean) ** 4 for r in returns) / n
                kurtosis = (m4 / (m2 ** 2)) - 3.0

    sizes = _finite_values([t.position_size for t in trades if t.position_size is not None])
    avg_position_size = (sum(sizes) / len(sizes)) if sizes else None
    median_position_size = statistics.median(sizes) if sizes else None
    max_position_size = max(sizes) if sizes else None

    leverages = _finite_values([t.leverage for t in trades if t.leverage is not None])
    avg_leverage = (sum(leverages) / len(leverages)) if leverages else None

    risk_pct_used = None
    if _normalize_sizing_mode(sizing_mode) == "fixed_fractional":
        risk_pct_used = _positive_float(risk_per_trade_pct)

    return {
        "n_trades": n_trades,
        "win_rate": _safe_round(win_rate, 6),
        "gross_pnl": _safe_round(gross_pnl, 6),
        "gross_profit": _safe_round(gross_profit, 6),
        "gross_loss": _safe_round(gross_loss, 6),
        "net_pnl": _safe_round(net_pnl, 6),
        "avg_trade_pnl": _safe_round(avg_trade, 6),
        "median_trade_pnl": _safe_round(median_trade, 6),
        "profit_factor": _safe_round(profit_factor, 6),
        "expectancy_per_trade": _safe_round(expectancy, 6),
        "avg_win": _safe_round(avg_win, 6),
        "avg_loss": _safe_round(avg_loss, 6),
        "avg_win_loss_ratio": _safe_round(avg_win_loss_ratio, 6),
        "max_drawdown": _safe_round(max_dd, 6),
        "max_drawdown_pct": _safe_round(max_dd_pct, 6),
        "sharpe": _safe_round(sharpe, 6),
        "sortino": _safe_round(sortino, 6),
        "returns_volatility": _safe_round(returns_vol, 6),
        "skew": _safe_round(skew, 6),
        "kurtosis": _safe_round(kurtosis, 6),
        "avg_holding_time": _safe_round(avg_holding, 2),
        "median_holding_time": _safe_round(median_holding, 2),
        "exposure_time_pct": _safe_round(exposure_time_pct, 6),
        "avg_r": _safe_round(avg_r, 6),
        "median_r": _safe_round(median_r, 6),
        "r_win_rate": _safe_round(r_win_rate, 6),
        "max_r": _safe_round(max_r, 6),
        "min_r": _safe_round(min_r, 6),
        "pct_r_gt_2": _safe_round(pct_r_gt_2, 6),
        "pct_r_gt_1": _safe_round(pct_r_gt_1, 6),
        "pct_r_lt_-1": _safe_round(pct_r_lt_minus_1, 6),
        "pct_r_lt_-2": _safe_round(pct_r_lt_minus_2, 6),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "trade_pnl_std": _safe_round(trade_pnl_std, 6),
        "largest_win": _safe_round(largest_win, 6),
        "largest_loss": _safe_round(largest_loss, 6),
        "avg_position_size": _safe_round(avg_position_size, 6),
        "median_position_size": _safe_round(median_position_size, 6),
        "max_position_size": _safe_round(max_position_size, 6),
        "avg_leverage": _safe_round(avg_leverage, 6),
        "risk_per_trade_pct": _safe_round(risk_pct_used, 6),
    }

def summarize_results(
    trades: List[TradeResult],
    *,
    initial_capital: Optional[float] = None,
    backtest_start: Optional[datetime] = None,
    backtest_end: Optional[datetime] = None,
    risk_per_trade_pct: Optional[float] = None,
    sizing_mode: Optional[str] = None,
    exclude_timeouts: bool = False,
) -> Dict[str, Any]:
    trades_for_metrics, n_timeouts = _filter_trades_for_metrics(trades, exclude_timeouts)
    metrics = compute_backtest_metrics(
        trades_for_metrics,
        initial_capital=initial_capital,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        risk_per_trade_pct=risk_per_trade_pct,
        sizing_mode=sizing_mode,
    )
    if exclude_timeouts:
        metrics["n_trades_total"] = len(trades)
        metrics["n_timeouts"] = n_timeouts

    trades_for_summary = trades_for_metrics if exclude_timeouts else trades
    routing_report = _build_routing_report(trades_for_summary)
    if not trades_for_summary:
        return {"trades": 0, "net_pnl": 0.0, "metrics": metrics, "routing_report": routing_report}

    net = sum(t.pnl for t in trades_for_summary)
    wins = [t for t in trades_for_summary if t.pnl > 0]
    win_rate = len(wins) / len(trades_for_summary) if trades_for_summary else 0.0
    avg_r = None
    r_vals = [t.r_multiple for t in trades_for_summary if t.r_multiple is not None]
    if r_vals:
        avg_r = sum(r_vals) / len(r_vals)

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades_for_summary:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades_for_summary),
        "net_pnl": net,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4) if avg_r is not None else None,
        "max_drawdown": round(max_dd, 4),
        "metrics": metrics,
        "routing_report": routing_report,
    }

def _flatten_trade_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    entry_model = out.pop("entry_model", None)
    if not isinstance(entry_model, dict):
        entry_model = {}
    out["entry_model_mean"] = entry_model.get("mean")
    out["entry_model_vol"] = entry_model.get("vol")
    out["entry_model_calibrated"] = entry_model.get("calibrated")

    sl_tp_model = out.pop("sl_tp_model", None)
    if not isinstance(sl_tp_model, dict):
        sl_tp_model = {}
    out["sl_tp_method"] = sl_tp_model.get("method")
    out["sl_tp_risk_appetite"] = sl_tp_model.get("risk_appetite")
    out["sl_tp_friction_sigma"] = sl_tp_model.get("friction_sigma")
    out["sl_tp_unit"] = sl_tp_model.get("unit")

    regime = out.pop("regime", None)
    if not isinstance(regime, dict):
        regime = {}
    out["regime_enabled"] = regime.get("enabled")
    out["regime_label"] = regime.get("label")
    out["regime_confidence"] = regime.get("confidence")
    out["regime_lookback"] = regime.get("lookback")
    out["regime_route"] = regime.get("route")
    out["regime_mode"] = regime.get("mode") or regime.get("regime_mode")
    out["regime_mode_used"] = regime.get("mode_used") or regime.get("regime_mode_used")
    out["regime_detector"] = regime.get("detector")
    return out

def _serialize_trade_result(trade: TradeResult, *, flat: bool = False) -> Dict[str, Any]:
    data = asdict(trade)
    for key in ("as_of", "entry_time", "exit_time"):
        val = data.get(key)
        if isinstance(val, datetime):
            data[key] = val.isoformat()
    if flat:
        data = _flatten_trade_fields(data)
    return data

# -----------------------------
# Run logging
# -----------------------------

_RUN_PARAM_FIELDS = [
    "symbol",
    "timeframe",
    "lookback",
    "date_from",
    "date_to",
    "price_csv",
    "gen_mode",
    "forecast_mode",
    "forecast_backend",
    "forecast_asof",
    "forecast_asof_field",
    "trade_mode",
    "forecast_batch_size",
    "macro_mode",
    "risk_appetite",
    "llm_model",
    "llm_temperature",
    "horizons",
    "max_hold_bars",
    "entry_expiry_bars",
    "decision_every",
    "warmup_bars",
    "feature_window",
    "position_mode",
    "fill_policy",
    "batch_size",
    "batch_step",
    "batch_rolling",
    "batch_overlap",
    "batch_parallel",
    "batch_workers",
    "sizing_mode",
    "risk_per_trade_pct",
    "initial_capital",
    "min_position_size",
    "max_position_size",
    "max_leverage",
    "target_vol",
    "vol_lookback",
    "exclude_timeouts",
    "trades_json_nested",
    "skip_range_regime",
    "log_progress_date",
]

def _extract_run_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for field in _RUN_PARAM_FIELDS:
        if hasattr(args, field):
            val = getattr(args, field)
            if isinstance(val, Path):
                val = str(val)
            params[field] = val
    return params

def _get_git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None

def _should_record_runs(args: argparse.Namespace) -> bool:
    flag = getattr(args, "record_runs", 1)
    try:
        return bool(int(flag))
    except Exception:
        return bool(flag)

def _record_backtest_run(
    *,
    args: argparse.Namespace,
    summary: Dict[str, Any],
    backtest_start: Optional[datetime] = None,
    backtest_end: Optional[datetime] = None,
    batch_info: Optional[Dict[str, Any]] = None,
) -> None:
    if not _should_record_runs(args):
        return
    runs_dir = Path(getattr(args, "runs_dir", "backtests_runs"))
    try:
        _ensure_dir(runs_dir)
        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        commit_hash = _get_git_commit_hash()
        params = _extract_run_params(args)
        if backtest_start is not None:
            params["backtest_start"] = backtest_start.isoformat()
        if backtest_end is not None:
            params["backtest_end"] = backtest_end.isoformat()
        if batch_info:
            clean_batch = dict(batch_info)
            if isinstance(clean_batch.get("start_ts"), datetime):
                clean_batch["start_ts"] = clean_batch["start_ts"].isoformat()
            if isinstance(clean_batch.get("end_ts"), datetime):
                clean_batch["end_ts"] = clean_batch["end_ts"].isoformat()
            params["batch"] = clean_batch

        metrics = summary.get("metrics") or {}
        summary_core = {k: v for k, v in summary.items() if k != "metrics"}
        record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_commit": commit_hash,
            "params": params,
            "summary": summary_core,
            "metrics": metrics,
        }

        runs_path = runs_dir / "runs.jsonl"
        with runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        latest_path = runs_dir / "latest_run.json"
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=True, indent=2)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to record backtest run: {exc}")

# -----------------------------
# Market data storage
# -----------------------------

def _should_store_market_data(args: argparse.Namespace) -> bool:
    flag = getattr(args, "store_market_data", False)
    try:
        return bool(int(flag))
    except Exception:
        return bool(flag)

def _market_data_path(base_dir: Path, symbol: str, timeframe: str, bars: List[Bar]) -> Path:
    safe = _safe_symbol(symbol)
    tf = _normalize_timeframe(timeframe)
    if not bars:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}.csv"
    else:
        start_ts = bars[0].ts if bars[0].ts.tzinfo is not None else bars[0].ts.replace(tzinfo=timezone.utc)
        end_ts = bars[-1].ts if bars[-1].ts.tzinfo is not None else bars[-1].ts.replace(tzinfo=timezone.utc)
        start = start_ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        end = end_ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{start}_{end}.csv"
    return base_dir / safe / tf / name

def _store_market_data(
    *,
    args: argparse.Namespace,
    symbol: str,
    timeframe: str,
    bars: List[Bar],
) -> None:
    if not _should_store_market_data(args):
        return
    try:
        out_dir = Path(getattr(args, "market_data_dir", "market_data"))
        out_path = _market_data_path(out_dir, symbol, timeframe, bars)
        save_bars_to_csv(bars, out_path)
        print(f"Market data saved: {out_path}")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Failed to store market data: {exc}")

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
    forecast_asof_mode: str,
    forecast_asof_fixed: Optional[datetime],
    forecast_asof_field: str,
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
        req_asof = _resolve_forecast_asof(forecast_asof_mode, as_of, forecast_asof_fixed)
        to_fetch.append(idx)
        req = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizons": horizons,
            "trade_mode": trade_mode,
            "forecast_url": forecast_url,
        }
        if req_asof:
            req["as_of"] = req_asof
            req["asof_field"] = forecast_asof_field
        requests.append(req)
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

def _run_backtest_core(
    bars: List[Bar],
    args: argparse.Namespace,
    *,
    write_outputs: bool,
    record_runs: bool = False,
    filter_start_ts: Optional[datetime] = None,
    filter_end_ts: Optional[datetime] = None,
) -> tuple[List[TradeResult], Dict[str, Any]]:
    symbol = args.symbol
    timeframe = _normalize_timeframe(args.timeframe)
    args.timeframe = timeframe
    _parse_timeframe(timeframe)

    cache_dir = Path(args.cache_dir)
    _ensure_dir(cache_dir)

    if len(bars) < 10:
        raise RuntimeError("Not enough bars for backtest.")

    decision_every = max(1, args.decision_every)
    warmup = max(5, args.warmup_bars)
    horizons = _resolve_horizons(args.horizons, timeframe)
    max_hold_bars = args.max_hold_bars or max(horizons)
    forecast_batch_size = max(1, args.forecast_batch_size)
    forecast_mem: Dict[int, Dict[str, Any]] = {}
    if args.gen_mode == "forecast" and args.forecast_mode == "off":
        raise RuntimeError("gen-mode=forecast requires forecast_mode=live or replay.")
    management = _management_for_risk_appetite(args.risk_appetite)
    sizing_mode = _normalize_sizing_mode(getattr(args, "sizing_mode", "none"))
    initial_capital = _positive_float(getattr(args, "initial_capital", None))
    risk_per_trade_pct = _positive_float(getattr(args, "risk_per_trade_pct", None))
    min_position_size = _positive_float(getattr(args, "min_position_size", None))
    max_position_size = _positive_float(getattr(args, "max_position_size", None))
    max_leverage = _positive_float(getattr(args, "max_leverage", None))
    target_vol = _positive_float(getattr(args, "target_vol", None))
    vol_lookback = int(getattr(args, "vol_lookback", 20) or 20)
    sizing_warnings: set = set()
    entry_expiry_bars = max(1, int(args.entry_expiry_bars))
    log_progress_date = bool(getattr(args, "log_progress_date", False))
    overlay = None
    if RegimeRiskOverlay is not None:
        try:
            overlay_config = OverlayConfig(vol_ref=target_vol) if OverlayConfig is not None else None
            overlay = RegimeRiskOverlay(overlay_config) if overlay_config is not None else RegimeRiskOverlay()
        except Exception:
            overlay = None
    forecast_backend = args.forecast_backend
    forecast_asof_mode, forecast_asof_fixed = _parse_forecast_asof(args.forecast_asof)
    forecast_asof_field = args.forecast_asof_field
    local_app_path = args.local_app_path
    local_engine_path = args.local_engine_path
    time_travel_enabled = _env_flag("ALPHALENS_BACKTEST_TIME_TRAVEL", True) and args.gen_mode == "forecast"
    time_travel_ready = False
    time_travel_error: Optional[Exception] = None
    if time_travel_enabled and args.forecast_mode != "off":
        try:
            _load_local_engine_module(local_engine_path)
            time_travel_ready = True
        except Exception as exc:  # noqa: BLE001
            time_travel_error = exc
    price_frame_full = None
    if time_travel_enabled and time_travel_ready:
        try:
            price_frame_full = _bars_to_price_frame(bars)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Time-travel price frame build failed: {exc}") from exc
    time_travel_train_every = max(1, _env_int("ALPHALENS_BACKTEST_TRAIN_EVERY", 1))
    time_travel_train_window = max(0, _env_int("ALPHALENS_BACKTEST_TRAIN_WINDOW", 0))
    time_travel_use_mc = _env_flag("ALPHALENS_BACKTEST_USE_MC", True)
    time_travel_paths_raw = _env_int("ALPHALENS_BACKTEST_MC_PATHS", 0)
    time_travel_paths = time_travel_paths_raw if time_travel_paths_raw > 0 else None
    time_travel_roll_update = _env_flag("ALPHALENS_BACKTEST_ROLLING_UPDATE", False)
    time_travel_roll_window = max(0, _env_int("ALPHALENS_BACKTEST_ROLLING_WINDOW", 0))
    time_travel_mean_model = None
    time_travel_model_type = None
    time_travel_device = None
    time_travel_last_train_idx = None
    time_travel_min_bars = 0
    if time_travel_enabled and time_travel_ready:
        time_travel_min_bars = _time_travel_min_bars(timeframe, local_engine_path)
        if time_travel_min_bars > warmup:
            warmup = time_travel_min_bars
    time_travel_min_bars_warned = False

    trades: List[TradeResult] = []
    i = warmup
    end_idx = len(bars) - 2
    while i < end_idx:
        if (i - warmup) % decision_every != 0:
            i += 1
            continue

        as_of = bars[i].ts
        if log_progress_date:
            ts = as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
            _tqdm_write(f"[progress] as_of={ts.astimezone(timezone.utc).isoformat()} idx={i}/{end_idx}")
        req_asof = _resolve_forecast_asof(forecast_asof_mode, as_of, forecast_asof_fixed)

        forecast_data = None
        if args.forecast_mode != "off":
            f_cache = cache_path(cache_dir, "forecast", symbol, timeframe, as_of)
            if time_travel_enabled:
                if not time_travel_ready:
                    detail = f": {time_travel_error}" if time_travel_error else ""
                    raise RuntimeError(
                        "Time-travel forecast requires the local engine to be available"
                        f"{detail}. Set ALPHALENS_BACKTEST_TIME_TRAVEL=0 to disable."
                    )
                if time_travel_min_bars > 0 and (i + 1) < time_travel_min_bars:
                    if not time_travel_min_bars_warned:
                        print(
                            f"[time-travel] waiting for min bars: {time_travel_min_bars} "
                            f"(current={i + 1})"
                        )
                        time_travel_min_bars_warned = True
                    i += 1
                    continue
                if (
                    time_travel_mean_model is None
                    or time_travel_last_train_idx is None
                    or (i - time_travel_last_train_idx) >= time_travel_train_every
                ):
                    if price_frame_full is None:
                        raise RuntimeError("Time-travel price frame is not available.")
                    start_idx = 0
                    if time_travel_train_window > 0:
                        effective_window = max(time_travel_train_window, time_travel_min_bars) if time_travel_min_bars > 0 else time_travel_train_window
                        start_idx = max(0, i + 1 - effective_window)
                    train_slice = price_frame_full.iloc[start_idx : i + 1]
                    if time_travel_min_bars > 0 and len(train_slice) < time_travel_min_bars:
                        i += 1
                        continue
                    time_travel_mean_model, time_travel_model_type, time_travel_device = _train_time_travel_mean_model(
                        train_slice,
                        timeframe,
                        local_engine_path,
                    )
                    time_travel_last_train_idx = i
                elif time_travel_roll_update and time_travel_mean_model is not None:
                    if price_frame_full is None:
                        raise RuntimeError("Time-travel price frame is not available.")
                    update_window = time_travel_roll_window or time_travel_train_window or 0
                    if time_travel_min_bars > 0:
                        update_window = max(update_window, time_travel_min_bars)
                    if update_window > 0:
                        start_idx = max(0, i + 1 - update_window)
                        update_slice = price_frame_full.iloc[start_idx : i + 1]
                    else:
                        update_slice = price_frame_full.iloc[: i + 1]
                    _update_time_travel_model_series(
                        time_travel_mean_model,
                        update_slice,
                        timeframe,
                    )
                if f_cache.exists():
                    forecast_data = _read_json(f_cache)
                if (
                    not forecast_data
                    or not _forecast_asof_matches_bar(forecast_data, as_of, timeframe)
                    or not _forecast_is_time_travel(forecast_data)
                ):
                    forecast_data = forecast_local_engine_time_travel(
                        symbol=symbol,
                        timeframe=timeframe,
                        horizons=horizons,
                        trade_mode=args.trade_mode,
                        local_engine_path=local_engine_path,
                        execution_price=bars[i].close,
                        bars=bars[: i + 1] if price_frame_full is None else None,
                        price_frame=price_frame_full.iloc[: i + 1] if price_frame_full is not None else None,
                        model_cache_dir=cache_dir / "time_travel_models",
                        mean_model_override=time_travel_mean_model,
                        model_type_override=time_travel_model_type,
                        device_override=time_travel_device,
                        use_montecarlo=time_travel_use_mc,
                        paths=time_travel_paths,
                    )
                    if isinstance(forecast_data, dict) and forecast_data.get("status") == "error":
                        raise RuntimeError(f"Time-travel forecast failed: {forecast_data.get('message')}")
                    _write_json(f_cache, forecast_data)
            else:
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
                        forecast_asof_mode=forecast_asof_mode,
                        forecast_asof_fixed=forecast_asof_fixed,
                        forecast_asof_field=forecast_asof_field,
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
                                as_of=req_asof,
                                asof_field=forecast_asof_field,
                            )
                        elif forecast_backend == "local_app":
                            forecast_data = forecast_local_app(
                                symbol=symbol,
                                timeframe=timeframe,
                                horizons=horizons,
                                trade_mode=args.trade_mode,
                                local_app_path=local_app_path,
                                as_of=req_asof,
                                asof_field=forecast_asof_field,
                            )
                        else:
                            forecast_data = forecast_aws_api(
                                symbol=symbol,
                                timeframe=timeframe,
                                horizons=horizons,
                                trade_mode=args.trade_mode,
                                forecast_url=args.forecast_url,
                                as_of=req_asof,
                                asof_field=forecast_asof_field,
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
                    skip_range_regime=bool(getattr(args, "skip_range_regime", False)),
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
            if sizing_mode != "none":
                size_info = compute_position_size(
                    sizing_mode=sizing_mode,
                    entry_price=result.entry,
                    stop_price=result.stop,
                    capital=initial_capital,
                    risk_per_trade_pct=risk_per_trade_pct,
                    min_size=min_position_size,
                    max_size=max_position_size,
                    max_leverage=max_leverage,
                    market_features=market_features,
                    bars=bars[: i + 1],
                    forecast_data=forecast_data,
                    target_vol=target_vol,
                    vol_lookback=vol_lookback,
                )
                if size_info.get("skip"):
                    reason = size_info.get("reason") or "size_unavailable"
                    if reason not in sizing_warnings:
                        _warn(f"Sizing skipped trade: {reason}.")
                        sizing_warnings.add(reason)
                    result = None
                else:
                    size = size_info.get("size")
                    if size is not None:
                        result.position_size = size
                        result.notional = size_info.get("notional")
                        result.leverage = size_info.get("leverage")
                        result.pnl = result.pnl * size
            if result:
                result.entry_model = getattr(trade, "entry_model", None)
                result.sl_tp_model = getattr(trade, "sl_tp_model", None)
                result.regime = getattr(trade, "regime", None)
            if result and overlay is not None:
                context = _build_overlay_context(
                    trade,
                    result,
                    market_features=market_features,
                    forecast_data=forecast_data,
                    max_leverage=max_leverage,
                    vol_ref=target_vol,
                )
                result = overlay.apply(result, context)
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

    if filter_start_ts is not None or filter_end_ts is not None:
        start_ts = filter_start_ts
        end_ts = filter_end_ts
        if start_ts is not None and start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=timezone.utc)
        if end_ts is not None and end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=timezone.utc)
        filtered: List[TradeResult] = []
        for t in trades:
            ts = t.as_of if t.as_of.tzinfo is not None else t.as_of.replace(tzinfo=timezone.utc)
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
            filtered.append(t)
        trades = filtered

    backtest_start = filter_start_ts or (bars[0].ts if bars else None)
    backtest_end = filter_end_ts or (bars[-1].ts if bars else None)
    summary = summarize_results(
        trades,
        initial_capital=initial_capital,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        risk_per_trade_pct=risk_per_trade_pct,
        sizing_mode=sizing_mode,
        exclude_timeouts=bool(getattr(args, "exclude_timeouts", False)),
    )

    if record_runs:
        _record_backtest_run(
            args=args,
            summary=summary,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
        )

    if write_outputs:
        print("\nBacktest summary")
        print(json.dumps(summary, indent=2))

        if args.trades_json:
            out = Path(args.trades_json)
            _ensure_dir(out.parent)
            flat_trades_json = not bool(getattr(args, "trades_json_nested", False))
            with out.open("w", encoding="utf-8") as f:
                json.dump(
                    [_serialize_trade_result(t, flat=flat_trades_json) for t in trades],
                    f,
                    ensure_ascii=True,
                    indent=2,
                )
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

    return trades, summary

def _run_batch_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    args = payload["args"]
    bars = payload["bars"]
    filter_start_ts = payload.get("filter_start_ts")
    filter_end_ts = payload.get("filter_end_ts")
    if args.env_file:
        _load_env(Path(args.env_file))
    _configure_regime_missing_log(args)
    trades, summary = _run_backtest_core(
        bars,
        args,
        write_outputs=False,
        record_runs=False,
        filter_start_ts=filter_start_ts,
        filter_end_ts=filter_end_ts,
    )
    return {
        "trades": trades,
        "summary": summary,
        "batch": payload.get("batch"),
    }

def _run_backtest_batches(bars: List[Bar], args: argparse.Namespace) -> None:
    batch_size = (args.batch_size or "").strip()
    sizing_mode = _normalize_sizing_mode(getattr(args, "sizing_mode", "none"))
    initial_capital = _positive_float(getattr(args, "initial_capital", None))
    risk_per_trade_pct = _positive_float(getattr(args, "risk_per_trade_pct", None))
    if not batch_size:
        _run_backtest_core(bars, args, write_outputs=True, record_runs=True)
        return

    timestamps = _timestamp_list(bars)
    batch_bars = _parse_batch_bars(batch_size)
    batches: List[Dict[str, Any]] = []

    if batch_bars is not None:
        step_bars = _parse_batch_bars(args.batch_step) if args.batch_step else None
        if args.batch_rolling and step_bars is None:
            step_bars = batch_bars
        if not args.batch_rolling:
            step_bars = batch_bars
        base_batches = _build_bar_batches(len(bars), batch_bars, step_bars)
        for entry in base_batches:
            start_idx = entry["start_idx"]
            end_idx = entry["end_idx"]
            if end_idx <= start_idx:
                continue
            start_ts = timestamps[start_idx]
            end_ts = timestamps[end_idx - 1]
            batches.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                }
            )
    else:
        window_offset = _parse_batch_offset(batch_size)
        step_offset = _parse_batch_offset(args.batch_step) if args.batch_step else None
        if args.batch_rolling and step_offset is None:
            step_offset = window_offset
        if not args.batch_rolling:
            step_offset = window_offset
        time_batches = _build_time_batches(timestamps, window_offset, step_offset)
        for entry in time_batches:
            start_ts = entry["start_ts"]
            end_ts = entry["end_ts"]
            start_idx = bisect_left(timestamps, start_ts)
            end_idx = bisect_right(timestamps, end_ts)
            if end_idx <= start_idx:
                continue
            batches.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                }
            )

    if not batches:
        raise RuntimeError("No batches generated; check batch_size and available data.")

    if args.batch_rolling:
        print("[batch] rolling mode enabled; overlapping windows may double-count in global aggregation.")

    overlap = max(0, int(args.batch_overlap))
    time_travel_enabled = _env_flag("ALPHALENS_BACKTEST_TIME_TRAVEL", True) and args.gen_mode == "forecast"
    parallel_requested = bool(args.batch_parallel)
    force_parallel = bool(getattr(args, "batch_parallel_force", False)) or _env_flag(
        "ALPHALENS_BACKTEST_TT_PARALLEL",
        False,
    )
    overlap_active = overlap > 0
    parallel_allowed = parallel_requested and not overlap_active and (not time_travel_enabled or force_parallel)
    if parallel_requested and not parallel_allowed:
        if overlap_active:
            print("[batch] parallel disabled (overlap enabled); running sequential.")
        else:
            print(
                "[batch] parallel disabled (time-travel enabled); use --batch-parallel-force "
                "or set ALPHALENS_BACKTEST_TT_PARALLEL=1 to override."
            )
    if parallel_allowed and time_travel_enabled and force_parallel:
        print("[batch] parallel forced with time-travel; ensure enough GPU/CPU per worker.")

    batch_payloads: List[Dict[str, Any]] = []
    for idx, batch in enumerate(batches, start=1):
        run_start_idx = max(0, batch["start_idx"] - overlap)
        run_end_idx = batch["end_idx"]
        subset = bars[run_start_idx:run_end_idx]
        if len(subset) < 10:
            continue
        payload = {
            "idx": idx,
            "batch": batch,
            "bars": subset,
            "args": args,
            "filter_start_ts": batch["start_ts"],
            "filter_end_ts": batch["end_ts"],
        }
        batch_payloads.append(payload)

    all_trades: List[TradeResult] = []
    batch_reports: List[Dict[str, Any]] = []

    if parallel_allowed:
        workers = args.batch_workers or max(1, (mp.cpu_count() - 1))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(_run_batch_worker, batch_payloads)
            for result in _maybe_tqdm(iterator, total=len(batch_payloads), desc="Batches"):
                batch_info = result.get("batch") or {}
                trades = result.get("trades") or []
                summary = result.get("summary") or {}
                all_trades.extend(trades)
                _record_backtest_run(
                    args=args,
                    summary=summary,
                    backtest_start=batch_info.get("start_ts"),
                    backtest_end=batch_info.get("end_ts"),
                    batch_info=batch_info,
                )
                batch_reports.append(
                    {
                        "start": batch_info.get("start_ts").isoformat() if batch_info.get("start_ts") else None,
                        "end": batch_info.get("end_ts").isoformat() if batch_info.get("end_ts") else None,
                        "bars": (batch_info.get("end_idx", 0) - batch_info.get("start_idx", 0)),
                        "summary": summary,
                    }
                )
    else:
        iterator = _maybe_tqdm(enumerate(batch_payloads, start=1), total=len(batch_payloads), desc="Batches")
        for idx, payload in iterator:
            batch = payload["batch"]
            label = f"{idx}/{len(batch_payloads)} {batch['start_ts'].isoformat()} -> {batch['end_ts'].isoformat()}"
            print(f"\nBatch {label}")
            trades, summary = _run_backtest_core(
                payload["bars"],
                args,
                write_outputs=False,
                record_runs=False,
                filter_start_ts=batch["start_ts"],
                filter_end_ts=batch["end_ts"],
            )
            print(json.dumps(summary, indent=2))
            _record_backtest_run(
                args=args,
                summary=summary,
                backtest_start=batch["start_ts"],
                backtest_end=batch["end_ts"],
                batch_info=batch,
            )
            all_trades.extend(trades)
            batch_reports.append(
                {
                    "start": batch["start_ts"].isoformat(),
                    "end": batch["end_ts"].isoformat(),
                    "bars": batch["end_idx"] - batch["start_idx"],
                    "summary": summary,
                }
            )

    all_trades.sort(key=lambda t: t.as_of)
    backtest_start = bars[0].ts if bars else None
    backtest_end = bars[-1].ts if bars else None
    global_summary = summarize_results(
        all_trades,
        initial_capital=initial_capital,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        risk_per_trade_pct=risk_per_trade_pct,
        sizing_mode=sizing_mode,
        exclude_timeouts=bool(getattr(args, "exclude_timeouts", False)),
    )
    print("\nBacktest summary")
    print(json.dumps(global_summary, indent=2))
    _record_backtest_run(
        args=args,
        summary=global_summary,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        batch_info={"type": "global"},
    )

    if args.trades_json:
        out = Path(args.trades_json)
        _ensure_dir(out.parent)
        flat_trades_json = not bool(getattr(args, "trades_json_nested", False))
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                [_serialize_trade_result(t, flat=flat_trades_json) for t in all_trades],
                f,
                ensure_ascii=True,
                indent=2,
            )
        print(f"Trades saved: {out}")

    if args.trades_csv:
        out = Path(args.trades_csv)
        _ensure_dir(out.parent)
        with out.open("w", encoding="utf-8", newline="") as f:
            if all_trades:
                writer = csv.DictWriter(f, fieldnames=list(asdict(all_trades[0]).keys()))
                writer.writeheader()
                for t in all_trades:
                    writer.writerow(asdict(t))
        print(f"Trades CSV saved: {out}")

    if args.batch_report_json:
        out = Path(args.batch_report_json)
        _ensure_dir(out.parent)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_summary": global_summary,
                    "batches": batch_reports,
                },
                f,
                ensure_ascii=True,
                indent=2,
            )
        print(f"Batch report saved: {out}")

    if args.batch_report_csv and batch_reports:
        out = Path(args.batch_report_csv)
        _ensure_dir(out.parent)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["start", "end", "bars", "summary"],
            )
            writer.writeheader()
            for row in batch_reports:
                writer.writerow(row)
        print(f"Batch report CSV saved: {out}")

def run_backtest(args: argparse.Namespace) -> None:
    _load_env(Path(args.env_file) if args.env_file else None)
    _maybe_autoname_trades_json(args)
    _configure_regime_missing_log(args)
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
        date_from = _parse_date_arg(getattr(args, "date_from", None), is_end=False)
        date_to = _parse_date_arg(getattr(args, "date_to", None), is_end=True)
        if date_to is not None and date_from is None:
            raise RuntimeError("--date-from is required when --date-to is provided.")
        if date_from is not None:
            start = date_from
            end = date_to or datetime.now(timezone.utc)
            if end < start:
                raise RuntimeError("--date-to must be >= --date-from.")
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=args.lookback)
        bars = fetch_twelvedata_bars(symbol, timeframe, start, end, api_key)

    date_from = _parse_date_arg(getattr(args, "date_from", None), is_end=False)
    date_to = _parse_date_arg(getattr(args, "date_to", None), is_end=True)
    if date_from or date_to:
        bars = _filter_bars_by_date(bars, date_from, date_to)

    if len(bars) < 10:
        raise RuntimeError("Not enough bars for backtest.")

    _store_market_data(args=args, symbol=symbol, timeframe=timeframe, bars=bars)

    _run_backtest_batches(bars, args)

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest module for trade generation.")
    p.add_argument("--env-file", default=".env", help="Path to .env file.")
    p.add_argument("--symbol", required=True, help="Instrument symbol, e.g. BTC/USD")
    p.add_argument(
        "--timeframe",
        default="1h",
        help="Timeframe (aliases supported): 1m,5m,15m,30m,45m,1h,2h,4h,1d,1w (or 1min,30min,1day,1week)",
    )
    p.add_argument("--lookback", type=int, default=120, help="Lookback days for price data (if not using CSV).")
    p.add_argument(
        "--date-from",
        default="",
        help="Start date/time for backtest (YYYY-MM-DD or ISO). Overrides lookback when set.",
    )
    p.add_argument(
        "--date-to",
        default="",
        help="End date/time for backtest (YYYY-MM-DD or ISO). Date-only implies end of day UTC.",
    )
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
    p.add_argument(
        "--forecast-asof",
        default="",
        help="Override forecast as_of. Use 'bar' to use each bar timestamp, or ISO datetime/epoch for fixed.",
    )
    p.add_argument(
        "--forecast-asof-field",
        choices=["as_of", "asOf"],
        default="as_of",
        help="Field name for as_of in forecast request payload.",
    )
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
    p.add_argument(
        "--horizons",
        default="auto",
        help="Comma-separated horizons (integers), or 'auto' to derive from timeframe.",
    )
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
    p.add_argument(
        "--sizing-mode",
        choices=["none", "fixed_fractional", "vol_target"],
        default="none",
        help="Position sizing mode (default: none for backward compatibility).",
    )
    p.add_argument(
        "--initial-capital",
        type=float,
        default=0.0,
        help="Initial capital used for sizing and drawdown percentage (0 disables).",
    )
    p.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=0.0,
        help="Risk per trade in percent (e.g. 0.5 for 0.5%%) for fixed fractional sizing.",
    )
    p.add_argument(
        "--min-position-size",
        type=float,
        default=0.0,
        help="Minimum position size (0 disables).",
    )
    p.add_argument(
        "--max-position-size",
        type=float,
        default=0.0,
        help="Maximum position size (0 disables).",
    )
    p.add_argument(
        "--max-leverage",
        type=float,
        default=0.0,
        help="Maximum leverage cap (0 disables).",
    )
    p.add_argument(
        "--target-vol",
        type=float,
        default=0.0,
        help="Target per-period volatility for vol targeting (0 disables).",
    )
    p.add_argument(
        "--vol-lookback",
        type=int,
        default=20,
        help="Lookback bars for realized volatility sizing.",
    )
    p.add_argument(
        "--record-runs",
        type=int,
        default=1,
        help="Append run metrics to runs.jsonl (1=yes, 0=no).",
    )
    p.add_argument(
        "--runs-dir",
        default="backtests_runs",
        help="Directory for backtest run logs (runs.jsonl, latest_run.json).",
    )
    p.add_argument(
        "--store-market-data",
        action="store_true",
        help="Store the market data used for the backtest to disk.",
    )
    p.add_argument(
        "--market-data-dir",
        default="market_data",
        help="Directory for stored market data CSVs.",
    )
    p.add_argument(
        "--exclude-timeouts",
        action="store_true",
        help="Exclude timeout trades from performance metrics (optional).",
    )
    p.add_argument(
        "--log-progress-date",
        action="store_true",
        default=True,
        help="Log progress timestamps at each decision point.",
    )
    p.add_argument(
        "--no-log-progress-date",
        action="store_false",
        dest="log_progress_date",
        help="Disable progress timestamp logging.",
    )
    p.add_argument(
        "--skip-range-regime",
        action="store_true",
        help="Skip trade generation when regime label is RANGE (forecast mode only).",
    )
    p.add_argument("--cache-dir", default="backtest_cache", help="Cache directory for replay.")
    p.add_argument(
        "--batch-size",
        default="",
        help="Enable batch backtest: N bars (e.g. 500) or date offset (1M, 3M, 1Y).",
    )
    p.add_argument(
        "--batch-step",
        default="",
        help="Step between batches (defaults to batch-size). For rolling time batches, provide a step offset.",
    )
    p.add_argument("--batch-rolling", action="store_true", help="Use rolling (overlapping) batch windows.")
    p.add_argument(
        "--batch-overlap",
        type=int,
        default=0,
        help="Extra bars to include before each batch start for continuity.",
    )
    p.add_argument("--batch-parallel", action="store_true", help="Run batches in parallel with multiprocessing.")
    p.add_argument(
        "--batch-workers",
        type=int,
        default=0,
        help="Number of batch workers (0 uses cpu_count - 1).",
    )
    p.add_argument(
        "--batch-parallel-force",
        action="store_true",
        help="Force multiprocessing even with time-travel (use with care).",
    )
    p.add_argument("--batch-report-json", default="", help="Optional JSON file for per-batch summaries.")
    p.add_argument("--batch-report-csv", default="", help="Optional CSV file for per-batch summaries.")
    p.add_argument(
        "--trades-json",
        default="backtest_trades.json",
        help=(
            "Output JSON file for trades. Use 'auto' or a directory path (e.g. trade_data/) "
            "to auto-name the file from backtest parameters."
        ),
    )
    p.add_argument(
        "--trades-json-nested",
        action="store_true",
        help="Keep nested fields in trades JSON (legacy format).",
    )
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
