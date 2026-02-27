"""Trade panorama visualization for backtest review.

Usage:
  python -m alphalens_forecast.visualization.trade_panorama \\
      --csv trades_resume.csv \\
      --json backtest.json \\
      --out reports/visuals/backtest_trades.html \\
      --window 3D
"""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List

import pandas as pd


DEFAULT_OUT_DIR = Path("reports/visuals")


def load_trades_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("as_of", "entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if "regime_label" in df.columns:
        df["regime_label"] = df["regime_label"].fillna("UNKNOWN")
    else:
        df["regime_label"] = "UNKNOWN"
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.lower()
    return df


def load_trades_json(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("trades", "data", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise RuntimeError(f"Unsupported trades JSON format: {json_path}")
    df = pd.DataFrame(data)
    for col in ("as_of", "entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if "regime_label" in df.columns:
        df["regime_label"] = df["regime_label"].fillna("UNKNOWN")
    else:
        df["regime_label"] = "UNKNOWN"
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.lower()
    return df


def load_trades_dir(dir_path: str) -> pd.DataFrame:
    base = Path(dir_path)
    if not base.exists():
        raise RuntimeError(f"Trades directory not found: {dir_path}")
    files = sorted(base.glob("*.json"))
    if not files:
        raise RuntimeError(f"No trade JSON files found in {dir_path}")
    frames: List[pd.DataFrame] = []
    for file in files:
        try:
            frames.append(load_trades_json(str(file)))
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"No valid trade JSON files parsed in {dir_path}")
    return pd.concat(frames, ignore_index=True)


def load_backtest_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_symbol_timeframe(trades: pd.DataFrame, data: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    symbol = None
    timeframe = None
    if data:
        meta = data.get("metadata") or data.get("data") or {}
        if isinstance(meta, dict):
            symbol = meta.get("symbol") or meta.get("Symbol")
            timeframe = meta.get("timeframe") or meta.get("Timeframe")
    if symbol is None and "symbol" in trades.columns and trades["symbol"].notna().any():
        symbol = str(trades["symbol"].dropna().iloc[0])
    if timeframe is None and "timeframe" in trades.columns and trades["timeframe"].notna().any():
        timeframe = str(trades["timeframe"].dropna().iloc[0])
    return symbol or "unknown", timeframe or "unknown"


def _extract_price_series(data: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not isinstance(data, dict):
        return None

    def _try_list(obj: Any) -> Optional[pd.DataFrame]:
        if not isinstance(obj, list) or not obj:
            return None
        first = obj[0]
        if not isinstance(first, dict):
            return None
        ts_key = None
        for key in ("timestamp", "ts", "time", "datetime", "date"):
            if key in first:
                ts_key = key
                break
        if ts_key is None:
            return None
        close_key = "close" if "close" in first else None
        if close_key is None:
            for key in ("price", "last", "value", "y"):
                if key in first:
                    close_key = key
                    break
        if close_key is None:
            return None
        frame = pd.DataFrame(obj)
        frame["timestamp"] = pd.to_datetime(frame[ts_key], errors="coerce", utc=True)
        frame["close"] = pd.to_numeric(frame[close_key], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "close"])
        if frame.empty:
            return None
        return frame.sort_values("timestamp")[["timestamp", "close"]]

    for key in ("bars", "ohlcv", "prices", "price_series", "data"):
        candidate = data.get(key)
        frame = _try_list(candidate)
        if frame is not None:
            return frame

    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("bars", "ohlcv", "prices", "price_series"):
            frame = _try_list(nested.get(key))
            if frame is not None:
                return frame
    return None


def _fallback_price_series(trades: pd.DataFrame) -> Optional[pd.DataFrame]:
    if trades.empty:
        return None
    rows = []
    for _, row in trades.iterrows():
        entry_t = row.get("entry_time") or row.get("as_of")
        if pd.notna(entry_t):
            rows.append((entry_t, row.get("entry")))
        exit_t = row.get("exit_time")
        if pd.notna(exit_t):
            rows.append((exit_t, row.get("exit_price")))
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=["timestamp", "close"])
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    return frame


def _build_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: str,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        delta = pd.Timedelta(window)
    except Exception:
        try:
            delta = pd.Timedelta(days=int(window))
        except Exception:
            delta = pd.Timedelta("3D")
    cursor = start
    if delta.total_seconds() <= 0:
        delta = pd.Timedelta("3D")
    while cursor < end:
        win_end = min(end, cursor + delta)
        yield cursor, win_end
        cursor = win_end


def _render_plotly(
    trades: pd.DataFrame,
    price_frame: pd.DataFrame,
    out_html: Path,
    window: str,
    max_trades_per_window: int,
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ],
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=price_frame["timestamp"],
            y=price_frame["close"],
            name="Price",
            line=dict(color="#1f77b4", width=1.4),
            hovertemplate="Time=%{x}<br>Price=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    trades_view = trades
    if max_trades_per_window and len(trades_view) > max_trades_per_window:
        trades_view = trades_view.iloc[:: max(1, len(trades_view) // max_trades_per_window)]

    if not trades_view.empty:
        entry_long = trades_view[trades_view["direction"] == "long"]
        entry_short = trades_view[trades_view["direction"] == "short"]
        for label, frame, color in (
            ("Entry Long", entry_long, "#2ca02c"),
            ("Entry Short", entry_short, "#d62728"),
        ):
            if not frame.empty:
                fig.add_trace(
                    go.Scatter(
                        x=frame["entry_time"],
                        y=frame["entry"],
                        mode="markers",
                        name=label,
                        marker=dict(color=color, size=6),
                        customdata=frame[[
                            "direction",
                            "entry",
                            "stop",
                            "take_profit",
                            "pnl",
                            "outcome",
                            "regime_label",
                        ]].values,
                        hovertemplate=(
                            "Entry=%{customdata[1]:.4f}<br>"
                            "Stop=%{customdata[2]:.4f}<br>"
                            "TP=%{customdata[3]:.4f}<br>"
                            "PnL=%{customdata[4]:.4f}<br>"
                            "Outcome=%{customdata[5]}<br>"
                            "Regime=%{customdata[6]}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        outcomes = {
            "tp": "#2ca02c",
            "sl": "#d62728",
            "timeout": "#7f7f7f",
        }
        for outcome, color in outcomes.items():
            frame = trades_view[trades_view["outcome"] == outcome] if "outcome" in trades_view.columns else pd.DataFrame()
            if not frame.empty:
                fig.add_trace(
                    go.Scatter(
                        x=frame["exit_time"],
                        y=frame["exit_price"],
                        mode="markers",
                        name=f"Exit {outcome.upper()}",
                        marker=dict(color=color, size=6, symbol="x"),
                        hovertemplate=f"Exit=%{{y:.4f}}<br>Outcome={outcome}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        shapes = []
        for _, row in trades_view.iterrows():
            entry_t = row.get("entry_time")
            exit_t = row.get("exit_time")
            if pd.isna(entry_t) or pd.isna(exit_t):
                continue
            for level_key, color in (("stop", "#d62728"), ("take_profit", "#2ca02c")):
                level = row.get(level_key)
                if pd.isna(level):
                    continue
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=entry_t,
                        x1=exit_t,
                        y0=level,
                        y1=level,
                        line=dict(color=color, width=1, dash="dot"),
                    )
                )
        fig.update_layout(shapes=shapes)

        regime_colors = {
            "TREND_UP": "rgba(44,160,44,0.12)",
            "TREND_DOWN": "rgba(214,39,40,0.12)",
            "RANGE": "rgba(31,119,180,0.12)",
            "BREAKOUT_VOL_EXPANSION": "rgba(255,127,14,0.12)",
            "STRESS_CHOP": "rgba(148,103,189,0.12)",
            "UNKNOWN": "rgba(127,127,127,0.08)",
        }
        shapes = list(fig.layout.shapes or [])
        for _, row in trades_view.iterrows():
            entry_t = row.get("entry_time")
            exit_t = row.get("exit_time")
            if pd.isna(entry_t) or pd.isna(exit_t):
                continue
            label = str(row.get("regime_label") or "UNKNOWN")
            color = regime_colors.get(label, "rgba(127,127,127,0.08)")
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=entry_t,
                    x1=exit_t,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                )
            )
        fig.update_layout(shapes=shapes)

    # Cumulative PnL
    if "pnl" in trades.columns:
        pnl_series = trades.dropna(subset=["pnl", "exit_time"]).sort_values("exit_time")
        pnl_series["cum_pnl"] = pnl_series["pnl"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=pnl_series["exit_time"],
                y=pnl_series["cum_pnl"],
                name="Cumulative PnL",
                line=dict(color="#17becf"),
            ),
            row=2,
            col=1,
        )

    # PnL by regime
    if "pnl" in trades.columns:
        pnl_by_regime = trades.groupby("regime_label")["pnl"].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=pnl_by_regime.index,
                y=pnl_by_regime.values,
                name="PnL by Regime",
                marker=dict(color="#9467bd"),
            ),
            row=2,
            col=2,
        )

    # Win rate heatmap
    if "pnl" in trades.columns and "direction" in trades.columns:
        trades["win"] = trades["pnl"] > 0
        pivot = trades.pivot_table(
            index="regime_label",
            columns="direction",
            values="win",
            aggfunc="mean",
        )
        if not pivot.empty:
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="WinRate"),
                ),
                row=3,
                col=1,
            )

    # R multiple distribution
    if "r_multiple" in trades.columns:
        r_vals = trades["r_multiple"].dropna()
        if not r_vals.empty:
            fig.add_trace(
                go.Histogram(
                    x=r_vals,
                    name="R Multiple",
                    marker=dict(color="#ff7f0e"),
                ),
                row=3,
                col=2,
            )

    # Window navigation
    windows = list(_build_windows(price_frame["timestamp"].min(), price_frame["timestamp"].max(), window))
    buttons = []
    for start, end in windows:
        label = f"{start.date()} → {end.date()}"
        buttons.append(
            dict(
                label=label,
                method="relayout",
                args=[{"xaxis.range": [start, end]}],
            )
        )
    if buttons:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=1.01,
                    y=1.05,
                    showactive=True,
                    buttons=buttons,
                )
            ]
        )

    fig.update_layout(
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=30),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), include_plotlyjs="embed", full_html=True)
    return str(out_html)


def _render_error_html(
    out_html: Path,
    plotly_error: Optional[BaseException] = None,
    mpl_error: Optional[BaseException] = None,
) -> str:
    details = []
    if plotly_error is not None:
        details.append(f"Plotly error: {plotly_error}")
    if mpl_error is not None:
        details.append(f"Matplotlib error: {mpl_error}")
    detail_text = "<br>".join(details) if details else "Unknown error."
    html = (
        "<html><body>"
        "<h3>Plot unavailable.</h3>"
        "<p>Install plotly (preferred) or matplotlib.</p>"
        f"<p>{detail_text}</p>"
        "</body></html>"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return str(out_html)


def _render_matplotlib(
    trades: pd.DataFrame,
    price_frame: pd.DataFrame,
    out_html: Path,
    *,
    plotly_error: Optional[BaseException] = None,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _render_error_html(out_html, plotly_error=plotly_error, mpl_error=Exception("matplotlib not installed"))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_frame["timestamp"], price_frame["close"], color="black", linewidth=1.2)
    if not trades.empty:
        longs = trades[trades["direction"] == "long"]
        shorts = trades[trades["direction"] == "short"]
        ax.scatter(longs["entry_time"], longs["entry"], c="green", s=10, label="Entry Long")
        ax.scatter(shorts["entry_time"], shorts["entry"], c="red", s=10, label="Entry Short")
    ax.set_title("Trade Panorama (static fallback)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    img_path = out_html.with_suffix(".png")
    fig.savefig(img_path, dpi=120)
    plt.close(fig)

    encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    html = f"<html><body><img src='data:image/png;base64,{encoded}' /></body></html>"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return str(out_html)


def render_trade_panorama(
    csv_path: str,
    json_path: Optional[str] = None,
    trades_json_path: Optional[str] = None,
    trades_dir: Optional[str] = None,
    out_html: Optional[str] = None,
    window: str = "3D",
    max_trades_per_window: int = 300,
) -> str:
    trades: pd.DataFrame
    if trades_dir:
        trades = load_trades_dir(trades_dir)
    elif trades_json_path:
        trades = load_trades_json(trades_json_path)
    else:
        trades = load_trades_csv(csv_path)
    for col in ("entry_time", "exit_time", "as_of", "entry", "stop", "take_profit", "exit_price", "pnl", "outcome", "direction"):
        if col not in trades.columns:
            trades[col] = pd.NA
    data = load_backtest_json(json_path) if json_path else None
    price_frame = _extract_price_series(data)
    if price_frame is None:
        price_frame = _fallback_price_series(trades)
    if price_frame is None or price_frame.empty:
        raise RuntimeError("No price data available for visualization.")

    symbol, timeframe = _maybe_symbol_timeframe(trades, data)
    if out_html is None:
        out_html = str(DEFAULT_OUT_DIR / f"backtest_trades_{symbol}_{timeframe}.html")
    out_path = Path(out_html)

    plotly_error: Optional[BaseException] = None
    try:
        import plotly  # noqa: F401

        return _render_plotly(
            trades=trades,
            price_frame=price_frame,
            out_html=out_path,
            window=window,
            max_trades_per_window=max_trades_per_window,
        )
    except Exception as exc:
        plotly_error = exc
        print(f"[visualization] Plotly failed: {exc}", file=sys.stderr)
        return _render_matplotlib(
            trades=trades,
            price_frame=price_frame,
            out_html=out_path,
            plotly_error=plotly_error,
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render backtest trade panorama HTML.")
    p.add_argument("--csv", default="", help="Path to trades resume CSV.")
    p.add_argument("--trades-json", default="", help="Path to trades JSON (list).")
    p.add_argument("--trades-dir", default="", help="Directory containing trade JSON files.")
    p.add_argument("--json", default="", help="Optional backtest JSON path.")
    p.add_argument("--out", default="", help="Output HTML path.")
    p.add_argument("--window", default="3D", help="Window size (pandas offset, e.g. 3D, 12H).")
    p.add_argument("--max-trades-per-window", type=int, default=300)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.csv and not args.trades_json and not args.trades_dir:
        raise SystemExit("Provide --csv, --trades-json, or --trades-dir.")
    out = render_trade_panorama(
        csv_path=args.csv,
        json_path=args.json or None,
        trades_json_path=args.trades_json or None,
        trades_dir=args.trades_dir or None,
        out_html=args.out or None,
        window=args.window,
        max_trades_per_window=args.max_trades_per_window,
    )
    print(out)


if __name__ == "__main__":
    main()
