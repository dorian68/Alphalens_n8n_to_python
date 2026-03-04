# AlphaLens - Lambda + Backtest + Forecast (Local)

**Business Scope (Top Priority)**  
This repo delivers institutional-style trade plans for FX / commodities / crypto by combining macro context, probabilistic forecasting, and risk management in a reproducible workflow.  
Outputs are structured for automation and backtesting (entry/SL/TP, horizon, invalidation, risks, decision verdict).

**Skills Covered (Top Priority)**
- Macro research and institutional synthesis (news + economic calendar + partner research).
- Institutional trade plan structuring (`InstitutionalTradePlan`) with decision and risk framing.
- Multi-mode generation: LLM (JSON), ATR heuristics, probabilistic forecast candidates.
- Risk management: SL/TP calibration, horizons, risk appetite, trailing/break-even.
- Probabilistic surfaces: Monte Carlo TP/SL optimization (target probability, sigma, dof).
- Position sizing: fixed fractional, volatility targeting, size and leverage caps.
- Regime overlays: filtering and scaling by market regime.
- Performance analytics: win rate, profit factor, drawdown, Sharpe/Sortino, R-multiples, exposure.
- Visualization and reporting: trade panoramas and exportable summaries.

**Overview**  
This repository contains a `FastAPI` trade-generation service and a local backtest module.  
The core purpose is to produce structured trade setups and backtest them on market data.

**End-to-End Business Flow**
1. Market ingestion: OHLCV + user parameters (instrument, timeframe, risk profile).
2. Macro context: macro news + economic calendar + partner research (optional).
3. Quant layer: probabilistic forecast + Monte Carlo TP/SL surface for risk calibration.
4. Setup generation: LLM / heuristics / forecast to build an institutional trade plan.
5. Risk and execution rules: SL/TP, risk appetite, trailing/break-even, fill policy, sizing.
6. Backtest and reporting: simulation, metrics, exports, visualizations.

**Architecture / Key Components**
- `alphalens_lambda/app.py` - FastAPI orchestrator (macro, forecast, surface, trade plan).
- `backtest_module.py` - Backtest CLI and performance metrics.
- `alphalens_forecast/trading/overlays/*.py` - Regime overlays and risk controls.
- `alphalens_forecast/visualization/trade_panorama.py` - Trade visualization.
- `macro_commentary_langchain.py` - Macro pipeline (news + institutional synthesis).
- `backtest_cache/` - Local cache for forecasts and replay mode.
- `backtest_trades.json` - Default output file for simulated trades.

**Prerequisites**
- Python 3.10+ recommended.
- Active virtual environment.
- Required API keys for live usage.

**Installation (service app)**
```bash
pip install -r requirements.txt
```

**Run `app.py` locally**
```powershell
uvicorn app:app --host 0.0.0.0 --port 8001
```

Auto-reload:
```powershell
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Notes:
- Port `8001` is just a default choice.
- API keys must be loaded (via `.env`).

**Environment File**
Create a `.env` in `alphalens_lambda/` (examples):
- `OPENAI_API_KEY`
- `TWELVE_DATA_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `CEREBRAS_API_KEY`
- `FINNHUB_TOKEN` (for live macro mode)

**Data Sources and Services**
- Twelve Data: OHLCV for backtest and market features.
- Finnhub News + Economic Calendar: macro context and event agenda.
- Forecast API: probabilistic forecasts and trade candidates.
- Surface API: Monte Carlo TP/SL surface.
- ABCG partner research: optional macro enrichment.

**Backtest Concepts**
- `--gen-mode`: `simple` (local ATR rules), `llm` (LLM generation), `forecast` (uses forecasts and reconstructs setups).
- `--forecast-mode`: `live` (compute and write cache), `replay` (read cache only), `off` (no forecast; incompatible with `gen-mode forecast`).
- `--forecast-backend`: `remote` (HTTP to forecast-proxy), `local_app` (imports `app.py` but still calls forecast-proxy), `local_engine` (calls local `inference_api.py`).

**Trade Idea Engine (Detailed Behavior)**
This engine produces a `TradeSetup` (direction, entry, stop, take_profits, horizon, confidence) which is then simulated by the backtest.

- `gen-mode simple`: local heuristic without ML. Direction = forecast bias if present, otherwise SMA. Entry = last close. SL/TP = ATR multiples (1.5x / 3.0x). Robust fallback, no external dependency.
- `gen-mode llm`: LLM generates full JSON (direction, entry, stop, take_profits). Prompt inputs include symbol/timeframe/as_of, market_features (SMA/ATR/etc), forecast_data (if available), and macro_context (news/econ/ABCG if enabled). Default model is `gpt-4.1-mini` (override with `--llm-model`).
- `gen-mode forecast`: steps: (1) forecast provides candidates (direction, entry, horizon, confidence, risk_appetite). (2) select a candidate by target horizon derived from timeframe (independent of risk_appetite), using confidence as a tiebreaker. (3) reconstruct SL/TP using a standardized method: risk unit = ATR if available, otherwise forecast sigma (vol) times entry. Multipliers by risk_appetite: conservative (SL 1.0x / TP 1.5x), moderate (2.25x / 2.5x), aggressive (5.0x / 3.5x). Apply friction (timeframe + asset class) to widen SL/TP. If ATR and sigma are both missing, the trade is ignored.

**Forecast Backend and Model Selection**
- `remote`: model lives in the remote service (not visible here).
- `local_app`: imports `app.py` but still calls the remote forecast-proxy.
- `local_engine`: loads the local engine repo (example: `alphalens_trade_generation`) and selects the model via `alphalens_forecast.models.selection.select_model_type(timeframe)`. In time-travel mode, the model is trained/updated in a rolling fashion. The model is exposed in metadata (`backtest_time_travel_model`) in the cache.

**Execution Semantics (How a Setup Becomes a Simulated Trade)**
- `forecast`: limit entry within `entry_expiry_bars` window; if not touched, trade is dropped.
- `llm` and `simple`: market entry at the next `open`.
- Only one TP is used (`take_profits[0]`).
- If SL and TP are touched in the same bar, priority follows `--fill-policy` (`sl_first` by default).
- In `forecast` mode, trailing + break-even are applied based on `risk_appetite`.

**Quick Backtest (Replay)**
```powershell
python ..\backtest_module.py --symbol "BTC/USD" --timeframe 1h --lookback 720 --gen-mode forecast --forecast-mode replay --risk-appetite aggressive --entry-expiry-bars 4 --env-file .\.env
```

**Trade Panorama (Offline Visualization)**
```powershell
python -m alphalens_forecast.visualization.trade_panorama `
  --csv trades_resume.csv `
  --json backtest.json `
  --out reports/visuals/backtest_trades_BTC_USD_1h.html `
  --window 3D
```

**Trade Panorama from JSON (trade_data folder)**
```powershell
python -m alphalens_forecast.visualization.trade_panorama `
  --trades-dir .\trade_data `
  --out ..\reports\visuals\backtest_trades_from_json.html `
  --window 3D
```

**Local Forecast Engine**
Local engine location:
`C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation`

Backtest with local engine:
```powershell
python ..\backtest_module.py --symbol "EUR/USD" --timeframe 1h --lookback 200 --gen-mode forecast --forecast-mode live --forecast-backend local_engine --local-engine-path "C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation" --forecast-batch-size 4 --risk-appetite aggressive --entry-expiry-bars 20 --env-file .\.env
```

Important note: `local_engine` imports `inference_api.py` and requires the engine dependencies (for example: `pandas`, `torch`, `darts`). Two options:
- Install the engine dependencies in the same venv.
- Start the local server (`inference_api.py`) and use `--forecast-backend remote` with `--forecast-url http://127.0.0.1:8000/forecast`.

**Performance and Batching**
- `--forecast-batch-size` runs multiple forecasts in parallel. 1 = sequential, 4 to 8 is often optimal, 40 is often too high (CPU/RAM contention).
- `--decision-every` controls generation frequency. 1 = every bar, 2 = every other bar, 4 = every 4 bars.

**Entry and Horizon Management**
- In `gen-mode forecast`, entry is a limit order.
- If entry is not touched within `--entry-expiry-bars`, the trade is dropped.
- Trade horizon also limits the wait. Actual delay is `min(entry_expiry_bars, horizon_trade)`.

**Backtest Assumptions (Notional and Sizing)**
- No sizing or initial capital by default: each trade equals 1 unit of the base asset. Example: on `BTC/USD`, 1 trade = 1 BTC, pnl is USD per BTC.
- `net_pnl` is the sum of unit pnl (no compounding).
- No fees, funding, slippage, spread, or latency modeled.
- Fill uses OHLC bars. Entry is limit in forecast mode (filled if `low <= entry <= high`). If SL and TP are touched in the same bar, priority follows `--fill-policy` (`sl_first` by default).
- Default conditions if not overridden: `--warmup-bars=50`, `--decision-every=1`, `--horizons="auto"` so `--max-hold-bars` = max(horizons), `--position-mode=one_at_time`.
- `risk-appetite` does not affect horizon selection. It only adjusts SL/TP and trailing/break-even (see `_management_for_risk_appetite` in `..\backtest_module.py`).

**Parallelism (Multiprocessing and Multithreading)**
- `--batch-parallel` enables multiprocessing across batches. Each worker runs `_run_backtest_core()` for its window. `--batch-workers` controls process count (default `cpu_count - 1`).
- `--forecast-batch-size > 1` enables multithreading (ThreadPoolExecutor) to launch multiple forecasts in parallel within a batch. Applies to `remote`, `local_app`, and `local_engine`.
- With `gen-mode forecast` + `local_engine`, the engine can run internal parallelism (for example: torch DataLoader). To avoid conflicts, set `TRAIN_NUM_WORKERS=0`.

**GPU / Acceleration**
- GPU is used only by Torch models (example: NHiTS) via Darts / PyTorch Lightning. Set `TORCH_DEVICE=cuda` to enable acceleration if CUDA is available.
- EGARCH, Monte Carlo, and most transformations are CPU.
- In time-travel + `local_engine`, each batch process may try to use the GPU. If you have a single GPU, avoid `--batch-parallel` or set `--batch-workers=1`.

**Backtest Startup Defaults**
- Price source: Twelve Data (unless `--price-csv` is used). Period is `now_utc - lookback` to `now_utc` (UTC).
- Important defaults: `--warmup-bars=50`, `--decision-every=1`.
- `--horizons="auto"` (bars) project rule: 15m -> 12, 30min -> 24, 1h -> 24, 4h -> 30.
- `--max-hold-bars=0` uses max(horizons).
- `--position-mode=one_at_time`.
- `--fill-policy=sl_first`.
- In `gen-mode forecast`, time-travel is enabled by default if `ALPHALENS_BACKTEST_TIME_TRAVEL=1` (default). It trains/updates the local model as the backtest advances.

**Useful Backtest Parameters**
- `--symbol`: instrument (BTC/USD, EUR/USD, etc.).
- `--timeframe`: supported aliases (1m, 5m, 15m, 30m, 45m, 1h, 2h, 4h, 1d, 1w or 1min, 30min, 1day, 1week).
- `--lookback`: days of data.
- `--gen-mode`: `simple`, `llm`, `forecast`.
- `--risk-appetite`: `auto`, `aggressive`, `moderate`, `conservative`.
- `--entry-expiry-bars`: entry expiration.
- `--forecast-mode`: `live`, `replay`, `off`.
- `--forecast-backend`: `remote`, `local_app`, `local_engine`.
- `--forecast-url`: URL for a local engine (if remote).
- `--forecast-batch-size`: parallelism.
- `--decision-every`: trade frequency.
- `--max-hold-bars`: max trade duration.
- `--fill-policy`: `sl_first` or `tp_first`.
- `--position-mode`: `one_at_time` or `overlap`.

**Troubleshooting**
- `No module named 'pandas'`: install local engine dependencies or use the local server via HTTP.
- Multi-line PowerShell commands: backtick ` must be the last character on the line.
- Slow locally: reduce `--forecast-batch-size` and/or increase `--decision-every`.

**Notes**
- `app.py` is intentionally unchanged.
- Backtest is driven from `..\backtest_module.py`.
- Caches are written file by file, even in batch mode.

**Limits and Extensions**
- Options valuation / greeks / volatility smile: not implemented here. You can add an external pricing engine (Black-Scholes, Heston, etc.) and connect it to the forecast/surface backend.
- Real-time execution / OMS / smart routing: integrate on the execution side.

**Merged Documentation**
Content from `alphalens_lambda/README.md` is fully merged here to avoid information loss. The original file remains for historical reference.
