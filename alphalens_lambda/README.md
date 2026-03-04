# AlphaLens Lambda + Backtest (Local)

**Vue d'ensemble**  
Ce dépôt contient le service `FastAPI` de génération de trades (fichier `app.py`) et s’appuie sur un module de backtest externe `..\backtest_module.py`.  
L’objectif principal est de **générer des setups de trade** (ou d’exécuter des forecasts) et **backtester** ces niveaux (entry/SL/TP) sur les données Twelve Data.

**Positionnement métier (résumé)**  
Ce projet outille la **prise de décision de trading** (FX / commodities / crypto) en combinant macro, quant et gestion du risque dans un flux reproductible, puis en **backtestant** les niveaux proposés.  
Il vise des livrables de type **plan de trade institutionnel** (macro anchor, niveaux, invalidation, risques) et des sorties normalisées pour l’automatisation.

**Flux métier de bout en bout**
1. **Ingestion marché** : OHLCV via Twelve Data + paramètres utilisateur (instrument, timeframe, risque, stratégie).
2. **Contexte macro** : synthèse news macro + calendrier éco + recherche partenaire (ABCG) si activé.
3. **Couche quant** : forecast probabiliste (Forecast API) + surface TP/SL (Monte Carlo) pour calibrer le risque.
4. **Génération de setup** : LLM / heuristique / forecast → structuration en plan de trade institutionnel.
5. **Risque & exécution** : SL/TP, risk appetite, trailing/break‑even, policy de fill, sizing.
6. **Backtest & reporting** : simulation, métriques de performance, exports JSON/CSV, visualisations HTML.

**Compétences / skills couvertes (dans ce repo)**
- **Recherche macro & synthèse institutionnelle** (agents macro, Finnhub News + Economic Calendar).
- **Structuration d’un plan de trade** (`InstitutionalTradePlan`) : macro anchor, setups, décision, risques, invalidation.
- **Génération multi‑modes** : LLM (génération JSON), heuristique ATR, forecast probabiliste.
- **Gestion du risque** : SL/TP, risk appetite, horizons, trailing/break‑even, entry expiry, fill policy.
- **Surfaces probabilistes** : génération Monte Carlo d’une surface TP/SL (cibles de probabilité, sigma, dof).
- **Position sizing** : fixed fractional, vol targeting, caps de taille et de leverage.
- **Regime overlays** : filtrage et scaling selon régimes (trend/range/breakout) et signaux de contexte.
- **Mesure de performance** : win rate, profit factor, drawdown, Sharpe/Sortino, R‑multiples, exposition.
- **Visualisation** : trade panorama et exports dédiés au suivi et au reporting.

**Périmètre & extensions (à brancher si besoin)**
- **Valorisation d’options / greeks / volatility smile** : non implémenté dans ce repo.  
  Possible via un moteur externe de pricing (ex: Black‑Scholes, Heston) branché au backend forecast/surface.
- **Execution temps réel / OMS / market making** : non couverts ici, à intégrer côté exécution.

**Composants principaux**
- `app.py`  
  Entrypoint FastAPI / Lambda. Orchestre les tools (forecast, market data, surface).
- `..\backtest_module.py`  
  CLI de backtest. C’est là que se fait la simulation de trades.
- `backtest_cache/`  
  Cache local des forecasts et autres données pour le mode replay.
- `backtest_trades.json`  
  Sortie par défaut des trades simulés.

---

**Prérequis**
- Python 3.10+ recommandé.
- Un environnement virtuel actif.
- Accès aux clés API nécessaires si usage live.

**Installation (service app)**
```bash
pip install -r requirements.txt
```

---

**Lancer `app.py` en serveur local**
Le service est un `FastAPI`. Pour le démarrer localement :

```powershell
uvicorn app:app --host 0.0.0.0 --port 8001
```

Pour activer le reload automatique :

```powershell
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Notes :
- Le port `8001` est libre à toi.
- Les clés d’API doivent être chargées (via `.env`).

**Fichier d’environnement**
Créer un `.env` dans ce dossier (exemples) :
- `OPENAI_API_KEY`
- `TWELVE_DATA_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `CEREBRAS_API_KEY`
- `FINNHUB_TOKEN` (si macro live)

---

**Backtest : concepts clés**
- `--gen-mode`  
  `simple` : règles ATR locales  
  `llm` : génération via LLM  
  `forecast` : **utilise les forecasts** et reconstruit les setups
- `--forecast-mode`  
  `live` : calcule et écrit dans le cache  
  `replay` : lit uniquement le cache  
  `off` : pas de forecast (incompatible avec `gen-mode forecast`)
- `--forecast-backend`  
  `remote` : HTTP vers forecast-proxy  
  `local_app` : importe `app.py` (mais appelle toujours forecast-proxy)  
  `local_engine` : appelle le moteur local `inference_api.py`

---

**Moteur de génération d’idée de trade (fonctionnement détaillé)**
Ce moteur fabrique un `TradeSetup` (direction, entry, stop, take_profits, horizon, confidence) qui sera ensuite simulé par le backtest.

- `gen-mode simple`  
  Heuristique locale sans ML. Direction = biais forecast si dispo, sinon SMA. Entry = dernier close. SL/TP = multiples d’ATR (1.5x / 3.0x).  
  Objectif : fallback robuste, zéro dépendance externe.

- `gen-mode llm`  
  Le LLM génère directement le JSON complet (direction, entry, stop, take_profits).  
  Entrées du prompt : symbol/timeframe/as_of + market_features (SMA/ATR/etc) + forecast_data (si présent) + macro_context (news/econ/ABCG si activé).  
  Modèle par défaut : `gpt-4.1-mini` (modifiable via `--llm-model`).

- `gen-mode forecast`  
  1) Le forecast fournit des candidats (direction, entry, horizon, confidence, risk_appetite).  
  2) Sélection d’un candidat **par horizon cible** (indépendant du risk_appetite) :  
     - horizon cible dérivé du timeframe (règle projet).  
     - si plusieurs candidats proches, on prend le meilleur par confidence (sinon horizon le plus court).  
  3) Reconstruction SL/TP (méthodo standardisée) :  
     - Unité de risque = ATR si dispo, sinon sigma forecast (vol) × entry.  
     - Multiplicateurs selon `risk_appetite` :  
       `conservative` (SL 1.0x / TP 1.5x), `moderate` (2.25x / 2.5x), `aggressive` (5.0x / 3.5x).  
     - Ajustement “friction” (timeframe + asset class) pour élargir SL/TP.  
  Si ATR et sigma sont absents → trade ignoré.

- `forecast-backend` et modèles utilisés  
  - `remote` : modèle côté service distant (non visible ici).  
  - `local_app` : importe `app.py`, mais appelle toujours le forecast-proxy distant.  
  - `local_engine` : charge le repo moteur local (ex: `alphalens_trade_generation`) et sélectionne le modèle via  
    `alphalens_forecast.models.selection.select_model_type(timeframe)`; en time-travel, le modèle est entraîné/actualisé en rolling.  
    Le modèle choisi est exposé dans les métadonnées (`backtest_time_travel_model` dans le cache).

**Exécution (comment l’idée devient un trade simulé)**  
- `forecast` : entrée “limit” dans une fenêtre `entry_expiry_bars` (si non touchée → trade ignoré).  
- `llm` / `simple` : entrée “market” au prochain `open`.  
- Un seul TP est utilisé (`take_profits[0]`).  
- Si SL et TP sont touchés dans la même bougie, la priorité dépend de `--fill-policy` (`sl_first` par défaut).  
- En `forecast`, un trailing + break‑even est appliqué selon `risk_appetite`.

---

**Backtest rapide (replay)**
```powershell
python ..\backtest_module.py --symbol "BTC/USD" --timeframe 1h --lookback 720 --gen-mode forecast --forecast-mode replay --risk-appetite aggressive --entry-expiry-bars 4 --env-file .\.env
```

---

**Trade panorama (visualisation offline)**
```powershell
python -m alphalens_forecast.visualization.trade_panorama `
  --csv trades_resume.csv `
  --json backtest.json `
  --out reports/visuals/backtest_trades_BTC_USD_1h.html `
  --window 3D
```

**Trade panorama depuis JSON (dossier trade_data)**
```powershell
python -m alphalens_forecast.visualization.trade_panorama `
  --trades-dir .\trade_data `
  --out ..\reports\visuals\backtest_trades_from_json.html `
  --window 3D
```

**Forecast local (moteur local)**
Le moteur local se trouve ici :  
`C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation`

Backtest avec calcul local :
```powershell
python ..\backtest_module.py --symbol "EUR/USD" --timeframe 1h --lookback 200 --gen-mode forecast --forecast-mode live --forecast-backend local_engine --local-engine-path "C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation" --forecast-batch-size 4 --risk-appetite aggressive --entry-expiry-bars 20 --env-file .\.env
```

**Note importante**  
Le backend `local_engine` importe `inference_api.py` et nécessite les dépendances du moteur (ex. `pandas`, `torch`, `darts`, etc.).  
Deux options possibles :
- Installer les dépendances du moteur **dans ce même venv**.
- Démarrer le serveur local (`inference_api.py`) et utiliser `--forecast-backend remote` avec `--forecast-url http://127.0.0.1:8000/forecast`.

---

**Performance et batch**
`--forecast-batch-size` lance plusieurs forecasts en parallèle.
- `1` = séquentiel
- `4` à `8` = souvent optimal
- `40` = souvent trop élevé (contention CPU/RAM)

`--decision-every` contrôle **la fréquence de génération** :
- `1` = chaque bougie
- `2` = une bougie sur deux
- `4` = toutes les 4 bougies

---

**Gestion d’entrée et horizon**
En `gen-mode forecast` :
- entry est **limit**  
- si l’entry n’est pas touchée dans `--entry-expiry-bars`, le trade est abandonné  
- l’horizon du trade limite aussi l’attente  
  → délai réel = `min(entry_expiry_bars, horizon_trade)`

---

**Hypothèses de backtest (notionnel & sizing)**
- Il n’y a **pas de sizing** ni de capital initial dans le backtest : chaque trade vaut **1 unité** de l’actif de base.  
  Exemple : sur `BTC/USD`, 1 trade = **1 BTC** ; le `pnl` est donc en **USD par BTC**.
- `net_pnl` = somme des `pnl` unitaires de chaque trade (pas de compounding).
- Aucun frais, funding, slippage, spread, ni latence ne sont modélisés.
- Le remplissage se fait via les bougies OHLC :  
  - entry **limit** en mode `forecast` (rempli si `low <= entry <= high`)  
  - si **SL et TP touchés dans la même bougie**, le choix dépend de `--fill-policy` (`sl_first` par défaut).
- Les conditions de départ par défaut (si non surchargées en CLI) :  
  - `--warmup-bars=50`  
  - `--decision-every=1`  
  - `--horizons="auto"` donc `--max-hold-bars` = max(horizons)  
  - `--position-mode=one_at_time`
- Le `risk-appetite` n’influe plus sur le choix d’horizon.  
  Il sert uniquement à ajuster SL/TP et le trailing/break‑even (voir `_management_for_risk_appetite` dans `..\backtest_module.py`).

---

**Parallélisme (multiprocessing & multithreading)**
- `--batch-parallel` active un **multiprocessing** (processus séparés) sur les batches.  
  Chaque worker exécute `_run_backtest_core()` sur sa fenêtre.  
  `--batch-workers` contrôle le nombre de processus (par défaut `cpu_count - 1`).
- `--forecast-batch-size > 1` active un **multithreading** (ThreadPoolExecutor) pour lancer **plusieurs forecasts en parallèle** dans un batch.  
  Cela s’applique aux backends `remote`, `local_app` et `local_engine`.
- Avec `gen-mode forecast` + `local_engine`, le moteur peut aussi lancer du parallélisme interne (ex: torch DataLoader).  
  Pour éviter les conflits, vous pouvez fixer `TRAIN_NUM_WORKERS=0`.

---

**GPU / Accélération**
- Le GPU est utilisé **uniquement** par les modèles Torch (ex: **NHiTS**) via Darts / PyTorch Lightning.  
  `TORCH_DEVICE=cuda` active l’accélération si CUDA est disponible.
- Les modules **EGARCH**, **Monte Carlo** et la plupart des transformations sont **CPU**.
- En mode `time-travel` + `local_engine`, chaque **processus batch** peut tenter d’utiliser le GPU.  
  Si vous n’avez qu’un GPU, évitez `--batch-parallel` ou limitez `--batch-workers=1`.

---

**Conditions de démarrage du backtest (par défaut)**
- Source de prix : Twelve Data (sauf `--price-csv`).  
  La période est `now_utc - lookback` à `now_utc` (UTC).
- Valeurs par défaut importantes :  
  - `--warmup-bars=50`  
  - `--decision-every=1`  
- `--horizons="auto"` (barres)  
  Règle projet (auto) :  
  - 15m → 12  
  - 30min → 24  
  - 1h → 24  
  - 4h → 30  
  - `--max-hold-bars=0` → prend `max(horizons)`  
  - `--position-mode=one_at_time`  
  - `--fill-policy=sl_first`
- En `gen-mode forecast`, **time‑travel est activé par défaut** si `ALPHALENS_BACKTEST_TIME_TRAVEL=1` (valeur par défaut).  
  Il entraîne/actualise le modèle local à mesure que le backtest avance.

---

**Paramètres utiles (backtest)**
- `--symbol` : instrument (`BTC/USD`, `EUR/USD`, etc.)
- `--timeframe` : alias supportés (`1m`, `5m`, `15m`, `30m`, `45m`, `1h`, `2h`, `4h`, `1d`, `1w` ou `1min`, `30min`, `1day`, `1week`)
- `--lookback` : jours de données
- `--gen-mode` : `simple`, `llm`, `forecast`
- `--risk-appetite` : `auto`, `aggressive`, `moderate`, `conservative`
- `--entry-expiry-bars` : expiration d’entrée
- `--forecast-mode` : `live`, `replay`, `off`
- `--forecast-backend` : `remote`, `local_app`, `local_engine`
- `--forecast-url` : URL d’un moteur local (si remote)
- `--forecast-batch-size` : parallélisme
- `--decision-every` : fréquence d’un trade
- `--max-hold-bars` : durée max d’un trade
- `--fill-policy` : `sl_first` ou `tp_first`
- `--position-mode` : `one_at_time` ou `overlap`

---

**Dépannage rapide**
- `No module named 'pandas'`  
  Installe les dépendances du moteur local, ou utilise le serveur local via HTTP.
- Commandes PowerShell multi‑ligne  
  Le backtick ` doit être **le dernier caractère** de la ligne.
- Lent en local  
  Réduis `--forecast-batch-size` et/ou augmente `--decision-every`.

---

**Notes**
- `app.py` est **volontairement inchangé**.
- Le backtest est piloté depuis `..\backtest_module.py`.
- Les caches sont écrits **fichier par fichier**, même en mode batch.
