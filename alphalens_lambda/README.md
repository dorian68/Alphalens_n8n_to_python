# AlphaLens Lambda + Backtest (Local)

**Vue d'ensemble**  
Ce dépôt contient le service `FastAPI` de génération de trades (fichier `app.py`) et s’appuie sur un module de backtest externe `..\backtest_module.py`.  
L’objectif principal est de **générer des setups de trade** (ou d’exécuter des forecasts) et **backtester** ces niveaux (entry/SL/TP) sur les données Twelve Data.

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

**Backtest rapide (replay)**
```powershell
python ..\backtest_module.py --symbol "BTC/USD" --timeframe 1h --lookback 720 --gen-mode forecast --forecast-mode replay --risk-appetite aggressive --entry-expiry-bars 4 --env-file .\.env
```

---

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

**Paramètres utiles (backtest)**
- `--symbol` : instrument (`BTC/USD`, `EUR/USD`, etc.)
- `--timeframe` : `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`
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
