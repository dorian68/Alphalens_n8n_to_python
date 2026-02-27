# %%
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# ================= CONFIGURATION =================
API_KEY = "e40fcead02054731aef55d2dfe01cf47"  # Remplace par ta vraie clé
SYMBOL = "EUR/USD"                      # ou "EURUSD" si le format change
INTERVAL = "15min"
START_DATE = "2016-02-20"               # Début : 10 ans en arrière (ajuste si besoin)
END_DATE = datetime.now().strftime("%Y-%m-%d")  # Aujourd'hui

OUTPUT_FILE = "C:\\Users\\Labry\\Documents\\ALPHALENS_N8N_TO_PYTHON\\EURUSD_15min_10years.csv"
BATCH_DAYS = 50                         # ~50 jours → environ 4800 bougies 15min (sous les 5000 max)
SLEEP_BETWEEN_REQUESTS = 1.2            # Délai en secondes pour éviter rate-limit
# ================================================

def fetch_batch(start_date_str, end_date_str):
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "apikey": API_KEY,
        "outputsize": 5000,
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()  # Lève une erreur si statut != 200
        
        data = response.json()
        
        if "values" not in data or not data["values"]:
            print(f"Aucune donnée pour {start_date_str} → {end_date_str}")
            return None
        
        # Conversion en DataFrame
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        
        # Colonnes attendues : open, high, low, close, volume (si présent)
        df = df[["open", "high", "low", "close"]].astype(float)
        
        print(f"Batch récupéré : {start_date_str} → {end_date_str} | {len(df)} bougies")
        return df
    
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP {http_err} sur {start_date_str} → {end_date_str}")
        print("Réponse serveur :", response.text)
        return None
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return None


# ====================== MAIN ======================
print(f"Récupération EUR/USD 15min de {START_DATE} à {END_DATE}")

all_batches = []
current_start = datetime.strptime(START_DATE, "%Y-%m-%d")

while current_start < datetime.now():
    current_end = min(current_start + timedelta(days=BATCH_DAYS), datetime.now())
    
    start_str = current_start.strftime("%Y-%m-%d")
    end_str = current_end.strftime("%Y-%m-%d")
    
    batch_df = fetch_batch(start_str, end_str)
    
    if batch_df is not None and not batch_df.empty:
        all_batches.append(batch_df)
    
    # Avance au lendemain de la fin du batch
    current_start = current_end + timedelta(minutes=15)  # petit overlap pour éviter trous
    
    time.sleep(SLEEP_BETWEEN_REQUESTS)  # Respecte le rate-limit

# Fusion et nettoyage final
if all_batches:
    full_df = pd.concat(all_batches).sort_index()
    # Supprime les doublons éventuels (à cause des overlaps)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    # Sauvegarde
    full_df.to_csv(OUTPUT_FILE)
    print(f"\nSauvegarde terminée : {OUTPUT_FILE}")
    print(f"Total bougies récupérées : {len(full_df)}")
    print(f"Période couverte : {full_df.index.min()} → {full_df.index.max()}")
else:
    print("Aucune donnée récupérée. Vérifie ta clé API, le symbole ou les dates.")
# %%
import pandas as pd

trade_data_path = r"C:\Users\Labry\Documents\ALPHALENS_N8N_TO_PYTHON\alphalens_lambda\trade_data\trades_btc_usd_1h_2025-01-01_2026-01-10_forecast_live_local_engine_moderate_de1_20260224T131729Z.json"
df_trades = pd.read_json(trade_data_path)

tcd_trades = (
    df_trades[df_trades["outcome"] != "timeout"].assign(regime_label=df_trades["regime_label"].fillna("UNKNOWN"))
    .pivot_table(
        index=["regime_label", "outcome"],
        values="pnl",
        aggfunc=["count", "sum", "mean", "median"],
        dropna=False
    )
    .sort_values(("sum", "pnl"), ascending=False)
)

print(" ************************** PERFORMANCE GLOBALE ********************************")
print(tcd_trades)
print(" ************************** PERFORMANCE GLOBALE ********************************")

df_trades.to_csv(r"C:\Users\Labry\Documents\trades_resume1h_v14.csv",decimal=",")
# %%
