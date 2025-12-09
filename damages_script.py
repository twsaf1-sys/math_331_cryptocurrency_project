import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# Configuration
# ============================================================

# Root folder for all output
DATA_ROOT = "digiconomist_data"

# End date for the time series (inclusive, YYYY-MM-DD)
END_DATE_STR = "2025-11-15"

# Digiconomist API endpoints (Bitcoin, Ethereum, Dogecoin)
API_ENDPOINTS = {
    "bitcoin":  "https://digiconomist.net/wp-json/mo/v1/bitcoin/stats/{date}",
    "ethereum": "https://digiconomist.net/wp-json/mo/v1/ethereum/stats/{date}",
    "dogecoin": "https://digiconomist.net/wp-json/mo/v1/dogecoin/stats/{date}",
}

# Start dates for each asset (per Digiconomist docs)
START_DATES = {
    "bitcoin":  "2017-02-10",
    "ethereum": "2017-05-20",
    "dogecoin": "2021-01-01",
}

# Social cost of carbon per tonne (USD)
SOCIAL_COST_PER_TON = 51.0

# Path to the combined BitInfoCharts market price CSV
MARKET_PRICE_CSV_PATH = "market_price_2015_to_present_bitinfocharts.csv"

# Mapping from internal coin name -> column name in market price CSV
MARKET_PRICE_COLUMN_MAP = {
    "bitcoin":  "Bitcoin",
    "ethereum": "Ethereum Classic",
    "dogecoin": "Dogecoin",
}

# Path to Bitcoin total-coins CSV (Date + time, Coins)
BITCOIN_COINS_CSV_PATH = "total-bitcoins-mined.csv"

# Path to BitInfoCharts confirmation-time CSV
CONFIRMATION_TIME_CSV_PATH = "confirmation_time_per_block_2015_to_present_bitinfocharts.csv"


# ============================================================
# Generic helpers
# ============================================================

def daterange(start_date: datetime, end_date: datetime):
    """Yield dates from start_date to end_date inclusive."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def fetch_one_day(coin: str, base_url: str, out_dir: str, date_str: str):
    """
    Worker: fetch one day for one coin and save to JSON if successful.
    Returns (date_str, success_bool).
    """
    out_path = os.path.join(out_dir, f"{date_str}.json")

    # Skip if already exists
    if os.path.exists(out_path):
        return date_str, True  # treat as success

    url = base_url.format(date=date_str)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return date_str, True
        else:
            print(f"[{coin}] {date_str} - HTTP {resp.status_code}")
            return date_str, False
    except Exception as e:
        print(f"[{coin}] {date_str} - ERROR: {e}")
        return date_str, False


def fetch_and_save_json_for_coin_parallel(
    coin: str,
    start_date_str: str,
    end_date_str: str,
    max_workers: int = 16,
):
    """
    Parallel fetch:
    - Builds list of dates
    - Uses ThreadPoolExecutor to fetch many days concurrently.
    """
    base_url = API_ENDPOINTS[coin]
    out_dir = os.path.join(DATA_ROOT, coin)
    os.makedirs(out_dir, exist_ok=True)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str, "%Y-%m-%d")

    all_dates = [d.strftime("%Y%m%d") for d in daterange(start_date, end_date)]

    print(f"[{coin}] Fetching {len(all_dates)} days with up to {max_workers} workers...")

    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_one_day, coin, base_url, out_dir, ds)
            for ds in all_dates
        ]

        for future in as_completed(futures):
            date_str, ok = future.result()
            if ok:
                successes += 1
            else:
                failures += 1

    print(f"[{coin}] Done. Successes: {successes}, Failures: {failures}")


def build_dataframe_from_json_folder(coin: str) -> pd.DataFrame:
    """
    Reads all JSON files for a given coin and returns a DataFrame with:
    date (MM/DD/YYYY), 24hr_kWh, 24hr_kgCO2, Output_KWh, Output_kgCO2,
    tonnes_kg_co2, social_cost_per_ton, daily_social_cost.
    """
    folder = os.path.join(DATA_ROOT, coin)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No folder found for coin '{coin}' at {folder}")

    records = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(folder, filename)
        date_raw = os.path.splitext(filename)[0]  # YYYYMMDD from filename

        # Convert YYYYMMDD -> MM/DD/YYYY for storage in DataFrame/CSV
        try:
            dt = datetime.strptime(date_raw, "%Y%m%d")
            date_formatted = dt.strftime("%m/%d/%Y")
        except ValueError:
            date_formatted = date_raw

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Normalize raw JSON to a dict called `data`
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                data = raw[0]
            else:
                print(f"[{coin}] {date_raw} JSON is an empty/non-dict list; recording NaNs.")
                data = {}
        else:
            print(f"[{coin}] {date_raw} JSON has unexpected type {type(raw)}; recording NaNs.")
            data = {}

        _24hr_kwh   = data.get("24hr_kWh")
        _24hr_kgco2 = data.get("24hr_kgCO2")

        if coin == "ethereum":
            # Ethereum uses gas-unit metrics rather than per-output
            output_kwh   = data.get("Gas_unit_Wh")
            output_kgco2 = data.get("Gas_unit_gCO2")
        else:  # bitcoin and dogecoin
            output_kwh   = data.get("Output_KWh") or data.get("Output_kWh")
            output_kgco2 = data.get("Output_kgCO2")

        def to_float(v):
            try:
                return float(v) if v is not None else None
            except (ValueError, TypeError):
                return None

        val_24hr_kwh      = to_float(_24hr_kwh)
        val_24hr_kgco2    = to_float(_24hr_kgco2)
        val_output_kwh    = to_float(output_kwh)
        val_output_kgco2  = to_float(output_kgco2)
        val_tonnes_kgco2  = val_24hr_kgco2 / 1000 if val_24hr_kgco2 is not None else None

        val_social_cost_per_ton = SOCIAL_COST_PER_TON if val_tonnes_kgco2 is not None else None
        val_daily_social_cost   = (
            val_tonnes_kgco2 * SOCIAL_COST_PER_TON if val_tonnes_kgco2 is not None else None
        )

        records.append({
            "date":                date_formatted,  # MM/DD/YYYY
            "24hr_kWh":            val_24hr_kwh,
            "24hr_kgCO2":          val_24hr_kgco2,
            "Output_KWh":          val_output_kwh,
            "Output_kgCO2":        val_output_kgco2,
            "tonnes_kg_co2":       val_tonnes_kgco2,
            "social_cost_per_ton": val_social_cost_per_ton,
            "daily_social_cost":   val_daily_social_cost,
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


# ============================================================
# Ethereum Classic (ETC) – coins mined per day from BitInfoCharts CSV
# ============================================================

def build_etc_mined_per_day_from_bitinfo() -> pd.DataFrame:
    """
    Build ETC 'coins mined per day' from a local BitInfoCharts confirmation-time CSV.
    """
    if not os.path.exists(CONFIRMATION_TIME_CSV_PATH):
        raise FileNotFoundError(
            f"ETC confirmation-time CSV not found at: {CONFIRMATION_TIME_CSV_PATH}"
        )

    df = pd.read_csv(CONFIRMATION_TIME_CSV_PATH)

    if "Date" not in df.columns:
        raise ValueError("Confirmation-time CSV must contain a 'Date' column")
    if "Ethereum Classic" not in df.columns:
        raise ValueError("Confirmation-time CSV must contain an 'Ethereum Classic' column")

    df["date_dt"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["block_time_minutes"] = pd.to_numeric(df["Ethereum Classic"], errors="coerce")

    df = df.dropna(subset=["date_dt", "block_time_minutes"])

    # Compute approximate blocks per day
    df["blocks_per_day"] = 1440.0 / df["block_time_minutes"]

    # Emission schedule (5M20, eras from your notes)
    eras = [
        (datetime(2015, 7, 30), datetime(2017, 12, 10), 5.0),
        (datetime(2017, 12, 11), datetime(2020, 3, 16), 4.0),
        (datetime(2020, 3, 17), datetime(2022, 4, 24), 3.2),
        (datetime(2022, 4, 25), datetime(2024, 5, 29), 2.56),
        (datetime(2024, 5, 30), datetime(2026, 7, 31), 2.048),
        (datetime(2026, 8, 1), None, 1.6384),  # future era
    ]

    def era_block_reward(d: pd.Timestamp) -> float:
        for start, end, reward in eras:
            if end is None:
                if d >= start:
                    return reward
            else:
                if start <= d <= end:
                    return reward
        return np.nan

    df["block_reward"] = df["date_dt"].apply(era_block_reward)
    df["number_coins"] = df["blocks_per_day"] * df["block_reward"]

    df_daily = (
        df.dropna(subset=["number_coins"])
          .groupby("date_dt", as_index=False)["number_coins"]
          .sum()
    )

    print("[ethereum] Sample ETC mined/day from CSV:")
    print(df_daily.head())

    return df_daily


# ============================================================
# Dogecoin – coins mined per day from BitInfoCharts CSV
# ============================================================

def build_doge_mined_per_day_from_bitinfo() -> pd.DataFrame:
    """
    Build DOGE 'coins mined per day' from the BitInfoCharts confirmation-time CSV.

    Uses the 'Dogecoin' column as average block time (minutes) and applies:
        blocks_per_day(d) = 1440 / block_time_minutes(d)

    Block reward schedule (academic expectation):
        - 250,000 DOGE per block for blocks < 600,000  (before Feb 2014)
        - 10,000 DOGE per block thereafter

    For the 2015–present dataset, all dates are after block 600,000,
    so the effective reward is 10,000 DOGE per block on every row.
    """
    if not os.path.exists(CONFIRMATION_TIME_CSV_PATH):
        raise FileNotFoundError(
            f"Dogecoin confirmation-time CSV not found at: {CONFIRMATION_TIME_CSV_PATH}"
        )

    df = pd.read_csv(CONFIRMATION_TIME_CSV_PATH)

    if "Date" not in df.columns:
        raise ValueError("Confirmation-time CSV must contain a 'Date' column")
    if "Dogecoin" not in df.columns:
        raise ValueError("Confirmation-time CSV must contain a 'Dogecoin' column")

    df["date_dt"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["block_time_minutes"] = pd.to_numeric(df["Dogecoin"], errors="coerce")

    df = df.dropna(subset=["date_dt", "block_time_minutes"])

    # Approximate blocks per day
    df["blocks_per_day"] = 1440.0 / df["block_time_minutes"]

    # Block 600,000 occurred in February 2014
    block_600000_date = datetime(2014, 2, 1)

    def doge_block_reward(d: pd.Timestamp) -> float:
        if d < block_600000_date:
            return 250000.0
        else:
            return 10000.0

    df["block_reward"] = df["date_dt"].apply(doge_block_reward)
    df["number_coins"] = df["blocks_per_day"] * df["block_reward"]

    df_daily = (
        df.dropna(subset=["number_coins"])
          .groupby("date_dt", as_index=False)["number_coins"]
          .sum()
    )

    print("[dogecoin] Sample DOGE mined/day from CSV:")
    print(df_daily.head())

    return df_daily


# ============================================================
# Ethereum & Dogecoin: add number_coins column from BitInfoCharts
# ============================================================

def add_number_coins_column(df: pd.DataFrame, coin: str) -> pd.DataFrame:
    """
    For Ethereum (treated as ETC here):
      - Build ETC mined-per-day series.
      - Merge onto the Digiconomist Ethereum dataframe as 'number_coins'.

    For Dogecoin:
      - Build DOGE mined-per-day series from confirmation-time CSV.
      - Merge onto the Digiconomist Dogecoin dataframe as 'number_coins'.

    For other coins (bitcoin):
      - Set number_coins = 0.0
    """
    df = df.copy()
    coin_lower = coin.lower()

    # Ethereum -> ETC series
    if coin_lower == "ethereum":
        try:
            etc_daily = build_etc_mined_per_day_from_bitinfo()
        except Exception as e:
            print(f"[ethereum] Error building ETC mined-per-day series: {e}")
            df["number_coins"] = 0.0
            return df

        df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.normalize()

        merged = df.merge(etc_daily, how="left", on="date_dt")
        merged["number_coins"] = pd.to_numeric(merged["number_coins"], errors="coerce").fillna(0.0)

        print("[ethereum] number_coins non-zero count:", (merged["number_coins"] > 0).sum())
        return merged

    # Dogecoin -> DOGE series
    if coin_lower == "dogecoin":
        try:
            doge_daily = build_doge_mined_per_day_from_bitinfo()
        except Exception as e:
            print(f"[dogecoin] Error building DOGE mined-per-day series: {e}")
            df["number_coins"] = 0.0
            return df

        df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.normalize()

        # Left-merge to retain the rectangular shape of the Dogecoin dataframe
        merged = df.merge(doge_daily, how="left", on="date_dt")
        merged["number_coins"] = pd.to_numeric(merged["number_coins"], errors="coerce").fillna(0.0)

        print("[dogecoin] number_coins non-zero count:", (merged["number_coins"] > 0).sum())
        return merged

    # Default: other coins (e.g., bitcoin) get zeros
    df["number_coins"] = 0.0
    return df


# ============================================================
# Bitcoin total coins & coins mined from total-bitcoins-mined.csv
# ============================================================

def add_bitcoin_coins_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Bitcoin:
      - Load total-bitcoins-mined.csv
      - Expect columns 'Date' (timestamp) and 'Coins'
      - Convert 'Date' to datetime and normalise to date-only
      - Merge daily Coins and compute Coins_mined
    """
    df = df.copy()

    if not os.path.exists(BITCOIN_COINS_CSV_PATH):
        print(f"[bitcoin] Bitcoin coins CSV not found at {BITCOIN_COINS_CSV_PATH}; filling zeros.")
        df["Coins"] = 0.0
        df["Coins_mined"] = 0.0
        return df

    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.normalize()

    try:
        btc_coins = pd.read_csv(BITCOIN_COINS_CSV_PATH)
    except Exception as e:
        print(f"[bitcoin] Error reading {BITCOIN_COINS_CSV_PATH}: {e}; filling zeros.")
        df["Coins"] = 0.0
        df["Coins_mined"] = 0.0
        return df

    if "Date" not in btc_coins.columns or "Coins" not in btc_coins.columns:
        print("[bitcoin] Bitcoin coins CSV must contain 'Date' and 'Coins' columns; filling zeros.")
        df["Coins"] = 0.0
        df["Coins_mined"] = 0.0
        return df

    btc_coins["Date_dt"] = pd.to_datetime(btc_coins["Date"], errors="coerce")
    btc_coins = btc_coins.dropna(subset=["Date_dt"])
    btc_coins["date_dt"] = btc_coins["Date_dt"].dt.normalize()

    btc_daily = (
        btc_coins
        .sort_values("Date_dt")
        .groupby("date_dt", as_index=False)
        .last()
    )

    btc_daily["Coins"] = pd.to_numeric(btc_daily["Coins"], errors="coerce").fillna(0.0)

    merged = df.merge(btc_daily[["date_dt", "Coins"]], how="left", on="date_dt")
    merged["Coins"] = merged["Coins"].fillna(0.0)

    merged = merged.sort_values("date_dt").reset_index(drop=True)
    merged["Coins_mined"] = merged["Coins"].diff().fillna(0.0)
    merged["Coins_mined"] = merged["Coins_mined"].clip(lower=0.0)

    return merged


# ============================================================
# CEDI_t computation for Bitcoin, Ethereum, and Dogecoin
# ============================================================

def add_cedi_column(df: pd.DataFrame, coin: str) -> pd.DataFrame:
    """
    Compute the Cryptocurrency Environmental Damage Index (CEDI_t) for a given coin:
        CEDI_t = (daily_social_cost * number_of_coins_mined) / market_price

    daily_social_cost : 'daily_social_cost'
    number_of_coins   : 'Coins_mined' (bitcoin) or 'number_coins' (ethereum/dogecoin)
    market_price      : 'market_price'
    """
    df = df.copy()
    coin_lower = coin.lower()

    if df.empty:
        df["CEDI_t"] = pd.NA
        df["CEDI_pct_market_price"] = pd.NA
        return df

    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("date_dt").reset_index(drop=True)

    df["CEDI_t"] = pd.NA
    df["CEDI_pct_market_price"] = pd.NA

    if coin_lower not in ("bitcoin", "ethereum", "dogecoin"):
        return df

    if coin_lower == "bitcoin":
        nc_col = "Coins_mined"
        if nc_col not in df.columns:
            return df
        if not df[nc_col].empty:
            # Approximate first-day BTC mined for missing diff
            df.loc[df.index[0], nc_col] = 1737.5
    else:
        nc_col = "number_coins"
        if nc_col not in df.columns:
            return df

    df["daily_social_cost"] = pd.to_numeric(df["daily_social_cost"], errors="coerce")
    df[nc_col] = pd.to_numeric(df[nc_col], errors="coerce")
    df["market_price"] = pd.to_numeric(df["market_price"], errors="coerce")

    valid_mask = (
        (df["market_price"] > 0)
        & df["daily_social_cost"].notna()
        & df[nc_col].notna()
    )

    df.loc[valid_mask, "CEDI_t"] = (
        df.loc[valid_mask, "daily_social_cost"].astype(float)
        * df.loc[valid_mask, nc_col].astype(float)
        / df.loc[valid_mask, "market_price"].astype(float)
    )

    valid_pct_mask = (df["CEDI_t"].notna()) & (df["market_price"] > 0)
    df.loc[valid_pct_mask, "CEDI_pct_market_price"] = (
        df.loc[valid_pct_mask, "CEDI_t"].astype(float)
        / df.loc[valid_pct_mask, "market_price"].astype(float)
    ) * 100.0

    return df


# ============================================================
# Market price from combined BitInfoCharts CSV
# ============================================================

def load_market_price_df(path: str = MARKET_PRICE_CSV_PATH) -> pd.DataFrame:
    """
    Load the combined market price CSV from BitInfoCharts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Market price CSV not found at: {path}")

    mp = pd.read_csv(path)
    if "Date" not in mp.columns:
        raise ValueError("Market price CSV must contain a 'Date' column")

    mp["Date_dt"] = pd.to_datetime(mp["Date"], errors="coerce")
    mp = mp.dropna(subset=["Date_dt"])
    return mp


def add_market_price_column(df: pd.DataFrame, coin: str, mp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'market_price' column to df for the given coin, using BitInfoCharts CSV.
    """
    col_name = MARKET_PRICE_COLUMN_MAP.get(coin)
    if col_name is None:
        print(f"[{coin}] No market price column mapping configured; filling 0.")
        df["market_price"] = 0.0
        return df

    if col_name not in mp_df.columns:
        print(f"[{coin}] Column '{col_name}' not found in market price CSV; filling 0.")
        df["market_price"] = 0.0
        return df

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    temp = mp_df[["Date_dt", col_name]].copy()
    temp = temp.rename(columns={"Date_dt": "date_dt", col_name: "market_price"})
    temp["market_price"] = pd.to_numeric(temp["market_price"], errors="coerce").fillna(0.0)

    merged = df.merge(temp, how="left", on="date_dt")
    merged["market_price"] = merged["market_price"].fillna(0.0)
    return merged


# ============================================================
# Save & plotting helpers
# ============================================================

def save_dataframe_to_csv(df: pd.DataFrame, coin: str):
    csv_path = os.path.join(DATA_ROOT, f"{coin}_digiconomist_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{coin}] DataFrame saved to {csv_path}")


def plot_coin_timeseries(df: pd.DataFrame, coin: str):
    """
    Combined multi-series plot per coin.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt"])

    label_map_primary = {
        "24hr_kWh":     "Total energy consumption (kWh, 24h)",
        "24hr_kgCO2":   "Total carbon footprint (kg CO₂, 24h)",
        "Output_KWh":   "Average energy per output / gas unit",
        "Output_kgCO2": "Average carbon per output / gas unit",
    }

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()

    for col in ["24hr_kWh", "24hr_kgCO2", "Output_KWh", "Output_kgCO2"]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().any():
                ax1.plot(df["date_dt"], y, label=label_map_primary.get(col, col))

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Energy / Emissions (kWh, kg CO₂, per output/gas)")
    ax1.set_title(f"{coin.capitalize()} – Digiconomist Energy & Emissions Metrics Over Time")
    ax1.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()

    if "tonnes_kg_co2" in df.columns:
        y2 = pd.to_numeric(df["tonnes_kg_co2"], errors="coerce")
        if y2.notna().any():
            ax2 = ax1.twinx()
            line2, = ax2.plot(
                df["date_dt"],
                y2,
                linestyle="--",
                label="Total carbon footprint (tonnes CO₂, 24h)",
            )
            ax2.set_ylabel("Tonnes CO₂ (24h)")
            lines2 = [line2]
            labels2 = ["Total carbon footprint (tonnes CO₂, 24h)"]
        else:
            lines2, labels2 = [], []
    else:
        lines2, labels2 = [], []

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(DATA_ROOT, f"{coin}_timeseries_combined.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[{coin}] Combined time series plot saved to {out_path}")


def plot_individual_metric_timeseries(df: pd.DataFrame, coin: str):
    """
    For each selected numeric column, create a separate x–y plot.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt"])

    metric_columns = [
        "24hr_kWh",
        "24hr_kgCO2",
        "tonnes_kg_co2",
        "daily_social_cost",
        "Output_KWh",
        "Output_kgCO2",
        "number_coins",
        "market_price",
        "Coins",
        "Coins_mined",
        "CEDI_t",
        "CEDI_pct_market_price",
    ]

    pretty_labels = {
        "24hr_kWh":              "Total energy consumption (kWh, 24h)",
        "24hr_kgCO2":            "Total carbon footprint (kg CO₂, 24h)",
        "tonnes_kg_co2":         "Total carbon footprint (tonnes CO₂, 24h)",
        "daily_social_cost":     "Daily social cost of emissions (USD)",
        "Output_KWh":            "Average energy per output / gas unit (kWh / unit)",
        "Output_kgCO2":          "Average carbon per output / gas unit (kg CO₂ / unit)",
        "number_coins":          "Coins mined per day",
        "market_price":          "Market price (USD)",
        "Coins":                 "Total bitcoins in circulation (coins)",
        "Coins_mined":           "Bitcoins mined per day (coins)",
        "CEDI_t":                "CEDI_t (damage index, USD)",
        "CEDI_pct_market_price": "CEDI_t as % of market price",
    }

    for col in metric_columns:
        if col not in df.columns:
            continue

        y = pd.to_numeric(df[col], errors="coerce")
        if not y.notna().any():
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(df["date_dt"], y)

        plt.xlabel("Date")
        plt.ylabel(pretty_labels.get(col, col))
        plt.title(f"{coin.capitalize()} – {pretty_labels.get(col, col)} over time")
        plt.grid(True)

        plt.tight_layout()
        safe_col = col.replace("/", "_per_").replace(" ", "_")
        out_path = os.path.join(DATA_ROOT, f"{coin}_{safe_col}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[{coin}] Individual metric plot saved to {out_path}")


def plot_cedi_timeseries(df: pd.DataFrame, coin: str):
    """
    Plot raw CEDI_t for this cryptocurrency as a standalone graph.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt", "CEDI_t"])

    y = pd.to_numeric(df["CEDI_t"], errors="coerce")
    if not y.notna().any():
        print(f"[{coin}] No valid CEDI_t values to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df["date_dt"], y, linewidth=1.5)

    plt.xlabel("Date")
    plt.ylabel("CEDI_t (USD damage index)")
    plt.title(f"{coin.capitalize()} – Raw CEDI_t Over Time")
    plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(DATA_ROOT, f"{coin}_CEDI_t_timeseries.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[{coin}] Raw CEDI_t plot saved to {out_path}")


def plot_cedi_pct_timeseries(df: pd.DataFrame, coin: str):
    """
    Plot CEDI as a percentage of market price over time.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date_dt", "CEDI_pct_market_price"])

    y = pd.to_numeric(df["CEDI_pct_market_price"], errors="coerce")
    if not y.notna().any():
        print(f"[{coin}] No valid CEDI_pct_market_price values to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df["date_dt"], y, linewidth=1.5)

    plt.xlabel("Date")
    plt.ylabel("CEDI_t as % of Market Price")
    plt.title(f"{coin.capitalize()} – CEDI % of Market Price Over Time")
    plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(DATA_ROOT, f"{coin}_CEDI_pct_timeseries.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[{coin}] CEDI % plot saved to {out_path}")


def plot_cedi_across_coins(all_coin_dfs: dict):
    """
    Create a single plot comparing CEDI_t across all cryptocurrencies.
    """
    plt.figure(figsize=(12, 6))

    for coin, df in all_coin_dfs.items():
        if "CEDI_t" not in df.columns:
            continue

        temp = df.copy()
        temp["date_dt"] = pd.to_datetime(temp["date"], format="%m/%d/%Y", errors="coerce")
        temp = temp.dropna(subset=["date_dt", "CEDI_t"])
        y = pd.to_numeric(temp["CEDI_t"], errors="coerce")

        if not y.notna().any():
            continue

        plt.plot(temp["date_dt"], y, label=coin.capitalize())

    plt.xlabel("Date")
    plt.ylabel("CEDI_t (damage index, USD)")
    plt.title("Cryptocurrency Environmental Damage Index (CEDI_t) – Comparison")
    plt.grid(True)
    plt.legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(DATA_ROOT, "CEDI_all_coins_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[ALL] CEDI comparison plot saved to {out_path}")


# ============================================================
# Part 4 – Forecasting models (ARIMA & ETS) for each coin
# ============================================================

def run_cedi_forecasts_for_coin(
    df_coin: pd.DataFrame,
    coin: str,
    horizon_days: int = 180,
    innovation_factor: float = 0.7,
    arima_order=(1, 1, 1),
):
    """
    Apply ARIMA(p,d,q) and Exponential Smoothing (ETS) to the coin's CEDI_t series.
    Forecast 6 months ahead under baseline + innovation scenarios.
    Compare ARIMA vs ETS performance (AIC, RMSE, MAE) on a holdout window.
    """
    df = df_coin.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values("date_dt")

    series = (
        df.loc[df["CEDI_t"].notna(), ["date_dt", "CEDI_t"]]
        .dropna(subset=["date_dt", "CEDI_t"])
        .set_index("date_dt")["CEDI_t"]
    )
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) < 50:
        print(f"[{coin}] Not enough CEDI_t data to run time-series models (len={len(series)}).")
        return

    h = min(horizon_days, max(1, len(series) // 4))

    train = series.iloc[:-h]
    test = series.iloc[-h:]

    print(f"[{coin}] Fitting ARIMA{arima_order} on CEDI_t (train length={len(train)}, test length={len(test)})")
    arima_model = ARIMA(train, order=arima_order)
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=h)

    print(f"[{coin}] Fitting ETS (additive, damped trend) on CEDI_t")
    ets_model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal=None,
    )
    ets_fit = ets_model.fit(optimized=True)
    ets_pred = ets_fit.forecast(steps=h)

    def rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def mae(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean(np.abs(y_true - y_pred)))

    metrics = {
        "model": ["ARIMA", "ETS"],
        "AIC":   [arima_fit.aic, ets_fit.aic],
        "RMSE":  [rmse(test, arima_pred), rmse(test, ets_pred)],
        "MAE":   [mae(test, arima_pred), mae(test, ets_pred)],
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(DATA_ROOT, f"{coin}_cedi_forecast_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[{coin}] Forecast metrics saved to {metrics_path}")

    print(f"[{coin}] Refitting models on full CEDI_t series for {horizon_days}-day forecasts...")
    arima_full = ARIMA(series, order=arima_order).fit()
    ets_full = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal=None,
    ).fit(optimized=True)

    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    arima_fc = arima_full.forecast(steps=horizon_days)
    ets_fc   = ets_full.forecast(steps=horizon_days)

    forecasts_df = pd.DataFrame({
        "date": future_index,
        "CEDI_ARIMA_baseline": arima_fc.values,
        "CEDI_ETS_baseline": ets_fc.values,
    })

    forecasts_df["CEDI_ARIMA_innovation"] = forecasts_df["CEDI_ARIMA_baseline"] * innovation_factor
    forecasts_df["CEDI_ETS_innovation"]   = forecasts_df["CEDI_ETS_baseline"]   * innovation_factor

    forecast_csv_path = os.path.join(DATA_ROOT, f"{coin}_cedi_forecasts_6m.csv")
    forecasts_df.to_csv(forecast_csv_path, index=False)
    print(f"[{coin}] 6-month forecasts saved to {forecast_csv_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="Historical CEDI_t", linewidth=1.5)
    plt.plot(forecasts_df["date"], forecasts_df["CEDI_ARIMA_baseline"], label="ARIMA baseline")
    plt.plot(forecasts_df["date"], forecasts_df["CEDI_ETS_baseline"], label="ETS baseline")
    plt.plot(
        forecasts_df["date"],
        forecasts_df["CEDI_ARIMA_innovation"],
        "--",
        label=f"ARIMA innovation (×{innovation_factor:.2f})",
    )
    plt.plot(
        forecasts_df["date"],
        forecasts_df["CEDI_ETS_innovation"],
        "--",
        label=f"ETS innovation (×{innovation_factor:.2f})",
    )

    plt.xlabel("Date")
    plt.ylabel("CEDI_t (damage index, USD)")
    plt.title(f"{coin.capitalize()} CEDI_t – ARIMA vs ETS forecast (baseline & innovation)")
    plt.grid(True)
    plt.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(DATA_ROOT, f"{coin}_cedi_forecasts_6m.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[{coin}] Forecast plot saved to {plot_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    end_date_str = END_DATE_STR

    # Ensure output root exists
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Load market price dataframe once
    market_price_df = load_market_price_df(MARKET_PRICE_CSV_PATH)

    # 1) Parallel fetch for each coin (Bitcoin, Ethereum, Dogecoin)
    for coin, start_date_str in START_DATES.items():
        print(f"\n=== Fetching {coin} data (parallel) ===")
        fetch_and_save_json_for_coin_parallel(
            coin=coin,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            max_workers=16,
        )

    # Store final DFs so we can plot CEDI comparison later
    all_coin_dfs = {}

    # 2) Build DataFrames, add coin-specific columns, save CSVs, and create plots
    for coin in START_DATES.keys():
        print(f"\n=== Building DataFrame and plotting for {coin} ===")
        df_coin = build_dataframe_from_json_folder(coin)

        # Ethereum & Dogecoin: coins/day from confirmation-time CSV
        df_coin = add_number_coins_column(df_coin, coin)

        # Bitcoin: total coins + coins mined
        if coin == "bitcoin":
            df_coin = add_bitcoin_coins_column(df_coin)

        # Market price from BitInfoCharts CSV
        df_coin = add_market_price_column(df_coin, coin, market_price_df)

        # Compute CEDI_t and CEDI_pct_market_price for this coin
        df_coin = add_cedi_column(df_coin, coin)

        # Save final DataFrame for this coin
        save_dataframe_to_csv(df_coin, coin)

        # Plots per coin
        plot_coin_timeseries(df_coin, coin)
        plot_individual_metric_timeseries(df_coin, coin)
        plot_cedi_timeseries(df_coin, coin)
        plot_cedi_pct_timeseries(df_coin, coin)

        all_coin_dfs[coin] = df_coin

        # Part 4 – Forecasting models for this coin
        run_cedi_forecasts_for_coin(df_coin, coin)

    # 3) Combined CEDI plot across all cryptocurrencies
    plot_cedi_across_coins(all_coin_dfs)
