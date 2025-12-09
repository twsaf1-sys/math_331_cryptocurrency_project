import requests
import re
import csv
from pathlib import Path

# -----------------------------
# 1. PARAMETERS / CONSTANTS
# -----------------------------

# Social cost of carbon (SCC) in USD/ton CO2 – your project’s baseline
SCC_PER_TON = 51.0

# Digiconomist page that states emissions per Bitcoin mined
DIGICONOMIST_BTC_GOLD_URL = "https://digiconomist.net/bitcoin-mining-more-polluting-than-gold-mining/"

# Blockchain.com stats API – snapshot of last ~24h
BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats?cors=true"

# -----------------------------
# 2. SCRAPE EMISSIONS PER BITCOIN (Digiconomist)
# -----------------------------

def get_tonnes_co2_per_bitcoin():
    """
    Scrape Digiconomist's 'Bitcoin vs Gold' page to find
    the 'Mining One Bitcoin ... X tonnes' CO2 figure.
    Returns tonnes CO2 per BTC (float).
    """
    resp = requests.get(DIGICONOMIST_BTC_GOLD_URL, timeout=30)
    resp.raise_for_status()
    text = resp.text

    match = re.search(r"Mining One Bitcoin\s*([\d.,]+)\s*tonnes", text, re.IGNORECASE)
    if not match:
        raise RuntimeError("Could not find 'Mining One Bitcoin ... tonnes' pattern on Digiconomist page.")

    tonnes_str = match.group(1).replace(",", "")
    tonnes = float(tonnes_str)
    return tonnes

def compute_dscc_per_bitcoin(tonnes_co2_per_btc, scc_per_ton=SCC_PER_TON):
    """
    Daily social cost per coin (USD/coin) based on
    tonnes CO2 per Bitcoin and SCC (USD/ton).
    """
    return tonnes_co2_per_btc * scc_per_ton

# -----------------------------
# 3. SCRAPE NC_t AND MV_t FROM BLOCKCHAIN.COM
# -----------------------------

def get_blockchain_stats():
    """
    Get last-24h Bitcoin network stats from Blockchain.com.
    Returns dict with:
        - n_btc_mined (BTC/day)
        - miners_revenue_usd (USD/day)
    """
    resp = requests.get(BLOCKCHAIN_STATS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    n_btc_mined_sats = data.get("n_btc_mined")
    miners_revenue_usd = data.get("miners_revenue_usd")

    if n_btc_mined_sats is None or miners_revenue_usd is None:
        raise RuntimeError("Missing n_btc_mined or miners_revenue_usd in Blockchain stats response.")

    n_btc_mined = n_btc_mined_sats / 1e8  # convert satoshis → BTC

    return {
        "n_btc_mined": n_btc_mined,
        "miners_revenue_usd": float(miners_revenue_usd),
    }

# -----------------------------
# 4. COMPUTE CEDI_t
# -----------------------------

def compute_cedi_today():
    # (1) CO2 per Bitcoin mined from Digiconomist
    tonnes_per_btc = get_tonnes_co2_per_bitcoin()
    dscc_per_btc = compute_dscc_per_bitcoin(tonnes_per_btc)

    # (2) Coins mined per day & miner revenue from Blockchain.com
    stats = get_blockchain_stats()
    n_btc_mined = stats["n_btc_mined"]          # N C_t
    miners_revenue_usd = stats["miners_revenue_usd"]  # M V_t

    # (3) CEDI_t = DSCC_t * N C_t / M V_t
    CEDI_t = dscc_per_btc * n_btc_mined / miners_revenue_usd

    return {
        "tonnes_co2_per_btc": tonnes_per_btc,
        "dscc_per_btc_usd": dscc_per_btc,
        "n_btc_mined_per_day": n_btc_mined,
        "miners_revenue_usd_per_day": miners_revenue_usd,
        "CEDI_today": CEDI_t,
    }

# -----------------------------
# 5. SAVE RESULT TO CSV
# -----------------------------

def save_cedi_to_csv(results_dict, filename="bitcoin_cedi_today.csv"):
    """
    Save the results from compute_cedi_today() to a one-row CSV file.
    Overwrites the file if it already exists.
    """
    path = Path(filename)
    # keys = column names, one row of values
    fieldnames = list(results_dict.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(r
