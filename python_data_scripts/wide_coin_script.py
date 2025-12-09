import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# ---------------- Config ----------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Friendly name -> slug used in BitInfoCharts comparison URL
COINS = {
    "Bitcoin": "bitcoin",
    "Litecoin": "litecoin",
    "Monero": "monero",
    "Dash": "dash",
    "ZCash": "zcash",
    "Bitcoin Cash": "bitcoin-cash",
    "Ethereum Classic": "ethereum-classic",
    "Bitcoin Gold": "bitcoin-gold",
    "Dogecoin": "dogecoin",
    "Vertcoin": "vertcoin",
}

START_DATE = "2015-01-01"      # inclusive
END_DATE   = "2025-10-31"      # change if you want “today” instead

# ---------------- Scraper ----------------


def fetch_bitinfo_price_series(slug: str) -> pd.DataFrame:
    """
    Fetch daily USD price series for a coin from BitInfoCharts.

    Returns a DataFrame with columns:
        date (datetime64[ns])
        price_usd (float)
    """
    url = f"https://bitinfocharts.com/comparison/{slug}-price.html#alltime"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    scripts = soup.find_all("script")

    # On BitInfoCharts, the 5th <script> tag usually holds the data array
    script_text = scripts[4].text

    # Pattern like: [new Date("2015/01/01"),123.45]
    pattern = re.compile(
        r'\[new Date\("(\d{4}/\d{2}/\d{2})"\),([\d\.]+)\]'
    )
    records = pattern.findall(script_text)

    dates, prices = [], []
    for d, v in records:
        # parse the date string explicitly
        dates.append(pd.to_datetime(d, format="%Y/%m/%d"))
        prices.append(float(v))

    df = pd.DataFrame({"date": dates, "price_usd": prices})

    # ---- IMPORTANT FIX: force date column & bounds to datetime ----
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    start_ts = pd.to_datetime(START_DATE)
    end_ts = pd.to_datetime(END_DATE)

    df = df.loc[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
    return df


def build_price_panel() -> pd.DataFrame:
    """
    Returns a wide DataFrame:

        date | Bitcoin | Litecoin | ... | Vertcoin

    where each coin column is the daily average price in USD.
    """
    all_frames = []

    for coin_name, slug in COINS.items():
        print(f"Fetching price data for {coin_name}...")
        df_coin = fetch_bitinfo_price_series(slug)
        df_coin.rename(columns={"price_usd": coin_name}, inplace=True)
        all_frames.append(df_coin)

    # Start with the first one and outer-join the others
    prices = all_frames[0]
    for df_coin in all_frames[1:]:
        prices = prices.merge(df_coin, on="date", how="outer")

    prices.sort_values("date", inplace=True)
    prices.reset_index(drop=True, inplace=True)
    return prices


# ---------------- Main ----------------

if __name__ == "__main__":
    prices_df = build_price_panel()
    prices_df.to_csv("daily_prices_2015_2025.csv", index=False)
    print("Saved daily_prices_2015_2025.csv")
