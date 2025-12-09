import re
from urllib.parse import quote

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.asicminervalue.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Map the coins you care about to the Top Coin filter code
TOP_COINS = {
    "Bitcoin": "BTC",
    "Litecoin": "DOGE+LTC",   # Scrypt miners (DOGE+LTC)
    "Monero": "XMR",
    "Dash": "DASH",
    "ZCash": "ZEC+ZEN",
    "Bitcoin Cash": "BCH",
    "Ethereum Classic": "ETC",
    "Bitcoin Gold": "BTG",
    "Dogecoin": "DOGE+LTC",
    "Vertcoin": "VTC",
}

money_re = re.compile(r"\$([\d,]+(?:\.\d+)?)")


def parse_dollar(text: str) -> float | None:
    """Return first $… value in text as float, or None."""
    if not text:
        return None
    m = money_re.search(text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def _safe_slug(name: str) -> str:
    """'Ethereum Classic' -> 'ethereum_classic', 'Bitcoin (Gold)' -> 'bitcoin_gold'."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def scrape_coin_list_page(coin_name: str, top_coin_code: str) -> pd.DataFrame:
    """
    Scrape the main miners table for a given Top Coin and return a DataFrame
    with one row per miner (many rows, like in your screenshot).
    """
    url = f"{BASE_URL}/?topCoin={quote(top_coin_code)}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # --- find the table header and map column names -> indexes ---
    thead = soup.find("thead")
    if thead is None:
        raise RuntimeError(f"No table header found for {coin_name} at {url}")

    header_cells = [th.get_text(strip=True).lower() for th in thead.find_all("th")]
    col_index = {name: i for i, name in enumerate(header_cells)}

    def find_col(needle: str) -> int:
        """Find first column whose header contains needle."""
        needle = needle.lower()
        for name, idx in col_index.items():
            if needle in name:
                return idx
        raise KeyError(f"Could not find column containing '{needle}' in headers {header_cells}")

    model_idx = find_col("model")
    release_idx = find_col("release")
    hashrate_idx = find_col("hashrate")
    power_idx = find_col("power")
    algo_idx = find_col("algorithm")
    best_price_idx = find_col("best price")
    profit_idx = find_col("profit")

    # --- walk all rows in the tbody ---
    rows = []
    tbody = soup.find("tbody")
    if not tbody:
        raise RuntimeError(f"No <tbody> found for {coin_name} at {url}")

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) <= profit_idx:
            continue  # skip non-data rows

        def cell(idx: int) -> str:
            return tds[idx].get_text(" ", strip=True)

        model = cell(model_idx)
        release = cell(release_idx)
        hashrate = cell(hashrate_idx)
        power = cell(power_idx)
        algorithm = cell(algo_idx)
        best_price_text = cell(best_price_idx)
        profit_text = cell(profit_idx)

        # Skip junk rows if any
        if not model:
            continue

        best_price_usd = parse_dollar(best_price_text)
        profit_usd_per_day = parse_dollar(profit_text)

        rows.append(
            {
                "coin": coin_name,
                "model": model,
                "release": release,
                "hashrate": hashrate,
                "power": power,
                "algorithm": algorithm,
                "best_price_text": best_price_text,
                "best_price_usd": best_price_usd,
                "profit_text": profit_text,
                "profit_usd_per_day": profit_usd_per_day,
            }
        )

    return pd.DataFrame(rows)



all_dfs: dict[str, pd.DataFrame] = {}

for coin_name, code in TOP_COINS.items():
    print(f"Scraping table for {coin_name} (TopCoin={code})...")
    try:
        df_coin = scrape_coin_list_page(coin_name, code)
        all_dfs[coin_name] = df_coin

        slug = _safe_slug(coin_name)
        fname = f"{slug}_asic_miners.csv"
        df_coin.to_csv(fname, index=False)

        print(f"  {len(df_coin)} rows scraped, saved to {fname}\n")
    except Exception as e:
        print(f"  ERROR for {coin_name}: {e}\n")
