#!/usr/bin/env python3
"""
Scrape block rewards from BitInfoCharts for:

Bitcoin (BTC), Litecoin (LTC), Monero (XMR), Dash (DASH), ZCash (ZEC),
Bitcoin Cash (BCH), Ethereum Classic (ETC),
Bitcoin Gold (BTG), Dogecoin (DOGE), Vertcoin (VTC).

Output: pandas DataFrame with subsidy, avg fees, and total reward per block.
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Map tickers to BitInfoCharts coin page path fragments
# (spaces will be URL-encoded with quote()).
COIN_PATHS = {
    "BTC": "bitcoin",
    "LTC": "litecoin",
    "XMR": "monero",
    "DASH": "dash",
    "ZEC": "zcash",
    "BCH": "bitcoin%20cash",
    "ETC": "ethereum%20classic",
    "BTG": "bitcoin%20gold",
    "DOGE": "dogecoin",
    "VTC": "vertcoin",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# ----------------------------------------------------------------------
# Core scraping logic
# ----------------------------------------------------------------------

def fetch_current_block_rewards():
    """
    Scrape 'Reward Per Block' for each coin from BitInfoCharts
    and return a DataFrame with subsidy and fee components.

    Columns:
      - subsidy_per_block
      - avg_fee_per_block
      - total_reward_per_block
      - source_url
    Index:
      - ticker (BTC, LTC, etc.)
    """
    rows = []

    for ticker, name in COIN_PATHS.items():
        # Build stats page URL, e.g. https://bitinfocharts.com/bitcoin/
        url = f"https://bitinfocharts.com/{quote(name)}/"
        print(f"[INFO] Fetching {ticker} from {url}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"[WARN] HTTP error for {ticker}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        # Find the text node that contains "Reward Per Block"
        label_node = soup.find(string=re.compile(r"Reward Per Block"))
        if label_node is None:
            print(f"[WARN] Could not find 'Reward Per Block' for {ticker} at {url}")
            continue

        # The numeric part is usually in the same parent element
        line_text = label_node.parent.get_text(" ", strip=True)
        # Example format:
        # "Reward Per Block 3.125+0.02101 BTC ($267,611.69) next halving ..."
        m = re.search(
            r"Reward Per Block\s+([0-9.]+)(?:\+([0-9.]+))?",
            line_text
        )
        if not m:
            print(f"[WARN] Could not parse reward numbers for {ticker}: {line_text}")
            continue

        subsidy = float(m.group(1))  # base block subsidy
        fee_component = float(m.group(2)) if m.group(2) else 0.0  # avg fee part

        rows.append(
            {
                "ticker": ticker,
                "subsidy_per_block": subsidy,
                "avg_fee_per_block": fee_component,
                "total_reward_per_block": subsidy + fee_component,
                "source_url": url,
            }
        )

    if not rows:
        raise RuntimeError("No rewards could be scraped; check connectivity or HTML structure.")

    df = pd.DataFrame(rows).set_index("ticker")
    return df

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

def main():
    rewards_df = fetch_current_block_rewards()
    print("\n=== Block Reward Snapshot (subsidy + average fees) ===")
    print(rewards_df)

    out_path = "block_rewards_snapshot_bitinfocharts.csv"
    rewards_df.to_csv(out_path)
    print(f"\n[INFO] Saved snapshot to {out_path}")

if __name__ == "__main__":
    main()
