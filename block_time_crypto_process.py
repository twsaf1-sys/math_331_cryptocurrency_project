import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from datetime import datetime
from urllib.parse import quote

import matplotlib.pyplot as plt

# ------------- Config -------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# Friendly name -> slug used in BitInfoCharts comparison URL
COINS = {
    "Bitcoin": "bitcoin",
    "Litecoin": "litecoin",
    "Monero": "monero",
    "Dash": "dash",
    "ZCash": "zcash",
    "Bitcoin Cash": "bitcoin%20cash",
    "Ethereum Classic": "etc",
    "Bitcoin Gold": "bitcoin%20gold",
    "Dogecoin": "dogecoin",
    "Vertcoin": "vertcoin",
}

START_DATE = "2015-01-01"   # inclusive
# "Present day" will be resolved at runtime with pandas date_range


# ------------- Core scraping helper -------------

def extract_hashrate_series(coin_label: str) -> pd.Series:
    """
    Fetch hashrate history for a single coin from BitInfoCharts
    and return as a pandas Series indexed by datetime.date.

    coin_label: human-readable key from COINS dict (e.g. "Bitcoin").
    """

    base = COINS[coin_label]
    slug = quote(base)
    url = f"https://bitinfocharts.com/comparison/{slug}-confirmationtime.html"

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the <script> tag that contains the Dygraph data array.
    target_script = None
    for s in soup.find_all("script"):
        txt = s.string or s.text or ""
        # Heuristic: Dygraph init + data array
        if "new Dygraph" in txt and "[[" in txt:
            target_script = txt
            break

    if not target_script:
        raise RuntimeError(f"Could not locate data script for {coin_label} at {url}")

    # Grab the first [[ ... ]] block after Dygraph init.
    start = target_script.find("[[")
    end = target_script.find("]]", start)
    if start == -1 or end == -1:
        raise RuntimeError(f"Could not find data array brackets for {coin_label}")

    data_str = target_script[start:end+2]

    # --- Normalize date formats into JSON-friendly strings ---

    # Case 1: Date.UTC(YYYY,MM,DD)
    # BitInfoCharts typically uses 0-based month in Date.UTC; add 1.
    def repl_utc(m):
        y, mth, d = map(int, m.groups())
        mth += 1
        return f'"{y}-{mth:02d}-{d:02d}"'

    data_str = re.sub(r'Date\.UTC\((\d+),(\d+),(\d+)\)', repl_utc, data_str)

    # Case 2: new Date("YYYY/MM/DD")
    data_str = re.sub(
        r'new Date\("(\d{4})/(\d{2})/(\d{2})"\)',
        r'"\1-\2-\3"',
        data_str
    )

    # Ensure JSON compatibility
    data_str = data_str.replace("'", '"')
    # Remove trailing commas before ] or }
    data_str = re.sub(r',\s*]', ']', data_str)
    data_str = re.sub(r',\s*}', '}', data_str)

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse failed for {coin_label}: {e}")

    # --- Build Series[date] = hashrate ---

    records = []
    for row in data:
        if len(row) < 2:
            continue

        date_val, hr_val = row[0], row[1]

        if hr_val in (None, "null"):
            continue

        # date_val is now expected as "YYYY-MM-DD" (string)
        if isinstance(date_val, str):
            dt = datetime.strptime(date_val, "%Y-%m-%d").date()
        else:
            # Fallback: if it’s still numeric timestamp
            ts = float(date_val)
            # Heuristic for ms vs s:
            if ts > 10**11:
                dt = datetime.utcfromtimestamp(ts / 1000).date()
            else:
                dt = datetime.utcfromtimestamp(ts).date()

        try:
            v = float(hr_val)
        except (TypeError, ValueError):
            continue

        records.append((dt, v))

    if not records:
        raise RuntimeError(f"No confirmation time per block records parsed for {coin_label}")

    # Use the last value per date if duplicates, then sort.
    tmp = {}
    for d, v in records:
        tmp[d] = v
    series = pd.Series(tmp).sort_index()
    series.name = coin_label

    return series


# ------------- Build unified DataFrame -------------

def build_hashrate_dataframe():
    # Date index from 2015-01-01 to today (UTC-based "today" ok for this use)
    end_date = pd.Timestamp.today().normalize()
    date_index = pd.date_range(start=START_DATE, end=end_date, freq="D")

    df = pd.DataFrame(index=date_index)

    for coin_label in COINS.keys():
        print(f"Fetching {coin_label} hashrate...")
        s = extract_hashrate_series(coin_label)
        # Align series to full date index
        s = s.reindex(date_index)
        df[coin_label] = s

    # Make 'Date' a column (as requested) instead of index
    df = df.reset_index().rename(columns={"index": "Date"})

    # Replace NaN/empty values with 0 in all columns
    df = df.fillna(0)

    return df


# ------------- Main: save CSV + plots -------------

def main():
    df = build_hashrate_dataframe()

    # Save CSV
    csv_filename = "confirmation_time_per_block_2015_to_present_bitinfocharts.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")

    # Plot each cryptocurrency individually
    # (one PNG per coin, full time range)
    plt.ioff()
    date_col = "Date"

    for coin in COINS.keys():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[date_col], df[coin])
        ax.set_title(f"{coin} Block Time (2015 - Present)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Block Time (Minutes)")
        fig.autofmt_xdate()
        fig.tight_layout()

        outname = f"{coin.lower().replace(' ', '_').replace('-', '')}_block_time.png"
        fig.savefig(outname, dpi=300)
        plt.close(fig)
        print(f"Saved plot for {coin} to {outname}")


    # Composite plot: all coins on one figure
    fig, ax = plt.subplots(figsize=(12, 6))
    for coin in COINS.keys():
        ax.plot(df[date_col], df[coin], label=coin)

    ax.set_title("Block Time (2015 - Present)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Block Time (Minutes)")
    ax.legend(loc="upper left", ncol=2, fontsize="small")
    fig.autofmt_xdate()
    fig.tight_layout()

    composite_name = "block_time_composite.png"
    fig.savefig(composite_name, dpi=400)
    plt.close(fig)
    print(f"Saved composite hashrate plot to {composite_name}")


if __name__ == "__main__":
    main()
