import re
import requests
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def scrape_bitcoin_asic_miners(
    url: str = "https://www.asicminervalue.com/en"
) -> pd.DataFrame:
    """
    Scrape ASIC Miner Value and return a DataFrame of *Bitcoin* miners with:

      - manufacturer
      - model
      - release_date
      - hashrate
      - power
      - algorithm
      - best_price_text      (original text, e.g. "$7,800 $6,724 / Ph")
      - best_price_usd       (numeric, e.g. 7800.0)
      - profit_text          (original text, e.g. "$19.70 /day")
      - profit_usd_per_day   (float, positive = profit, negative = loss)

    Filtering:
      1. Prefer rows whose “Top” column mentions Bitcoin (if available).
      2. If that fails (e.g. Top column has only icons), fall back to
         Algorithm containing 'SHA-256' (Bitcoin ASICs).
    """

    # --- Download page and parse the profitability table ---
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("No tables found on ASIC Miner Value page.")

    df = tables[0]
    df.columns = [str(c).strip() for c in df.columns]

    required = {
        "Manufacturer", "Model", "Release",
        "Hashrate", "Power", "Algorithm",
        "Best price", "Profit"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Unexpected table structure. Missing: {missing}")

    # --- Filter down to Bitcoin miners ---

    # Try to use the "Top" column if it exists and has text
    btc_mask = None
    if "Top" in df.columns:
        top_str = df["Top"].astype(str)
        btc_mask = top_str.str.contains("Bitcoin", case=False, na=False)

    if btc_mask is not None and btc_mask.any():
        btc_df = df.loc[btc_mask].copy()
    else:
        # Fallback: SHA-256 ASICs (Bitcoin algorithm)
        algo_str = df["Algorithm"].astype(str)
        btc_df = df.loc[algo_str.str.contains("SHA-256", case=False, na=False)].copy()

    # --- Cleaning helpers ---

    def extract_first_dollar(text):
        """Return first $ amount as float, or None."""
        if pd.isna(text):
            return None
        m = re.search(r"\$([0-9][\d,]*(?:\.\d+)?)", str(text))
        if not m:
            return None
        # Remove commas and convert to float
        return float(m.group(1).replace(",", ""))

    def extract_signed_number(text):
        """Return first signed number as float (for profit/loss), or None."""
        if pd.isna(text):
            return None
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
        return float(m.group(0)) if m else None

    # --- Build the output frame with the requested fields ---

    out = pd.DataFrame({
        "manufacturer": btc_df["Manufacturer"].str.strip(),
        "model":        btc_df["Model"].str.strip(),
        "release_date": btc_df["Release"].str.strip(),
        "hashrate":     btc_df["Hashrate"].str.strip(),
        "power":        btc_df["Power"].str.strip(),
        "algorithm":    btc_df["Algorithm"].str.strip(),
        "best_price_text": btc_df["Best price"].astype(str).str.strip(),
        "profit_text":     btc_df["Profit"].astype(str).str.strip(),
    })

    out["best_price_usd"] = out["best_price_text"].apply(extract_first_dollar)
    out["profit_usd_per_day"] = out["profit_text"].apply(extract_signed_number)

    return out


btc_miners = scrape_bitcoin_asic_miners()

    # Save to CSV if you need it:
btc_miners.to_csv("bitcoin_asic_miners.csv", index=False)
