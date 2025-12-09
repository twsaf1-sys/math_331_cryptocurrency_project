import re
import pandas as pd

URL = "https://en.bitcoin.it/wiki/Mining_hardware_comparison"

def clean_number(value):
    """
    Take things like '1,000,000', '350[2]', '14,000,000', '52.34', '—'
    and convert to a numeric value, or NaN if it can't be parsed.
    """
    if pd.isna(value):
        return pd.NA

    s = str(value)
    # Remove Wikipedia footnote markers like [2]
    s = re.sub(r"\[[^\]]*\]", "", s)
    # Remove all characters except digits, decimal point, and minus sign
    s = re.sub(r"[^0-9.\-]", "", s)

    if s == "":
        return pd.NA

    return pd.to_numeric(s, errors="coerce")


def main():
    # 1. Read all HTML tables from the page
    tables = pd.read_html(URL)

    # 2. Pick the main summary table (the one with a 'Product' column)
    summary_tables = [t for t in tables if "Product" in t.columns]
    if not summary_tables:
        raise RuntimeError("Could not find a table with a 'Product' column.")
    df = max(summary_tables, key=lambda t: t.shape[0])  # largest one

    # 3. (Optional) rename columns for nicer CSV headers
    df = df.rename(
        columns={
            "Product": "product",
            "Advertised Mhash/s": "advertised_mhash_per_s",
            "Mhash/J": "mhash_per_joule",
            "Mhash/s/$": "mhash_per_s_per_usd",
            "Watts": "watts",
            "Price (USD)": "price_usd",
            "Currently shipping": "currently_shipping",
            "Comm ports": "comm_ports",
            "Dev-friendly": "dev_friendly",
        }
    )

    # 4. Clean numeric columns
    numeric_cols = [
        "advertised_mhash_per_s",
        "mhash_per_joule",
        "mhash_per_s_per_usd",
        "watts",
        "price_usd",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_number)

    # 5. Save to CSV
    output_path = "bitcoin_mining_hardware_summary_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned table to {output_path}")


if __name__ == "__main__":
    main()
