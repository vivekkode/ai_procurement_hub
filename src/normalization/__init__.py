"""
normalize_prices.py
-------------------
Post-ingestion normalization layer.

Reads:
    data/processed/all_suppliers.csv

Outputs:
    data/processed/normalized_prices.csv

Purpose:
    - Standardize product names
    - Normalize supplier names
    - Ensure price consistency (MXN per liter)
    - Clean invalid / missing data
    - Prepare dataset for optimization engine
"""

import pandas as pd
from pathlib import Path


# ==============================
# CONFIG
# ==============================

INPUT_FILE = Path("data/processed/all_suppliers.csv")
OUTPUT_FILE = Path("data/processed/normalized_prices.csv")

VALID_PRODUCTS = ["regular", "premium", "diesel"]


# ==============================
# NORMALIZATION FUNCTIONS
# ==============================

def normalize_product(product: str) -> str:
    """Convert product names to standard: regular, premium, diesel"""
    if pd.isna(product):
        return None

    p = str(product).lower()

    if "regular" in p or "magna" in p or "87" in p:
        return "regular"
    elif "premium" in p or "91" in p:
        return "premium"
    elif "diesel" in p:
        return "diesel"

    return None


def normalize_supplier(supplier: str) -> str:
    """Standardize supplier names"""
    if pd.isna(supplier):
        return None

    s = str(supplier).lower()

    if "pemex" in s:
        return "PEMEX"
    elif "valero" in s:
        return "Valero"
    elif "exxon" in s:
        return "ExxonMobil"
    elif "marathon" in s:
        return "Marathon"

    return s.title()


def clean_price(price):
    """Ensure price is numeric and valid"""
    try:
        price = float(price)
        if price <= 0:
            return None
        return price
    except:
        return None


# ==============================
# MAIN PIPELINE
# ==============================

def normalize_prices():

    print("📥 Loading data...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Initial rows: {len(df)}")

    # ----------------------------------------
    # DATE CLEANING
    # ----------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ----------------------------------------
    # PRODUCT NORMALIZATION
    # ----------------------------------------
    df["product"] = df["product_type"].apply(normalize_product)

    # ----------------------------------------
    # SUPPLIER NORMALIZATION
    # ----------------------------------------
    df["supplier"] = df["supplier"].apply(normalize_supplier)

    # ----------------------------------------
    # PRICE CLEANING
    # ----------------------------------------
    df["price_mxn_litre"] = df["price_mxn_per_l"].apply(clean_price)

    # ----------------------------------------
    # REMOVE INVALID ROWS
    # ----------------------------------------
    df = df.dropna(subset=[
        "date",
        "product",
        "price_mxn_litre"
    ])

    # ----------------------------------------
    # FILTER VALID PRODUCTS
    # ----------------------------------------
    df = df[df["product"].isin(VALID_PRODUCTS)]

    # ----------------------------------------
    # REMOVE OUTLIERS (optional but useful)
    # ----------------------------------------
    df = df[
        (df["price_mxn_litre"] >= 10) &
        (df["price_mxn_litre"] <= 40)
    ]

    # ----------------------------------------
    # FINAL SCHEMA
    # ----------------------------------------
    final_df = df[[
        "date",
        "supplier",
        "terminal_id",
        "terminal_name",
        "state",
        "country",
        "product",
        "price_mxn_litre",
        "contract_type",
        "source_file"
    ]].copy()

    # ----------------------------------------
    # REMOVE DUPLICATES
    # ----------------------------------------
    final_df = final_df.drop_duplicates()

    # ----------------------------------------
    # SORT
    # ----------------------------------------
    final_df = final_df.sort_values(
        ["date", "supplier", "terminal_id"]
    ).reset_index(drop=True)

    print(f"✅ Cleaned rows: {len(final_df)}")

    # ----------------------------------------
    # SAVE OUTPUT
    # ----------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"💾 Saved to: {OUTPUT_FILE}")

    return final_df


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    normalize_prices()
