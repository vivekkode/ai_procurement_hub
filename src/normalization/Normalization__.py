"""
normalize_prices.py
-------------------
Post-ingestion normalization layer.

Reads:
    data/processed/all_suppliers.csv

Outputs:
    data/processed/normalized_prices.csv
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
    try:
        price = float(price)
        if price <= 0:
            return None
        return price
    except:
        return None


# ==============================
# TERMINAL + STATE FIXES
# ==============================

def fix_terminal_id(df):
    # Step 1: fill with terminal_name
    df["terminal_id"] = df["terminal_id"].fillna(df["terminal_name"])

    # Step 2: generate IDs ONLY for remaining missing
    missing_mask = df["terminal_id"].isna()

    df.loc[missing_mask, "terminal_id"] = (
        "T_" + df.loc[missing_mask].index.astype(str)
    )

    return df

def extract_state_from_name(name):
    if pd.isna(name):
        return None

    name = str(name).lower()

    states_map = {
        "veracruz": "Veracruz",
        "tamaulipas": "Tamaulipas",
        "nuevo leon": "Nuevo Leon",
        "monterrey": "Nuevo Leon",
        "guadalajara": "Jalisco",
        "jalisco": "Jalisco",
        "cdmx": "CDMX",
        "mexico": "Estado de Mexico",
        "sonora": "Sonora",
        "sinaloa": "Sinaloa",
        "coahuila": "Coahuila",
        "chihuahua": "Chihuahua",
        "puebla": "Puebla",
        "yucatan": "Yucatan",
    }

    for key, value in states_map.items():
        if key in name:
            return value

    return None


def fix_state(df):
    df["state"] = df.apply(
        lambda row: extract_state_from_name(row["terminal_name"])
        if pd.isna(row["state"]) else row["state"],
        axis=1
    )

    df["state"] = df["state"].fillna("UNKNOWN")

    return df


# ==============================
# MAIN PIPELINE
# ==============================

def normalize_prices():

    print("📥 Loading data...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Initial rows: {len(df)}")

    # DATE
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # PRODUCT
    df["product"] = df["product_type"].apply(normalize_product)

    # SUPPLIER
    df["supplier"] = df["supplier"].apply(normalize_supplier)

    # PRICE
    df["price_mxn_litre"] = df["price_mxn_per_l"].apply(clean_price)

    # REMOVE INVALID
    df = df.dropna(subset=["date", "product", "price_mxn_litre"])

    # VALID PRODUCTS
    df = df[df["product"].isin(VALID_PRODUCTS)]

    # REMOVE OUTLIERS
    df = df[
        (df["price_mxn_litre"] >= 10) &
        (df["price_mxn_litre"] <= 40)
    ]

    # FIX TERMINAL + STATE
    df = fix_terminal_id(df)
    df = fix_state(df)

    # FINAL SCHEMA
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

    # REMOVE DUPLICATES
    final_df = final_df.drop_duplicates()

    # SORT
    final_df = final_df.sort_values(
        ["date", "supplier", "terminal_id"]
    ).reset_index(drop=True)

    print(f"✅ Cleaned rows: {len(final_df)}")

    # SAVE
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"💾 Saved to: {OUTPUT_FILE}")

    return final_df


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    normalize_prices()
