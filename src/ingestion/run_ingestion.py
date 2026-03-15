"""
run_ingestion.py
----------------
Master ingestion pipeline — runs all 4 supplier parsers and saves
clean CSV files to data/processed/.

What this script does:
    1. Parses all Valero HTML files    -> data/processed/valero_clean.csv
    2. Parses all Exxon Excel files    -> data/processed/exxon_clean.csv
    3. Parses all Marathon TXT files   -> data/processed/marathon_clean.csv
    4. Parses all Pemex PDF files      -> data/processed/pemex_clean.csv
    5. Combines all 4 into one file    -> data/processed/all_suppliers.csv

Output schema for all_suppliers.csv (common columns across all suppliers):
    date            | YYYY-MM-DD
    supplier        | Valero / Exxon / Marathon / Pemex
    terminal_id     | supplier-specific terminal/region code
    terminal_name   | human-readable terminal name
    state           | state/region code where available
    country         | MX or US
    product_type    | Regular / Premium / Diesel
    price_mxn_per_l | final invoice price in MXN per Liter
    contract_type   | Sin Marca / Wholesale / Unbranded / TAR
    source_file     | original filename for traceability
    price_flag      | True if price is outside expected range

Usage:
    # Run everything
    python src/ingestion/run_ingestion.py

    # Skip specific suppliers (useful for testing)
    python src/ingestion/run_ingestion.py --skip pemex
    python src/ingestion/run_ingestion.py --skip valero exxon
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.parse_valero   import parse_valero_folder,   validate_prices as validate_valero
from src.ingestion.parse_exxon    import parse_exxon_folder,    validate_prices as validate_exxon
from src.ingestion.parse_marathon import parse_marathon_folder, validate_prices as validate_marathon
from src.ingestion.parse_pemex    import parse_pemex_folder,    validate_prices as validate_pemex

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -- Paths --------------------------------------------------------------------
DATA_RAW       = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

PATHS = {
    "valero":   DATA_RAW / "valero",
    "exxon":    DATA_RAW / "exxon",
    "marathon": DATA_RAW / "marathon",
    "pemex":    DATA_RAW / "pemex" / "reportes_pdf",
}

OUTPUT_FILES = {
    "valero":   DATA_PROCESSED / "valero_clean.csv",
    "exxon":    DATA_PROCESSED / "exxon_clean.csv",
    "marathon": DATA_PROCESSED / "marathon_clean.csv",
    "pemex":    DATA_PROCESSED / "pemex_clean.csv",
    "combined": DATA_PROCESSED / "all_suppliers.csv",
}

# -- Common columns for the combined file -------------------------------------
COMMON_COLUMNS = [
    "date",
    "supplier",
    "terminal_id",
    "terminal_name",
    "state",
    "country",
    "product_type",
    "price_mxn_per_l",
    "contract_type",
    "source_file",
    "price_flag",
]


# -- Individual supplier runners ----------------------------------------------

def run_valero(skip=False):
    if skip:
        logger.info("Skipping Valero")
        return pd.DataFrame()
    logger.info("=" * 60)
    logger.info("VALERO — Parsing HTML files")
    logger.info("=" * 60)
    if not PATHS["valero"].exists():
        logger.error("Valero folder not found: %s", PATHS["valero"])
        return pd.DataFrame()
    df = parse_valero_folder(str(PATHS["valero"]))
    df = validate_valero(df)
    if df.empty:
        return df
    df.to_csv(OUTPUT_FILES["valero"], index=False)
    logger.info("Saved -> %s  (%d rows)", OUTPUT_FILES["valero"], len(df))
    return df


def run_exxon(skip=False):
    if skip:
        logger.info("Skipping Exxon")
        return pd.DataFrame()
    logger.info("=" * 60)
    logger.info("EXXON — Parsing Excel files")
    logger.info("=" * 60)
    if not PATHS["exxon"].exists():
        logger.error("Exxon folder not found: %s", PATHS["exxon"])
        return pd.DataFrame()
    df = parse_exxon_folder(str(PATHS["exxon"]))
    df = validate_exxon(df)
    if df.empty:
        return df
    df.to_csv(OUTPUT_FILES["exxon"], index=False)
    logger.info("Saved -> %s  (%d rows)", OUTPUT_FILES["exxon"], len(df))
    return df


def run_marathon(skip=False):
    if skip:
        logger.info("Skipping Marathon")
        return pd.DataFrame()
    logger.info("=" * 60)
    logger.info("MARATHON — Parsing TXT email files")
    logger.info("=" * 60)
    if not PATHS["marathon"].exists():
        logger.error("Marathon folder not found: %s", PATHS["marathon"])
        return pd.DataFrame()
    df = parse_marathon_folder(str(PATHS["marathon"]))
    df = validate_marathon(df)
    if df.empty:
        return df
    df.to_csv(OUTPUT_FILES["marathon"], index=False)
    logger.info("Saved -> %s  (%d rows)", OUTPUT_FILES["marathon"], len(df))
    return df


def run_pemex(skip=False):
    if skip:
        logger.info("Skipping Pemex")
        return pd.DataFrame()
    logger.info("=" * 60)
    logger.info("PEMEX — Parsing PDF files")
    logger.info("=" * 60)
    if not PATHS["pemex"].exists():
        logger.error("Pemex folder not found: %s", PATHS["pemex"])
        return pd.DataFrame()
    df = parse_pemex_folder(str(PATHS["pemex"]))
    df = validate_pemex(df)
    if df.empty:
        return df
    df.to_csv(OUTPUT_FILES["pemex"], index=False)
    logger.info("Saved -> %s  (%d rows)", OUTPUT_FILES["pemex"], len(df))
    return df


# -- Combine all suppliers ----------------------------------------------------

def combine_suppliers(frames):
    logger.info("=" * 60)
    logger.info("COMBINING all suppliers")
    logger.info("=" * 60)

    valid_frames = []
    for supplier, df in frames.items():
        if df.empty:
            logger.warning("No data for %s — skipping from combined file", supplier)
            continue
        available = [c for c in COMMON_COLUMNS if c in df.columns]
        missing   = [c for c in COMMON_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("%s is missing columns: %s", supplier, missing)
        valid_frames.append(df[available])

    if not valid_frames:
        logger.error("No data to combine")
        return pd.DataFrame()

    combined = pd.concat(valid_frames, ignore_index=True)
    combined = combined.sort_values(["date", "supplier", "terminal_id"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_FILES["combined"], index=False)
    logger.info("Saved -> %s  (%d rows)", OUTPUT_FILES["combined"], len(combined))
    return combined


# -- Summary report -----------------------------------------------------------

def print_summary(frames, combined):
    print()
    print("=" * 65)
    print("  INGESTION SUMMARY")
    print("=" * 65)
    print(f"  {'Supplier':<12} {'Records':>8}  {'Terminals':>10}  {'Flagged':>8}")
    print("-" * 65)

    total_records = 0
    total_flagged = 0

    for supplier, df in frames.items():
        if df.empty:
            print(f"  {supplier.capitalize():<12} {'SKIPPED':>8}")
            continue
        records   = len(df)
        terminals = df["terminal_id"].nunique() if "terminal_id" in df.columns else 0
        flagged   = int(df["price_flag"].sum()) if "price_flag" in df.columns else 0
        total_records += records
        total_flagged += flagged
        print(f"  {supplier.capitalize():<12} {records:>8,}  {terminals:>10,}  {flagged:>8,}")

    print("-" * 65)
    print(f"  {'TOTAL':<12} {total_records:>8,}  {'':>10}  {total_flagged:>8,}")
    print("=" * 65)

    if not combined.empty:
        print()
        print(f"  Date range  : {combined['date'].min().date()} to {combined['date'].max().date()}")
        print(f"  Products    : {sorted(combined['product_type'].unique())}")
        print(f"  Output file : {OUTPUT_FILES['combined']}")

    print("=" * 65)
    print()


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full AI Procurement Hub data ingestion pipeline"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["valero", "exxon", "marathon", "pemex"],
        default=[],
        help="Suppliers to skip (e.g. --skip pemex valero)",
    )
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ingestion pipeline")
    if args.skip:
        logger.info("Skipping: %s", args.skip)

    frames = {
        "valero":   run_valero(skip="valero" in args.skip),
        "exxon":    run_exxon(skip="exxon" in args.skip),
        "marathon": run_marathon(skip="marathon" in args.skip),
        "pemex":    run_pemex(skip="pemex" in args.skip),
    }

    combined = combine_suppliers(frames)
    print_summary(frames, combined)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()