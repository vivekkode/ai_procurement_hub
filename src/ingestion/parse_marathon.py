"""
parse_marathon.py
-----------------
Parses Marathon daily wholesale pricing TXT (email) files.

File format:
    marathon/marathon_YYYYMMDD_REGION-IT-City.txt
    - One file per terminal per trading day
    - Email-style format with header metadata
    - Contains full tax breakdown per product

    Filename encodes both date AND terminal:
        marathon_20240101_MX-IT-Chihuahua.txt
        marathon_20240101_US-IT-El_Paso.txt

    Price rows in the file:
        Base Price      - price before taxes
        Net Price       - same as base (no adjustments)
        Tax MX IEPS 2D  - federal fuel excise tax
        Tax MX IEPS 2H  - environmental complement tax
        Tax MX IEPS 2A  - additional excise tax
        Unit Price      - subtotal before VAT
        IVA             - 16% VAT
        Invoice Price   - final price paid  <- this is what we use

    Products (3 columns in fixed-width format):
        UNBRANDED REGULAR | UNBRANDED PREMIUM | UNBRANDED DIESEL

Output schema (matches parse_valero.py and parse_exxon.py):
    date            | YYYY-MM-DD
    supplier        | "Marathon"
    terminal_id     | e.g. "MX-IT-Chihuahua"
    terminal_name   | e.g. "Chihuahua"
    state           | e.g. "MX-IT" (region prefix)
    country         | "MX" or "US"
    product_raw     | original product string from file
    product_type    | normalized: "Regular", "Premium", "Diesel"
    price_mxn_per_l | float - invoice price MXN/Liter
    base_price      | float - price before all taxes
    unit_price      | float - price before IVA
    iva             | float - VAT amount
    ieps_2d         | float - federal fuel excise tax
    ieps_2h         | float - environmental complement tax
    ieps_2a         | float - additional excise tax
    contract_type   | always "Unbranded"
    source_file     | filename for traceability

Marathon advantage over other suppliers:
    Full tax breakdown available - critical for the landed cost engine
    to understand the true cost structure and model tax policy changes.

Usage:
    # Parse a single file
    from src.ingestion.parse_marathon import parse_marathon_file
    df = parse_marathon_file("data/raw/marathon/marathon_20240101_MX-IT-Chihuahua.txt")

    # Parse all files in a folder
    from src.ingestion.parse_marathon import parse_marathon_folder
    df = parse_marathon_folder("data/raw/marathon/")
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# -- Price guardrails ---------------------------------------------------------
PRICE_MIN = 15.0   # MXN/L
PRICE_MAX = 35.0   # MXN/L

# -- Product columns ----------------------------------------------------------
# Marathon files always have exactly 3 product columns in this order
PRODUCTS = ["Regular", "Premium", "Diesel"]


# -- Filename parser ----------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """
    Extract date, terminal_id, terminal_name, and country from filename.

    Filename patterns:
        marathon_20240101_MX-IT-Chihuahua.txt
        marathon_20240101_US-IT-El_Paso.txt
        marathon_20240101_MX-IT-Ciudad_Juarez.txt

    Returns dict with: date, terminal_id, terminal_name, state, country
    """
    # Remove .txt extension and split on underscores
    # Pattern: marathon_YYYYMMDD_COUNTRY-IT-City
    match = re.match(
        r"marathon_(\d{8})_((MX|US)-IT-(.+))\.txt$",
        filename,
        re.IGNORECASE,
    )

    if not match:
        logger.warning("Unexpected filename format: %s", filename)
        return {
            "date": "",
            "terminal_id": "",
            "terminal_name": filename,
            "state": "",
            "country": "MX",
        }

    # Extract date
    date_raw = match.group(1)
    try:
        date_str = datetime.strptime(date_raw, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        date_str = ""
        logger.warning("Could not parse date from: %s", filename)

    # Extract terminal info
    terminal_id   = match.group(2)           # e.g. "MX-IT-Chihuahua"
    country       = match.group(3).upper()   # e.g. "MX" or "US"
    city_raw      = match.group(4)           # e.g. "Chihuahua" or "El_Paso"

    # Replace underscores with spaces for city name
    terminal_name = city_raw.replace("_", " ")

    # State is the country-region prefix
    state = f"{country}-IT"

    return {
        "date":          date_str,
        "terminal_id":   terminal_id,
        "terminal_name": terminal_name,
        "state":         state,
        "country":       country,
    }


# -- Price row parser ---------------------------------------------------------

def parse_price_row(line: str) -> list:
    """
    Extract 3 float values from a fixed-width price row.

    Marathon price rows look like:
        "Base Price                           12.1078               11.5512               12.2262"
        "Invoice Price                        20.5333               20.4096               22.1999"

    The row label is at the start, followed by 3 space-separated numbers.

    Returns list of 3 floats [regular, premium, diesel]
    or [None, None, None] if parsing fails.
    """
    # Find all numbers (including decimals) in the line
    numbers = re.findall(r"\d+\.\d+", line)

    if len(numbers) >= 3:
        try:
            return [float(numbers[0]), float(numbers[1]), float(numbers[2])]
        except ValueError:
            pass

    return [None, None, None]


# -- Core parser --------------------------------------------------------------

def parse_marathon_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single Marathon TXT email pricing file into a tidy DataFrame.

    Each row represents one product at one terminal on one date.
    Produces 3 rows per file (Regular, Premium, Diesel).

    Args:
        filepath: Path to the .txt file.

    Returns:
        pd.DataFrame with columns matching output schema defined at top.
        Returns empty DataFrame on failure.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return pd.DataFrame()

    # Extract date and terminal info from filename
    file_info = parse_filename(filepath.name)

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error("Could not read %s: %s", filepath.name, e)
        return pd.DataFrame()

    # -- Parse price rows from file content -----------------------------------
    # We look for specific row labels and extract their 3 values

    price_data = {
        "base_price": [None, None, None],
        "unit_price": [None, None, None],
        "iva":        [None, None, None],
        "invoice":    [None, None, None],
        "ieps_2d":    [None, None, None],
        "ieps_2h":    [None, None, None],
        "ieps_2a":    [None, None, None],
    }

    for line in content.splitlines():
        line_stripped = line.strip()

        if line_stripped.startswith("Base Price"):
            price_data["base_price"] = parse_price_row(line)

        elif line_stripped.startswith("Unit Price"):
            price_data["unit_price"] = parse_price_row(line)

        elif line_stripped.startswith("IVA"):
            price_data["iva"] = parse_price_row(line)

        elif line_stripped.startswith("Invoice Price"):
            price_data["invoice"] = parse_price_row(line)

        elif "IEPS 2D" in line_stripped:
            price_data["ieps_2d"] = parse_price_row(line)

        elif "IEPS 2H" in line_stripped:
            price_data["ieps_2h"] = parse_price_row(line)

        elif "IEPS 2A" in line_stripped:
            price_data["ieps_2a"] = parse_price_row(line)

    # -- Validate we got invoice prices ---------------------------------------
    if all(v is None for v in price_data["invoice"]):
        logger.warning("No invoice prices found in: %s", filepath.name)
        return pd.DataFrame()

    # -- Build one row per product --------------------------------------------
    rows = []
    product_map = {
        "Regular": "UNBRANDED REGULAR",
        "Premium": "UNBRANDED PREMIUM",
        "Diesel":  "UNBRANDED DIESEL",
    }

    for i, product in enumerate(PRODUCTS):
        invoice_price = price_data["invoice"][i]

        if invoice_price is None:
            logger.debug("No invoice price for %s in %s", product, filepath.name)
            continue

        rows.append({
            "date":            file_info["date"],
            "supplier":        "Marathon",
            "terminal_id":     file_info["terminal_id"],
            "terminal_name":   file_info["terminal_name"],
            "state":           file_info["state"],
            "country":         file_info["country"],
            "product_raw":     product_map[product],
            "product_type":    product,
            "price_mxn_per_l": invoice_price,
            "base_price":      price_data["base_price"][i],
            "unit_price":      price_data["unit_price"][i],
            "iva":             price_data["iva"][i],
            "ieps_2d":         price_data["ieps_2d"][i],
            "ieps_2h":         price_data["ieps_2h"][i],
            "ieps_2a":         price_data["ieps_2a"][i],
            "contract_type":   "Unbranded",
            "source_file":     filepath.name,
        })

    if not rows:
        logger.warning("No rows built from: %s", filepath.name)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["price_mxn_per_l"] = df["price_mxn_per_l"].astype(float)

    return df


# -- Folder parser ------------------------------------------------------------

def parse_marathon_folder(folder_path: str) -> pd.DataFrame:
    """
    Parse all Marathon TXT files in a folder and combine into one DataFrame.

    Marathon has one file per terminal per day, so expect many more files
    than Valero or Exxon. A full year = ~250 days x 10 terminals = ~2500 files.

    Args:
        folder_path: Path to folder containing marathon_*.txt files.

    Returns:
        Combined pd.DataFrame sorted by date ascending.
    """
    folder = Path(folder_path)
    txt_files = sorted(folder.glob("marathon_*.txt"))

    if not txt_files:
        logger.warning("No Marathon TXT files found in: %s", folder_path)
        return pd.DataFrame()

    logger.info("Found %d Marathon files to parse in: %s", len(txt_files), folder_path)

    frames = []
    failed = 0

    for f in txt_files:
        df = parse_marathon_file(f)
        if df.empty:
            failed += 1
        else:
            frames.append(df)

    if not frames:
        logger.error("All files failed to parse in: %s", folder_path)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Drop exact duplicates
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "terminal_id", "product_type"]
    )
    dupes_dropped = before - len(combined)
    if dupes_dropped:
        logger.info("Dropped %d duplicate rows", dupes_dropped)

    combined = combined.sort_values("date").reset_index(drop=True)

    logger.info(
        "Total: %d records | %d terminals | %d unique dates | %d files failed",
        len(combined),
        combined["terminal_id"].nunique(),
        combined["date"].nunique(),
        failed,
    )
    return combined


# -- Guardrails ---------------------------------------------------------------

def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag anomalous invoice prices outside expected MXN/L range.

    Also checks for cases where invoice price is lower than base price
    which would indicate a parsing error (taxes can't be negative overall).

    Args:
        df: Output of parse_marathon_file() or parse_marathon_folder().

    Returns:
        Same DataFrame with added 'price_flag' column.
    """
    if df.empty:
        return df

    df = df.copy()

    # Basic range check
    range_flag = (
        (df["price_mxn_per_l"] < PRICE_MIN) |
        (df["price_mxn_per_l"] > PRICE_MAX)
    )

    # Sanity check: invoice price must be higher than base price
    # (taxes add to the price, so invoice > base always)
    logic_flag = pd.Series(False, index=df.index)
    if "base_price" in df.columns:
        logic_flag = df["price_mxn_per_l"] < df["base_price"]

    df["price_flag"] = range_flag | logic_flag

    flagged = df["price_flag"].sum()
    if flagged:
        logger.warning(
            "%d price(s) flagged — outside range or invoice < base price",
            flagged,
        )

    return df


# -- CLI convenience ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_marathon.py <path_to_txt_or_folder>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        result = parse_marathon_folder(target)
    else:
        result = parse_marathon_file(target)

    result = validate_prices(result)

    if not result.empty:
        print(result.to_string(index=False))
        print(f"\nShape: {result.shape}")
        flagged = result["price_flag"].sum() if "price_flag" in result.columns else 0
        print(f"Flagged prices: {flagged}")