"""
parse_exxon.py
--------------
Parses Exxon daily wholesale pricing Excel (.xlsx) files.

File format:
    exxon/exxon_YYYYMMDD.xlsx
    - One file per trading day
    - Single sheet named "Precios Exxon"
    - Row 1: title with date  e.g. "Exxon Mobil — Precios por Terminal — 01/01/2024"
    - Row 2: empty
    - Row 3: column headers
    - Rows 4+: data rows (3 terminals x 3 products = 9 data rows per file)
    - Last row: disclaimer text, ignored

    Columns:
        Terminal                          | e.g. "0TJB - MTY MOBIL"
        Categoria Cuenta                  | always "Wholesale"
        Producto                          | "Regular", "Premium", "Diesel"
        Precio Referencia Industrial MXN/L| reference price before discount
        Descuento MXN/L                   | discount applied (negative value)
        Precio Facturacion con Impuestos  | final invoice price MXN/L ← this is what we use

Output schema (matches parse_valero.py exactly):
    date            | YYYY-MM-DD
    supplier        | "Exxon"
    terminal_id     | e.g. "0TJB"
    terminal_name   | e.g. "MTY MOBIL"
    state           | e.g. "MTY"
    country         | "MX"
    product_raw     | original product string from file
    product_type    | normalized: "Regular", "Premium", "Diesel"
    price_mxn_per_l | float - final invoice price MXN/Liter
    ref_price       | float - reference price before discount
    discount        | float - discount applied
    contract_type   | always "Wholesale"
    source_file     | filename for traceability

Key difference from Valero:
    Exxon provides a discount column, giving us more pricing detail.
    We use the final invoice price (after discount) as price_mxn_per_l
    to stay consistent with how Valero prices are reported.

Usage:
    # Parse a single file
    from src.ingestion.parse_exxon import parse_exxon_file
    df = parse_exxon_file("data/raw/exxon/exxon_20240101.xlsx")

    # Parse all files in a folder
    from src.ingestion.parse_exxon import parse_exxon_folder
    df = parse_exxon_folder("data/raw/exxon/")
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import openpyxl

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# -- Price guardrails ---------------------------------------------------------
PRICE_MIN = 15.0   # MXN/L
PRICE_MAX = 35.0   # MXN/L


# -- Terminal parsing ---------------------------------------------------------

def parse_terminal_string(raw: str) -> dict:
    """
    Extract structured fields from Exxon terminal string.

    Exxon terminal format observed in data:
        "0TJB - MTY MOBIL"
        "0TD3 - SLP - San Luis Potosi"
        "0TK4 - Monterra - EMD"

    Pattern is always: "TERMINAL_ID - DESCRIPTION"

    Returns dict with: terminal_id, terminal_name, state, country
    """
    if not raw or not isinstance(raw, str):
        return {
            "terminal_id": "",
            "terminal_name": raw or "",
            "state": "",
            "country": "MX",
        }

    raw = raw.strip()

    # Split on first " - " to separate ID from name
    parts = raw.split(" - ", 1)

    if len(parts) == 2:
        terminal_id = parts[0].strip()
        terminal_name = parts[1].strip()

        # Try to extract state from terminal name
        # e.g. "MTY MOBIL" -> state is "MTY"
        # e.g. "SLP - San Luis Potosi" -> state is "SLP"
        # e.g. "Monterra - EMD" -> use terminal_id prefix
        state_match = re.match(r"^([A-Z]{2,4})\b", terminal_name)
        state = state_match.group(1) if state_match else ""

        return {
            "terminal_id": terminal_id,
            "terminal_name": terminal_name,
            "state": state,
            "country": "MX",
        }

    # Fallback
    return {
        "terminal_id": "",
        "terminal_name": raw,
        "state": "",
        "country": "MX",
    }


# -- Date extraction ----------------------------------------------------------

def extract_date_from_title(title: str) -> str:
    """
    Extract date from Exxon title row string.

    Title format: "Exxon Mobil  -  Precios por Terminal  -  01/01/2024"
    Note: Exxon uses DD/MM/YYYY format (opposite of Valero's MM/DD/YYYY)

    Returns "YYYY-MM-DD" string or empty string if not parseable.
    """
    if not title:
        return ""

    # Look for DD/MM/YYYY pattern
    match = re.search(r"(\d{2}/\d{2}/\d{4})", str(title))
    if match:
        raw = match.group(1)
        try:
            # Exxon uses DD/MM/YYYY
            return datetime.strptime(raw, "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass

    return ""


def extract_date_from_filename(filename: str) -> str:
    """
    Fallback: extract date from filename pattern exxon_YYYYMMDD.xlsx

    Returns "YYYY-MM-DD" string or empty string if not parseable.
    """
    match = re.search(r"exxon_(\d{8})", filename)
    if match:
        raw = match.group(1)
        try:
            return datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            logger.warning("Could not parse date from filename: %s", filename)
    return ""


def safe_float(value) -> float:
    """Convert a value to float, returning None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# -- Core parser --------------------------------------------------------------

def parse_exxon_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single Exxon Excel pricing file into a tidy DataFrame.

    Each row represents one product at one terminal on one date.

    Args:
        filepath: Path to the .xlsx file.

    Returns:
        pd.DataFrame with columns matching the output schema defined
        at the top of this module. Returns empty DataFrame on failure.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return pd.DataFrame()

    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    except Exception as e:
        logger.error("Could not open %s: %s", filepath.name, e)
        return pd.DataFrame()

    # Exxon always uses one sheet
    ws = wb.active
    all_rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not all_rows:
        logger.warning("Empty file: %s", filepath.name)
        return pd.DataFrame()

    # -- Extract date ---------------------------------------------------------
    # Row 1 (index 0) is the title row containing the date
    title_row = all_rows[0][0] if all_rows[0] else ""
    date_str = extract_date_from_title(str(title_row))

    # Fallback to filename if title date extraction failed
    if not date_str:
        date_str = extract_date_from_filename(filepath.name)
        if date_str:
            logger.debug("Used filename date for %s", filepath.name)
        else:
            logger.warning("No date found for %s", filepath.name)

    # -- Extract data rows ----------------------------------------------------
    # Row 1 = title, Row 2 = empty, Row 3 = headers, Rows 4+ = data
    # Skip first 3 rows and start reading data from row index 3
    rows = []

    for row in all_rows[3:]:  # start after header row
        # Skip empty rows and disclaimer row
        if not row[0] or not isinstance(row[0], str):
            continue

        # Skip the disclaimer row at the bottom
        if "sintéticos" in str(row[0]) or "Datos" in str(row[0]):
            continue

        terminal_raw  = row[0]
        contract_type = row[1] if row[1] else "Wholesale"
        product_raw   = row[2]
        ref_price     = safe_float(row[3])
        discount      = safe_float(row[4])
        invoice_price = safe_float(row[5])

        # Must have a terminal and product and a valid price
        if not terminal_raw or not product_raw or invoice_price is None:
            continue

        terminal_info = parse_terminal_string(terminal_raw)

        rows.append({
            "date":            date_str,
            "supplier":        "Exxon",
            "terminal_id":     terminal_info["terminal_id"],
            "terminal_name":   terminal_info["terminal_name"],
            "state":           terminal_info["state"],
            "country":         terminal_info["country"],
            "product_raw":     str(product_raw).strip(),
            "product_type":    str(product_raw).strip(),  # Exxon already uses clean names
            "price_mxn_per_l": invoice_price,
            "ref_price":       ref_price,
            "discount":        discount,
            "contract_type":   str(contract_type).strip(),
            "source_file":     filepath.name,
        })

    if not rows:
        logger.warning("No data extracted from: %s", filepath.name)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["price_mxn_per_l"] = df["price_mxn_per_l"].astype(float)

    logger.info(
        "Parsed %s -> %d records across %d terminals",
        filepath.name,
        len(df),
        df["terminal_id"].nunique(),
    )
    return df


# -- Folder parser ------------------------------------------------------------

def parse_exxon_folder(folder_path: str) -> pd.DataFrame:
    """
    Parse all Exxon Excel files in a folder and combine into one DataFrame.

    Skips files that fail to parse and logs warnings.
    Removes duplicate records (same date + terminal + product).

    Args:
        folder_path: Path to folder containing exxon_YYYYMMDD.xlsx files.

    Returns:
        Combined pd.DataFrame sorted by date ascending.
    """
    folder = Path(folder_path)
    xlsx_files = sorted(folder.glob("exxon_*.xlsx"))

    if not xlsx_files:
        logger.warning("No Exxon Excel files found in: %s", folder_path)
        return pd.DataFrame()

    logger.info("Found %d Exxon files to parse in: %s", len(xlsx_files), folder_path)

    frames = []
    failed = 0

    for f in xlsx_files:
        df = parse_exxon_file(f)
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
        subset=["date", "terminal_id", "product_raw"]
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
    Flag anomalous prices outside expected MXN/L range.

    Adds boolean column 'price_flag' — True means the price is
    outside [PRICE_MIN, PRICE_MAX] and needs human review.

    Args:
        df: Output of parse_exxon_file() or parse_exxon_folder().

    Returns:
        Same DataFrame with added 'price_flag' column.
    """
    if df.empty:
        return df

    df = df.copy()
    df["price_flag"] = (
        (df["price_mxn_per_l"] < PRICE_MIN) |
        (df["price_mxn_per_l"] > PRICE_MAX)
    )

    flagged = df["price_flag"].sum()
    if flagged:
        logger.warning(
            "%d price(s) outside expected range [%.2f, %.2f]",
            flagged, PRICE_MIN, PRICE_MAX,
        )

    return df


# -- CLI convenience ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_exxon.py <path_to_xlsx_or_folder>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        result = parse_exxon_folder(target)
    else:
        result = parse_exxon_file(target)

    result = validate_prices(result)

    if not result.empty:
        print(result.to_string(index=False))
        print(f"\nShape: {result.shape}")
        flagged = result["price_flag"].sum() if "price_flag" in result.columns else 0
        print(f"Flagged prices: {flagged}")