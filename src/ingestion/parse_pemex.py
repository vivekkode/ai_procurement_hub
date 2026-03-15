"""
parse_pemex.py
--------------
Parses Pemex weekly wholesale pricing PDF files.

File format:
    pemex/reportes_pdf/pemex_tar_YYYYMMDD_YYYYMMDD.pdf
    - One file per week (covers a date range)
    - 6 pages per PDF (2 pages per product type)
    - 3 product types: Magna (Regular 87 Oct), Premium (91 Oct), Diesel
    - 78 regions total per product (split across 2 pages: 38 + 40)

    PDF structure per product (2 pages):
        Page 1:
            Header: "PETRÓLEOS MEXICANOS"
            Title:  "Precios de venta en TAR aplicables a la gasolina en $/m³"
            Date:   "del 1 al 5 de enero de 2024"
            Label:  "REGIÓN"
            Product: "GASOLINA CON CONTENIDO MÍNIMO 87 OCTANOS" / "DIESEL"
            Type:   "PEMEX MAGNA" / "PEMEX PREMIUM" / "PEMEX DIESEL"
            Then:   38 region names (all uppercase)
            Then:   38 prices (format: 19,645.6849)

        Page 2:
            Continuation with remaining 40 regions + prices

    CRITICAL: Pemex prices are in MXN per CUBIC METER (m³)
    All other suppliers use MXN per LITER (L)
    Conversion: divide by 1000 to get MXN/L

Output schema (matches all other parsers):
    date_start      | YYYY-MM-DD (start of validity period)
    date_end        | YYYY-MM-DD (end of validity period)
    date            | YYYY-MM-DD (date_start, for consistency with other parsers)
    supplier        | "Pemex"
    terminal_id     | region name e.g. "CHIHUAHUA"
    terminal_name   | title-cased e.g. "Chihuahua"
    state           | "" (Pemex does not provide state codes)
    country         | "MX"
    product_raw     | original product label from PDF
    product_type    | normalized: "Regular", "Premium", "Diesel"
    price_mxn_per_l | float - price converted from MXN/m³ to MXN/L
    price_mxn_per_m3| float - original price in MXN/m³ (kept for reference)
    contract_type   | "TAR" (Terminal de Almacenamiento y Reparto)
    source_file     | filename for traceability

Key differences from other suppliers:
    - Weekly cadence (not daily)
    - Prices in MXN/m³ must be divided by 1000 to get MXN/L
    - 78 regions vs 3-10 terminals for other suppliers
    - No Diesel in some older files (check product_type before using)
    - Government supplier - prices are reference/regulated rates

Usage:
    # Parse a single file
    from src.ingestion.parse_pemex import parse_pemex_file
    df = parse_pemex_file("data/raw/pemex/pemex_tar_20240101_20240105.pdf")

    # Parse all files in a folder
    from src.ingestion.parse_pemex import parse_pemex_folder
    df = parse_pemex_folder("data/raw/pemex/")
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import pdfminer.high_level

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------
# Pemex prices are in MXN per cubic meter — divide by 1000 to get MXN/L
M3_TO_L = 1000.0

# Price guardrails — applied AFTER conversion to MXN/L
PRICE_MIN = 15.0
PRICE_MAX = 35.0

# Product type mapping from PDF labels
PRODUCT_MAP = {
    "87":      "Regular",
    "magna":   "Regular",
    "91":      "Premium",
    "premium": "Premium",
    "diesel":  "Diesel",
}

# Spanish month names for date parsing
MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


# -- Date parsing -------------------------------------------------------------

def parse_pemex_date_range(text: str) -> tuple:
    """
    Extract start and end dates from Pemex date range string.

    Format: "del 1 al 5 de enero de 2024"
            "del 24 al 30 de enero de 2026"
            "del 28 de diciembre de 2024 al 3 de enero de 2025"

    Returns tuple of (date_start_str, date_end_str) as "YYYY-MM-DD"
    or ("", "") if parsing fails.
    """
    if not text:
        return "", ""

    text = text.lower().strip()

    # Pattern 1: "del D al D de MONTH de YYYY" — same month
    match = re.search(
        r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        text
    )
    if match:
        day_start = int(match.group(1))
        day_end   = int(match.group(2))
        month_str = match.group(3)
        year      = int(match.group(4))
        month     = MONTHS_ES.get(month_str, 0)
        if month:
            try:
                date_start = datetime(year, month, day_start).strftime("%Y-%m-%d")
                date_end   = datetime(year, month, day_end).strftime("%Y-%m-%d")
                return date_start, date_end
            except ValueError:
                pass

    # Pattern 2: "del D de MONTH de YYYY al D de MONTH de YYYY" — spans months
    match = re.search(
        r"del\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s+al\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        text
    )
    if match:
        try:
            date_start = datetime(
                int(match.group(3)),
                MONTHS_ES.get(match.group(2), 1),
                int(match.group(1))
            ).strftime("%Y-%m-%d")
            date_end = datetime(
                int(match.group(6)),
                MONTHS_ES.get(match.group(5), 1),
                int(match.group(4))
            ).strftime("%Y-%m-%d")
            return date_start, date_end
        except ValueError:
            pass

    logger.warning("Could not parse date range from: %r", text)
    return "", ""


def extract_dates_from_filename(filename: str) -> tuple:
    """
    Fallback: extract dates from filename pattern.
    pemex_tar_YYYYMMDD_YYYYMMDD.pdf

    Returns (date_start, date_end) as "YYYY-MM-DD" strings.
    """
    match = re.search(r"pemex_tar_(\d{8})_(\d{8})", filename)
    if match:
        try:
            d1 = datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
            d2 = datetime.strptime(match.group(2), "%Y%m%d").strftime("%Y-%m-%d")
            return d1, d2
        except ValueError:
            pass
    return "", ""


# -- Product type detection ---------------------------------------------------

def detect_product_type(page_text: str) -> str:
    """
    Detect product type from page header text.

    Order matters: check Diesel first because some pages contain
    the word "premium" in the header text even on diesel pages
    (e.g. "PEMEX DIESEL" vs "PEMEX PREMIUM").

    Returns "Regular", "Premium", "Diesel", or "" if unknown.
    """
    # Check first 200 chars only — that's where the header lives
    header = page_text[:200].lower()

    # Diesel must be checked before Premium to avoid false matches
    if "pemex diesel" in header or (
        "diesel" in header and "premium" not in header and "magna" not in header
    ):
        return "Diesel"
    elif "87" in header or "magna" in header:
        return "Regular"
    elif "91" in header or "premium" in header:
        return "Premium"

    return ""


def detect_product_raw(page_text: str) -> str:
    """Extract the raw product label from page header."""
    if "PEMEX MAGNA" in page_text:
        return "GASOLINA CON CONTENIDO MÍNIMO 87 OCTANOS - PEMEX MAGNA"
    elif "PEMEX PREMIUM" in page_text:
        return "GASOLINA CON CONTENIDO MÍNIMO 91 OCTANOS - PEMEX PREMIUM"
    elif "PEMEX DIESEL" in page_text or "DIESEL" in page_text:
        return "DIESEL - PEMEX DIESEL"
    return "UNKNOWN"


# -- Core page parser ---------------------------------------------------------

def parse_page(page_text: str, date_start: str, date_end: str,
               source_file: str) -> list:
    """
    Parse one PDF page into a list of price records.

    Each Pemex page has all region names listed first, then all prices
    in the exact same order. We split the lines into two groups:
    - Lines that are region names (all uppercase letters/spaces)
    - Lines that look like prices (format: 19,645.6849)

    Args:
        page_text:   Full text of one PDF page
        date_start:  Start of validity period
        date_end:    End of validity period
        source_file: Filename for traceability

    Returns:
        List of dicts, one per region/price pair.
    """
    product_type = detect_product_type(page_text)
    product_raw  = detect_product_raw(page_text)

    if not product_type:
        return []

    lines = [l.strip() for l in page_text.splitlines() if l.strip()]

    # Collect region names and prices separately
    # Region names: all uppercase, no digits, length > 2
    # Skip known header lines
    skip_words = {
        "PETRÓLEOS MEXICANOS", "REGIÓN", "PEMEX MAGNA", "PEMEX PREMIUM",
        "PEMEX DIESEL", "DIESEL", "GASOLINA CON CONTENIDO MÍNIMO 87 OCTANOS",
        "GASOLINA CON CONTENIDO MÍNIMO 91 OCTANOS",
        "GASOLINA CON CONTENIDO MÍNIMO 87 OCTANOS - PEMEX MAGNA",
    }

    regions = []
    prices  = []

    for line in lines:
        # Price pattern: digits, comma, digits, period, digits
        # e.g. "19,645.6849" or "21,467.2476"
        if re.match(r"^\d{2,3},\d{3}\.\d{4}$", line):
            prices.append(float(line.replace(",", "")))

        # Region name: uppercase letters, spaces, accented chars
        # Must not be a known header word
        elif (
            line.upper() == line
            and not any(c.isdigit() for c in line)
            and len(line) > 2
            and line not in skip_words
            and "MÍNIMO" not in line
            and "OCTANOS" not in line
            and "CONTENIDO" not in line
            and "SINTÉTICOS" not in line
            and "DATOS" not in line
            and "ACADÉMICOS" not in line
            and "PETRÓLEOS" not in line
            and "MEXICANOS" not in line
            and "EJERCICIOS" not in line
        ):
            regions.append(line)

    # Pair regions with prices
    if len(regions) != len(prices):
        logger.debug(
            "Region/price mismatch on page: %d regions, %d prices — product: %s",
            len(regions), len(prices), product_type
        )
        # Use minimum to avoid index errors
        count = min(len(regions), len(prices))
    else:
        count = len(regions)

    records = []
    for i in range(count):
        price_m3 = prices[i]
        price_l  = round(price_m3 / M3_TO_L, 6)

        records.append({
            "date_start":       date_start,
            "date_end":         date_end,
            "date":             date_start,
            "supplier":         "Pemex",
            "terminal_id":      regions[i],
            "terminal_name":    regions[i].title(),
            "state":            "",
            "country":          "MX",
            "product_raw":      product_raw,
            "product_type":     product_type,
            "price_mxn_per_l":  price_l,
            "price_mxn_per_m3": price_m3,
            "contract_type":    "TAR",
            "source_file":      source_file,
        })

    return records


# -- Core file parser ---------------------------------------------------------

def parse_pemex_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single Pemex PDF pricing file into a tidy DataFrame.

    Each row represents one product at one region for one validity period.
    A typical file produces ~234 rows (78 regions x 3 products).

    Args:
        filepath: Path to the .pdf file.

    Returns:
        pd.DataFrame with columns matching output schema.
        Returns empty DataFrame on failure.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return pd.DataFrame()

    try:
        text = pdfminer.high_level.extract_text(str(filepath))
    except Exception as e:
        logger.error("Could not read PDF %s: %s", filepath.name, e)
        return pd.DataFrame()

    if not text:
        logger.warning("Empty PDF: %s", filepath.name)
        return pd.DataFrame()

    # -- Extract date range ---------------------------------------------------
    # Look for date string in first 200 characters
    date_start, date_end = parse_pemex_date_range(text[:200])

    # Fallback to filename
    if not date_start:
        date_start, date_end = extract_dates_from_filename(filepath.name)
        if date_start:
            logger.debug("Used filename dates for %s", filepath.name)
        else:
            logger.warning("No dates found for %s", filepath.name)

    # -- Parse each page ------------------------------------------------------
    pages = text.split("\f")
    all_records = []

    for page in pages:
        page = page.strip()
        if not page:
            continue

        records = parse_page(page, date_start, date_end, filepath.name)
        all_records.extend(records)

    if not all_records:
        logger.warning("No records extracted from: %s", filepath.name)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df["price_mxn_per_l"] = df["price_mxn_per_l"].astype(float)

    logger.info(
        "Parsed %s -> %d records across %d regions",
        filepath.name,
        len(df),
        df["terminal_id"].nunique(),
    )
    return df


# -- Folder parser ------------------------------------------------------------

def parse_pemex_folder(folder_path: str) -> pd.DataFrame:
    """
    Parse all Pemex PDF files in a folder and combine into one DataFrame.

    Args:
        folder_path: Path to folder containing pemex_tar_*.pdf files.

    Returns:
        Combined pd.DataFrame sorted by date ascending.
    """
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("pemex_tar_*.pdf"))

    if not pdf_files:
        logger.warning("No Pemex PDF files found in: %s", folder_path)
        return pd.DataFrame()

    logger.info("Found %d Pemex PDF files to parse in: %s",
                len(pdf_files), folder_path)

    frames = []
    failed = 0

    for f in pdf_files:
        df = parse_pemex_file(f)
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
        subset=["date_start", "terminal_id", "product_type"]
    )
    dupes_dropped = before - len(combined)
    if dupes_dropped:
        logger.info("Dropped %d duplicate rows", dupes_dropped)

    combined = combined.sort_values("date").reset_index(drop=True)

    logger.info(
        "Total: %d records | %d regions | %d unique weeks | %d files failed",
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

    Note: Pemex prices are converted from MXN/m³ before this check.
    A common mistake would be forgetting the /1000 conversion — prices
    would then be ~1000x too high and all would be flagged.

    Args:
        df: Output of parse_pemex_file() or parse_pemex_folder().

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
            "%d price(s) outside expected range [%.2f, %.2f] MXN/L — "
            "check conversion from m³",
            flagged, PRICE_MIN, PRICE_MAX,
        )

    return df


# -- CLI convenience ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_pemex.py <path_to_pdf_or_folder>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        result = parse_pemex_folder(target)
    else:
        result = parse_pemex_file(target)

    result = validate_prices(result)

    if not result.empty:
        print(result.to_string(index=False))
        print(f"\nShape: {result.shape}")
        flagged = result["price_flag"].sum() if "price_flag" in result.columns else 0
        print(f"Flagged prices: {flagged}")
        if not result.empty:
            print(f"Date range: {result['date'].min()} to {result['date'].max()}")
            print(f"Sample price (Chihuahua/Regular): ",
                  result[
                      (result["terminal_id"] == "CHIHUAHUA") &
                      (result["product_type"] == "Regular")
                  ]["price_mxn_per_l"].values[:1])