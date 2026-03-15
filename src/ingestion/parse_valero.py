"""
parse_valero.py
---------------
Parses Valero daily wholesale pricing HTML files.

File format:
    valero/valero_YYYYMMDD.html
    - One file per trading day
    - Contains multiple terminal-card blocks
    - Each card has a terminal name + table of product prices

Output schema (one row per product per terminal per day):
    date            | YYYY-MM-DD
    supplier        | "Valero"
    terminal_id     | e.g. "TMX000001"
    terminal_name   | e.g. "Altamira"
    state           | e.g. "TMS"
    country         | "MX" or "US"
    product_raw     | original product string from file
    product_type    | normalized: "Regular", "Premium", "Diesel"
    price_mxn_per_l | float — current invoice price MXN/Liter
    prev_price      | float — previous price for change tracking
    contract_type   | e.g. "Sin Marca"
    source_file     | filename for traceability

Usage:
    # Parse a single file
    from src.ingestion.parse_valero import parse_valero_file
    df = parse_valero_file("data/raw/valero/valero_20240101.html")

    # Parse all files in a folder
    from src.ingestion.parse_valero import parse_valero_folder
    df = parse_valero_folder("data/raw/valero/")
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Product normalization map ─────────────────────────────────────────────────
# Valero uses verbose product names like "91 Oct RDP 10.0 RVP" or
# "87 Oct Regular 7.8 RVP". We map these to three canonical types
# used consistently across all suppliers in this project.

PRODUCT_MAP = {
    # Regular / Magna (87 octane family)
    "87": "Regular",
    "regular": "Regular",
    "magna": "Regular",
    # Premium (91 octane family)
    "91": "Premium",
    "premium": "Premium",
    # Diesel
    "diesel": "Diesel",
}


def normalize_product(raw: str) -> str:
    """
    Map Valero's verbose product descriptions to a canonical product type.

    Examples:
        "91 Oct RDP 10.0 RVP"          -> "Premium"
        "87 Oct Regular 10.0 RVP"      -> "Regular"
        "87 Oct Convencional 10.0 RVP" -> "Regular"
        "Diesel"                       -> "Diesel"
        "Sin clasificar"               -> "Unknown"
    """
    lower = raw.lower()
    for key, label in PRODUCT_MAP.items():
        if key in lower:
            return label
    logger.warning("Could not normalize product: %r — labelling as 'Unknown'", raw)
    return "Unknown"


# ── Terminal parsing helpers ──────────────────────────────────────────────────

def parse_terminal_header(header_text: str) -> dict:
    """
    Extract structured fields from a Valero terminal header string.

    Header formats observed in the data:
        "Altamira, TMS, MX - TMX000001"
        "TRUCK BUY/SELL-El Paso"
        "TRUCK B/S-Citgo-Brownsville-MX - T74-TX-2709"

    Returns a dict with keys: terminal_name, state, country, terminal_id.
    Falls back gracefully if the format is unexpected.
    """
    # Remove the "Save / Print" suffix that sometimes bleeds in
    header_text = header_text.replace("Save / Print", "").strip()

    # Pattern: "Name, STATE, COUNTRY - ID"
    match = re.match(
        r"^(.+?),\s*([A-Z]{2,3}),\s*(MX|US)\s*-\s*(\S+)$",
        header_text,
        re.IGNORECASE,
    )
    if match:
        return {
            "terminal_name": match.group(1).strip(),
            "state": match.group(2).strip().upper(),
            "country": match.group(3).strip().upper(),
            "terminal_id": match.group(4).strip(),
        }

    # Pattern: "Name - ID" (no state/country)
    match = re.match(r"^(.+?)\s*-\s*(\S+)$", header_text)
    if match:
        name = match.group(1).strip()
        tid = match.group(2).strip()
        # Infer country from name
        country = "US" if any(
            us in name.upper() for us in ["EL PASO", "HARLINGEN", "BROWNSVILLE"]
        ) else "MX"
        return {
            "terminal_name": name,
            "state": "",
            "country": country,
            "terminal_id": tid,
        }

    # Fallback: use entire string as name
    return {
        "terminal_name": header_text,
        "state": "",
        "country": "MX",
        "terminal_id": "",
    }


def extract_date_from_filename(filename: str) -> str:
    """
    Extract ISO date from Valero filename pattern: valero_YYYYMMDD.html

    Returns "YYYY-MM-DD" string or empty string if not parseable.
    """
    match = re.search(r"valero_(\d{8})", filename)
    if match:
        raw = match.group(1)
        try:
            return datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            logger.warning("Could not parse date from filename: %s", filename)
    return ""


def safe_float(value: str) -> float:
    """Convert a price string to float, returning None if not parseable."""
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_valero_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single Valero HTML pricing file into a tidy DataFrame.

    Each row represents one product at one terminal on one date.

    Args:
        filepath: Path to the .html file.

    Returns:
        pd.DataFrame with columns matching the output schema defined
        at the top of this module. Returns an empty DataFrame if the
        file cannot be parsed.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return pd.DataFrame()

    date_str = extract_date_from_filename(filepath.name)
    if not date_str:
        logger.warning("No date found in filename: %s", filepath.name)

    try:
        html = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error("Could not read file %s: %s", filepath, e)
        return pd.DataFrame()

    soup = BeautifulSoup(html, "lxml")

    rows = []

    # Each terminal is wrapped in a <div class="terminal-card">
    # Inside: <div class="card-header"><span>Terminal Name</span>...</div>
    #         <table>...</table>
    for card in soup.find_all("div", class_="terminal-card"):

        # ── Terminal name ──
        header_div = card.find("div", class_="card-header")
        if not header_div:
            logger.debug("No card-header found in a terminal-card block, skipping.")
            continue

        # First <span> holds the terminal name; second is "Save / Print"
        spans = header_div.find_all("span")
        header_text = spans[0].get_text(strip=True) if spans else header_div.get_text(strip=True)
        terminal_info = parse_terminal_header(header_text)

        # ── Price table ──
        table = card.find("table")
        if not table:
            logger.debug("No table in terminal card for: %s", header_text)
            continue

        tbody = table.find("tbody")
        if not tbody:
            # Some files have no <tbody>, rows are direct children of <table>
            data_rows = table.find_all("tr")[1:]  # skip header row
        else:
            data_rows = tbody.find_all("tr")

        for tr in data_rows:
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue  # skip malformed rows

            contract_type = cells[0].get_text(strip=True)
            product_raw   = cells[1].get_text(strip=True)
            # cells[2] is "Effective Since" — we use the filename date instead
            current_price = safe_float(cells[3].get_text(strip=True))
            prev_price    = safe_float(cells[4].get_text(strip=True)) if len(cells) > 4 else None

            # Skip rows with no usable price
            if current_price is None:
                logger.debug("No price for %s / %s on %s", header_text, product_raw, date_str)
                continue

            rows.append({
                "date":            date_str,
                "supplier":        "Valero",
                "terminal_id":     terminal_info["terminal_id"],
                "terminal_name":   terminal_info["terminal_name"],
                "state":           terminal_info["state"],
                "country":         terminal_info["country"],
                "product_raw":     product_raw,
                "product_type":    normalize_product(product_raw),
                "price_mxn_per_l": current_price,
                "prev_price":      prev_price,
                "contract_type":   contract_type,
                "source_file":     filepath.name,
            })

    if not rows:
        logger.warning("No data extracted from: %s", filepath.name)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["price_mxn_per_l"] = df["price_mxn_per_l"].astype(float)

    logger.info(
        "Parsed %s → %d records across %d terminals",
        filepath.name,
        len(df),
        df["terminal_id"].nunique(),
    )
    return df


# ── Folder parser ─────────────────────────────────────────────────────────────

def parse_valero_folder(folder_path: str) -> pd.DataFrame:
    """
    Parse all Valero HTML files in a folder and concatenate into one DataFrame.

    Skips files that fail to parse and logs warnings for them.
    Removes duplicate records (same date + terminal + product) keeping
    the first occurrence.

    Args:
        folder_path: Path to folder containing valero_YYYYMMDD.html files.

    Returns:
        Combined pd.DataFrame sorted by date ascending.
    """
    folder = Path(folder_path)
    html_files = sorted(folder.glob("valero_*.html"))

    if not html_files:
        logger.warning("No Valero HTML files found in: %s", folder_path)
        return pd.DataFrame()

    logger.info("Found %d Valero files to parse in: %s", len(html_files), folder_path)

    frames = []
    failed = 0

    for f in html_files:
        df = parse_valero_file(f)
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


# ── Guardrails ────────────────────────────────────────────────────────────────
# As discussed in the Mar 13 meeting, prices below or above realistic
# market ranges should be flagged, not silently passed to the engine.

PRICE_MIN = 15.0  # MXN/L — anything below this is almost certainly an error
PRICE_MAX = 35.0  # MXN/L — ceiling for current Mexican wholesale market


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag anomalous prices in the parsed DataFrame.

    Adds a boolean column 'price_flag' that is True when the price
    falls outside the expected range [PRICE_MIN, PRICE_MAX].
    Does not remove flagged rows — the engine layer decides what to do.

    Args:
        df: Output of parse_valero_file() or parse_valero_folder().

    Returns:
        Same DataFrame with an added 'price_flag' column.
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
            "%d price(s) outside expected range [%.2f, %.2f] — check 'price_flag' column",
            flagged, PRICE_MIN, PRICE_MAX,
        )

    return df


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_valero.py <path_to_html_or_folder>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        result = parse_valero_folder(target)
    else:
        result = parse_valero_file(target)

    result = validate_prices(result)

    if not result.empty:
        print(result.to_string(index=False))
        print(f"\nShape: {result.shape}")
        flagged = result["price_flag"].sum() if "price_flag" in result.columns else 0
        print(f"Flagged prices: {flagged}")