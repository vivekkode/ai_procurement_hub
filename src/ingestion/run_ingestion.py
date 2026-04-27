"""
run_ingestion.py
----------------
Master ingestion pipeline — discovers all supplier folders and
runs the generic parser on each one.

What changed from the original:
    - No longer imports 4 individual parsers (parse_valero, parse_exxon, etc.)
    - Uses GenericParser + supplier configs for all known suppliers
    - Auto-discovers new supplier folders — drop files in data/raw/<n>/
      and add a config to config/suppliers/<n>_config.py
    - LLM Config Generator handles unknown suppliers with no config
    - LLM Unstructured Parser handles emails and surcharge notices
    - Output is identical — same all_suppliers.csv schema, same row counts

What stays the same:
    - Output schema (all 11 columns)
    - Output file locations (data/processed/)
    - --skip argument behaviour
    - Summary report format

Usage:
    # Run everything (known suppliers from configs)
    python src/ingestion/run_ingestion.py

    # Skip specific suppliers
    python src/ingestion/run_ingestion.py --skip pemex
    python src/ingestion/run_ingestion.py --skip valero exxon

    # Include unstructured documents (emails, surcharge notices)
    python src/ingestion/run_ingestion.py --include-unstructured

    # Force new suppliers through LLM config generation
    python src/ingestion/run_ingestion.py --auto-onboard
"""

import sys
import argparse
import logging
import importlib.util
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.generic_parser       import GenericParser
from src.ingestion.format_detector      import detect_format, FileFormat
from src.ingestion.llm_config_generator import auto_onboard_supplier
from src.ingestion.parse_llm            import UnstructuredParser

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -- Paths --------------------------------------------------------------------
DATA_RAW       = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
CONFIG_DIR     = Path("config/suppliers")

OUTPUT_FILES = {
    "combined":   DATA_PROCESSED / "all_suppliers.csv",
    "surcharges": DATA_PROCESSED / "surcharges.csv",
}

# Known suppliers — folder name → config file name
KNOWN_SUPPLIERS = {
    "valero":   "valero_config",
    "exxon":    "exxon_config",
    "marathon": "marathon_config",
    "pemex":    "pemex_config",
}

# Suppliers whose files live in a subfolder
SUPPLIER_SUBFOLDERS = {
    "pemex": "reportes_pdf",
}

# Individual output CSV per supplier
SUPPLIER_OUTPUT = {
    "valero":   DATA_PROCESSED / "valero_clean.csv",
    "exxon":    DATA_PROCESSED / "exxon_clean.csv",
    "marathon": DATA_PROCESSED / "marathon_clean.csv",
    "pemex":    DATA_PROCESSED / "pemex_clean.csv",
}

# Standard output columns
COMMON_COLUMNS = [
    "date", "supplier", "terminal_id", "terminal_name",
    "state", "country", "product_type", "price_mxn_per_l",
    "contract_type", "source_file", "price_flag",
]

# Folder names that contain unstructured documents
UNSTRUCTURED_FOLDERS = {"notices", "alerts", "surcharges", "emails"}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    """Load a supplier config by module name from config/suppliers/."""
    config_path = CONFIG_DIR / f"{config_name}.py"
    if not config_path.exists():
        return None
    spec   = importlib.util.spec_from_file_location(config_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "CONFIG", None)


# ---------------------------------------------------------------------------
# Supplier runners
# ---------------------------------------------------------------------------

def run_supplier(supplier_name: str, config: dict,
                 skip: bool = False) -> pd.DataFrame:
    """Run the generic parser for one supplier."""
    if skip:
        logger.info("Skipping %s", supplier_name)
        return pd.DataFrame()

    logger.info("=" * 60)
    logger.info("%s — Parsing %s files",
                supplier_name.upper(),
                config.get("file_format", "").upper())
    logger.info("=" * 60)

    subfolder = SUPPLIER_SUBFOLDERS.get(supplier_name)
    folder    = DATA_RAW / supplier_name
    if subfolder:
        folder = folder / subfolder

    if not folder.exists():
        logger.warning("Folder not found: %s — checking for existing clean data", folder)
        return _load_existing_clean(supplier_name)

    parser = GenericParser(config)
    df     = parser.parse_folder(str(folder))

    # ── FALLBACK: if GenericParser found nothing, check for CSV files ──
    # CSV files that already match the standard schema can be ingested
    # directly without needing a format-specific parser
    if df.empty:
        logger.info("%s — No %s files found, checking for CSV fallback...",
                    supplier_name.upper(), config.get("file_format", ""))
        df = _try_csv_direct_ingest(supplier_name, folder)

    # ── FALLBACK 2: load previously generated clean file if it exists ──
    if df.empty:
        logger.info("%s — Checking for existing clean data on disk...",
                    supplier_name.upper())
        df = _load_existing_clean(supplier_name)

    if df.empty:
        logger.warning("No data extracted for: %s", supplier_name)
        return df

    output_path = SUPPLIER_OUTPUT.get(
        supplier_name,
        DATA_PROCESSED / f"{supplier_name}_clean.csv"
    )
    df.to_csv(output_path, index=False)
    logger.info("Saved -> %s  (%d rows)", output_path, len(df))
    return df


def _try_csv_direct_ingest(supplier_name: str, folder: Path) -> pd.DataFrame:
    """
    Look for CSV files in the supplier folder that already match the
    standard schema. This handles the case where someone drops a
    pre-formatted CSV instead of the original supplier format.
    """
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        # Also check parent folder (in case subfolder was used)
        parent = folder.parent
        if parent != DATA_RAW:
            csv_files = list(parent.glob("*.csv"))

    if not csv_files:
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Check if it has enough standard columns to be valid
            matches = [c for c in COMMON_COLUMNS if c in df.columns]
            if len(matches) >= 5:  # at least date, supplier, product, price + 1 more
                logger.info("CSV direct ingest: %s (%d rows, %d/%d columns match)",
                           f.name, len(df), len(matches), len(COMMON_COLUMNS))
                # Fill missing columns with defaults
                if "supplier" not in df.columns:
                    df["supplier"] = supplier_name.title()
                if "source_file" not in df.columns:
                    df["source_file"] = f.name
                if "price_flag" not in df.columns:
                    df["price_flag"] = False
                if "country" not in df.columns:
                    df["country"] = "MX"
                frames.append(df[[c for c in COMMON_COLUMNS if c in df.columns]])
            else:
                logger.debug("CSV %s has only %d matching columns — skipping",
                           f.name, len(matches))
        except Exception as e:
            logger.warning("Could not read CSV %s: %s", f.name, e)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        logger.info("CSV direct ingest total for %s: %d rows",
                   supplier_name, len(combined))
        return combined

    return pd.DataFrame()


def _load_existing_clean(supplier_name: str) -> pd.DataFrame:
    """Load a previously generated *_clean.csv if it exists on disk."""
    clean_path = SUPPLIER_OUTPUT.get(
        supplier_name,
        DATA_PROCESSED / f"{supplier_name}_clean.csv"
    )
    if clean_path.exists():
        try:
            df = pd.read_csv(clean_path, low_memory=False)
            if not df.empty:
                logger.info("Loaded existing clean data: %s (%d rows)",
                           clean_path.name, len(df))
                return df
        except Exception as e:
            logger.warning("Could not read %s: %s", clean_path.name, e)
    return pd.DataFrame()


def run_unstructured_folder(folder: Path,
                             supplier_hint: str = None) -> int:
    """Scan a folder for unstructured documents and extract surcharges."""
    unstructured_parser = UnstructuredParser()
    total_events = 0

    for filepath in sorted(folder.iterdir()):
        if filepath.is_dir():
            continue
        fmt = detect_format(str(filepath))
        if fmt in (FileFormat.TXT, FileFormat.UNKNOWN):
            logger.info("Unstructured document: %s", filepath.name)
            events = unstructured_parser.parse(str(filepath), supplier_hint)
            if events:
                unstructured_parser.save_events(events)
                total_events += len(events)

    return total_events


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_unknown_suppliers(known: set, auto_onboard: bool) -> dict:
    """Scan data/raw/ for supplier folders that have no known config."""
    new_configs = {}

    # Folders to always skip — not supplier data
    SKIP_FOLDERS = {"inbox", "unknown", "notices", "alerts", "surcharges", "emails",
                    "__pycache__", ".git"}

    for folder in sorted(DATA_RAW.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if name in known or name in SKIP_FOLDERS:
            continue

        config_path = CONFIG_DIR / f"{name}_config.py"
        if config_path.exists():
            config = load_config(f"{name}_config")
            if config:
                logger.info("Found config for new supplier: %s", name)
                new_configs[name] = config
            continue

        if not auto_onboard:
            logger.warning(
                "Unknown supplier folder '%s' has no config — "
                "run with --auto-onboard to generate one via LLM",
                name
            )
            continue

        sample_files = [
            f for f in folder.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        if not sample_files:
            logger.warning("No files in unknown folder: %s", folder)
            continue

        logger.info(
            "Auto-onboarding '%s' using: %s",
            name, sample_files[0].name
        )
        try:
            config = auto_onboard_supplier(
                str(sample_files[0]),
                supplier_name=name.title()
            )
            new_configs[name] = config
        except Exception as e:
            logger.error("Could not generate config for '%s': %s", name, e)

    return new_configs


# ---------------------------------------------------------------------------
# Combine and summarise
# ---------------------------------------------------------------------------

def combine_suppliers(frames: dict) -> pd.DataFrame:
    """Combine all supplier DataFrames into all_suppliers.csv."""
    logger.info("=" * 60)
    logger.info("COMBINING all suppliers")
    logger.info("=" * 60)

    valid_frames = []
    for supplier, df in frames.items():
        if df.empty:
            logger.warning(
                "No data for %s — skipping from combined file", supplier
            )
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

    # Normalize dates to consistent YYYY-MM-DD strings
    # This prevents mixed Timestamp/string issues downstream
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    combined = combined.sort_values(
        ["date", "supplier", "terminal_id"]
    ).reset_index(drop=True)
    combined.to_csv(OUTPUT_FILES["combined"], index=False)
    logger.info(
        "Saved -> %s  (%d rows)", OUTPUT_FILES["combined"], len(combined)
    )
    return combined


def print_summary(frames: dict, combined: pd.DataFrame,
                  surcharge_count: int = 0):
    """Print the ingestion summary report."""
    print()
    print("=" * 65)
    print("  INGESTION SUMMARY")
    print("=" * 65)
    print(f"  {'Supplier':<14} {'Records':>8}  {'Terminals':>10}  {'Flagged':>8}")
    print("-" * 65)

    total_records = 0
    total_flagged = 0

    for supplier, df in frames.items():
        if df.empty:
            print(f"  {supplier.capitalize():<14} {'SKIPPED':>8}")
            continue
        records   = len(df)
        terminals = df["terminal_id"].nunique() if "terminal_id" in df.columns else 0
        flagged   = int(df["price_flag"].sum()) if "price_flag" in df.columns else 0
        total_records += records
        total_flagged += flagged
        print(f"  {supplier.capitalize():<14} {records:>8,}  "
              f"{terminals:>10,}  {flagged:>8,}")

    print("-" * 65)
    print(f"  {'TOTAL':<14} {total_records:>8,}  {'':>10}  {total_flagged:>8,}")
    print("=" * 65)

    if not combined.empty:
        print()
        try:
            dates = pd.to_datetime(combined["date"], errors="coerce").dropna()
            if not dates.empty:
                print(f"  Date range     : "
                      f"{dates.min().date()} to "
                      f"{dates.max().date()}")
        except Exception:
            print(f"  Date range     : {combined['date'].min()} to {combined['date'].max()}")
        print(f"  Products       : {sorted(combined['product_type'].unique())}")
        print(f"  Output file    : {OUTPUT_FILES['combined']}")

    if surcharge_count > 0:
        print(f"  Surcharge events: {surcharge_count} → "
              f"{OUTPUT_FILES['surcharges']}")

    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the AI Procurement Hub data ingestion pipeline"
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Supplier names to skip (e.g. --skip pemex valero)",
    )
    parser.add_argument(
        "--auto-onboard", action="store_true",
        help="Auto-generate configs for unknown supplier folders via LLM",
    )
    parser.add_argument(
        "--include-unstructured", action="store_true",
        help="Scan for unstructured emails/notices and extract surcharges",
    )
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ingestion pipeline")
    logger.info("Using: GenericParser + supplier configs")
    if args.skip:
        logger.info("Skipping: %s", args.skip)

    skip_set = {s.lower() for s in args.skip}

    # Step A — load known supplier configs
    all_configs = {}
    for supplier_name, config_name in KNOWN_SUPPLIERS.items():
        config = load_config(config_name)
        if config:
            all_configs[supplier_name] = config
        else:
            logger.warning(
                "Config not found for '%s': %s", supplier_name, config_name
            )

    # Step B — discover unknown suppliers
    known_names = set(KNOWN_SUPPLIERS.keys()) | skip_set
    new_configs = discover_unknown_suppliers(known_names, args.auto_onboard)
    all_configs.update(new_configs)

    # Step C — run generic parser for each supplier
    frames = {}
    for supplier_name, config in all_configs.items():
        frames[supplier_name] = run_supplier(
            supplier_name, config,
            skip=(supplier_name in skip_set)
        )

    # Step D — scan for unstructured documents
    surcharge_count = 0
    if args.include_unstructured:
        logger.info("=" * 60)
        logger.info("UNSTRUCTURED — Scanning for emails and notices")
        logger.info("=" * 60)

        for folder_name in UNSTRUCTURED_FOLDERS:
            folder = DATA_RAW / folder_name
            if folder.exists():
                surcharge_count += run_unstructured_folder(folder)

        for supplier_name in all_configs:
            notices = DATA_RAW / supplier_name / "notices"
            if notices.exists():
                surcharge_count += run_unstructured_folder(
                    notices, supplier_hint=supplier_name.title()
                )

    # Step E — combine and save
    combined = combine_suppliers(frames)
    print_summary(frames, combined, surcharge_count)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()