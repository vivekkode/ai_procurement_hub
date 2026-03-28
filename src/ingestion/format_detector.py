"""
format_detector.py
------------------
Detects the file format of any incoming supplier file.

This is Step 1 of the generic ingestion layer. It has zero knowledge
of any specific supplier. Its only job is to answer:
"What kind of file is this?"

It uses TWO signals to determine format — not just the file extension,
because extensions can be wrong (a .txt file that is actually CSV,
a .xlsx file renamed to .xls, etc.):

    1. File extension  — fast, first check
    2. File signature  — reads the first few bytes (magic bytes) of
                         the file to confirm what it actually is

Supported formats:
    xlsx  — Excel workbook (OpenXML)
    xls   — Legacy Excel workbook
    pdf   — PDF document
    html  — HTML document
    csv   — Comma/tab separated values
    txt   — Plain text (emails, price lists, any unstructured text)
    json  — JSON data
    xml   — XML data
    unknown — anything else → goes to LLM fallback

Usage:
    from src.ingestion.format_detector import detect_format, FileFormat

    fmt = detect_format("data/raw/bp/bp_pricing_20240101.xlsx")
    print(fmt)           # FileFormat.XLSX
    print(fmt.value)     # "xlsx"
    print(fmt.is_tabular)   # True  (has rows and columns)
    print(fmt.is_structured) # True  (has clear structure)
    print(fmt.needs_llm)    # False (can be parsed with rules)
"""

import os
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FileFormat enum — all supported formats with metadata
# ---------------------------------------------------------------------------

class FileFormat(Enum):
    """
    Represents a detected file format with metadata about how to handle it.

    Attributes:
        value        : string identifier used in configs and logs
        is_tabular   : True if file has rows and columns (Excel, CSV)
        is_structured: True if file has predictable structure (tabular + HTML)
        needs_llm    : True if format alone suggests LLM fallback is likely
    """

    XLSX    = "xlsx"
    XLS     = "xls"
    PDF     = "pdf"
    HTML    = "html"
    CSV     = "csv"
    TXT     = "txt"
    JSON    = "json"
    XML     = "xml"
    UNKNOWN = "unknown"

    @property
    def is_tabular(self) -> bool:
        """True if the format natively has rows and columns."""
        return self in (FileFormat.XLSX, FileFormat.XLS, FileFormat.CSV)

    @property
    def is_structured(self) -> bool:
        """
        True if the format has predictable, machine-readable structure.
        Tabular formats + HTML (structured markup) qualify.
        PDF, TXT are structured-ish but require more interpretation.
        """
        return self in (
            FileFormat.XLSX, FileFormat.XLS,
            FileFormat.CSV,  FileFormat.HTML,
            FileFormat.JSON, FileFormat.XML,
        )

    @property
    def needs_llm(self) -> bool:
        """
        True if this format is likely to need LLM assistance.
        TXT emails and unknown formats almost always need LLM.
        PDFs sometimes do (if they are image-based or irregular).
        """
        return self in (FileFormat.TXT, FileFormat.UNKNOWN)

    @property
    def reader_library(self) -> str:
        """Which Python library to use to read this format."""
        mapping = {
            FileFormat.XLSX:    "openpyxl",
            FileFormat.XLS:     "xlrd",
            FileFormat.PDF:     "pdfminer.six",
            FileFormat.HTML:    "beautifulsoup4",
            FileFormat.CSV:     "pandas",
            FileFormat.TXT:     "built-in",
            FileFormat.JSON:    "built-in json",
            FileFormat.XML:     "built-in xml",
            FileFormat.UNKNOWN: "llm-fallback",
        }
        return mapping.get(self, "unknown")

    def __str__(self):
        return self.value


# ---------------------------------------------------------------------------
# Magic byte signatures — first bytes of a file reveal its true format
# regardless of what the extension says
# ---------------------------------------------------------------------------

# Each entry: (byte_sequence, offset_from_start, FileFormat)
# offset is where in the file those bytes appear
# most signatures are at offset 0 (the very start of the file)

MAGIC_SIGNATURES = [
    # XLSX / DOCX / any OpenXML format (they are ZIP files internally)
    # PK\x03\x04 is the ZIP magic header
    (b"PK\x03\x04",                    0,  FileFormat.XLSX),

    # Legacy XLS (BIFF format)
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", 0, FileFormat.XLS),

    # PDF — always starts with %PDF
    (b"%PDF",                           0,  FileFormat.PDF),

    # HTML — look for <!DOCTYPE or <html (case-insensitive handled below)
    (b"<!DOCTYPE",                      0,  FileFormat.HTML),
    (b"<!doctype",                      0,  FileFormat.HTML),
    (b"<html",                          0,  FileFormat.HTML),
    (b"<HTML",                          0,  FileFormat.HTML),

    # JSON — starts with { or [ (after optional whitespace)
    # handled separately since it needs whitespace stripping

    # XML
    (b"<?xml",                          0,  FileFormat.XML),
]

# How many bytes to read for magic detection
MAGIC_READ_BYTES = 16


# ---------------------------------------------------------------------------
# Extension map — fallback when magic bytes are inconclusive
# ---------------------------------------------------------------------------

EXTENSION_MAP = {
    ".xlsx": FileFormat.XLSX,
    ".xlsm": FileFormat.XLSX,   # macro-enabled Excel, same format
    ".xls":  FileFormat.XLS,
    ".pdf":  FileFormat.PDF,
    ".html": FileFormat.HTML,
    ".htm":  FileFormat.HTML,
    ".csv":  FileFormat.CSV,
    ".tsv":  FileFormat.CSV,    # tab-separated, handled same as CSV
    ".txt":  FileFormat.TXT,
    ".json": FileFormat.JSON,
    ".xml":  FileFormat.XML,
}


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def detect_format(filepath: str) -> FileFormat:
    """
    Detect the file format of any incoming supplier file.

    Uses two signals in order:
        1. Magic bytes  — reads first 16 bytes, most reliable
        2. Extension    — fallback if magic bytes are inconclusive

    Args:
        filepath: Path to any file (string or Path object)

    Returns:
        FileFormat enum member — never raises, always returns something.
        Returns FileFormat.UNKNOWN if format cannot be determined.

    Examples:
        detect_format("data/raw/bp/pricing.xlsx")   → FileFormat.XLSX
        detect_format("data/raw/bp/email.txt")      → FileFormat.TXT
        detect_format("data/raw/bp/report.pdf")     → FileFormat.PDF
        detect_format("data/raw/bp/mystery_file")   → FileFormat.UNKNOWN
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return FileFormat.UNKNOWN

    if filepath.stat().st_size == 0:
        logger.warning("Empty file: %s", filepath)
        return FileFormat.UNKNOWN

    # -- Signal 1: Magic bytes ------------------------------------------------
    fmt = _detect_by_magic(filepath)
    if fmt != FileFormat.UNKNOWN:
        # Log if extension disagrees with magic bytes
        ext_fmt = _detect_by_extension(filepath)
        if ext_fmt != FileFormat.UNKNOWN and ext_fmt != fmt:
            logger.warning(
                "Extension says %s but magic bytes say %s for: %s — "
                "trusting magic bytes",
                ext_fmt, fmt, filepath.name
            )
        logger.debug("Detected %s via magic bytes: %s", fmt, filepath.name)
        return fmt

    # -- Signal 2: Extension --------------------------------------------------
    fmt = _detect_by_extension(filepath)
    if fmt != FileFormat.UNKNOWN:
        logger.debug("Detected %s via extension: %s", fmt, filepath.name)
        return fmt

    # -- Cannot determine -----------------------------------------------------
    logger.warning(
        "Could not determine format for: %s — will route to LLM fallback",
        filepath.name
    )
    return FileFormat.UNKNOWN


def _detect_by_magic(filepath: Path) -> FileFormat:
    """
    Read the first bytes of the file and match against known signatures.

    Returns FileFormat.UNKNOWN if no signature matches.
    Never raises — catches all read errors gracefully.
    """
    try:
        with open(filepath, "rb") as f:
            header = f.read(MAGIC_READ_BYTES)
    except OSError as e:
        logger.error("Could not read file for magic detection: %s — %s",
                     filepath.name, e)
        return FileFormat.UNKNOWN

    if not header:
        return FileFormat.UNKNOWN

    # Check each known signature
    for signature, offset, fmt in MAGIC_SIGNATURES:
        end = offset + len(signature)
        if header[offset:end] == signature:
            return fmt

    # Special case: JSON starts with { or [ after optional whitespace
    stripped = header.lstrip()
    if stripped and stripped[0:1] in (b"{", b"["):
        return FileFormat.JSON

    # Special case: CSV / TXT — if the file is valid UTF-8 or ASCII text
    # and has no other signature, it is likely text-based
    try:
        sample = header.decode("utf-8", errors="strict")
        # If it contains commas or tabs in the first line, likely CSV
        first_line = sample.split("\n")[0]
        if "," in first_line or "\t" in first_line:
            return FileFormat.CSV
        # Otherwise treat as plain text
        return FileFormat.TXT
    except UnicodeDecodeError:
        pass

    return FileFormat.UNKNOWN


def _detect_by_extension(filepath: Path) -> FileFormat:
    """
    Map file extension to FileFormat.
    Case-insensitive. Returns FileFormat.UNKNOWN if extension is not recognized.
    """
    ext = filepath.suffix.lower()
    return EXTENSION_MAP.get(ext, FileFormat.UNKNOWN)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def detect_all_in_folder(folder_path: str) -> dict:
    """
    Scan a folder and detect the format of every file in it.

    Skips hidden files (starting with .) and directories.

    Args:
        folder_path: Path to folder to scan

    Returns:
        dict mapping filepath (str) → FileFormat
        e.g. {"data/raw/bp/pricing.xlsx": FileFormat.XLSX, ...}

    Example:
        formats = detect_all_in_folder("data/raw/bp/")
        for path, fmt in formats.items():
            print(f"{path} → {fmt}")
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.error("Folder not found: %s", folder_path)
        return {}

    results = {}
    for f in sorted(folder.iterdir()):
        if f.is_dir() or f.name.startswith("."):
            continue
        results[str(f)] = detect_format(str(f))

    logger.info(
        "Scanned %d files in %s: %s",
        len(results),
        folder.name,
        _summarize_formats(results)
    )
    return results


def _summarize_formats(format_map: dict) -> str:
    """Helper to produce a readable summary like 'xlsx:3, pdf:2, txt:1'."""
    from collections import Counter
    counts = Counter(str(fmt) for fmt in format_map.values())
    return ", ".join(f"{fmt}:{count}" for fmt, count in counts.most_common())


def group_by_format(folder_path: str) -> dict:
    """
    Scan a folder and group files by their detected format.

    Returns:
        dict mapping FileFormat → list of file paths
        e.g. {FileFormat.XLSX: ["data/raw/bp/day1.xlsx", ...],
              FileFormat.PDF:  ["data/raw/pemex/week1.pdf", ...]}

    Useful for the generic parser to process all files of the same
    format together.
    """
    format_map = detect_all_in_folder(folder_path)
    groups = {}
    for path, fmt in format_map.items():
        groups.setdefault(fmt, []).append(path)
    return groups


# ---------------------------------------------------------------------------
# CLI — run directly to test on any file or folder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python format_detector.py <file_or_folder>")
        print()
        print("Examples:")
        print("  python format_detector.py data/raw/exxon/exxon_20240101.xlsx")
        print("  python format_detector.py data/raw/pemex/")
        sys.exit(1)

    target = Path(sys.argv[1])

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    if target.is_dir():
        print(f"\nScanning folder: {target}")
        print("-" * 50)
        groups = group_by_format(str(target))
        for fmt, files in sorted(groups.items(), key=lambda x: x[0].value):
            print(f"\n{fmt.value.upper()} ({len(files)} files):")
            for f in files:
                print(f"    {Path(f).name}")
            print(f"    → Library: {fmt.reader_library}")
            print(f"    → Tabular: {fmt.is_tabular}")
            print(f"    → Needs LLM: {fmt.needs_llm}")
    else:
        fmt = detect_format(str(target))
        print(f"\nFile    : {target.name}")
        print(f"Format  : {fmt.value}")
        print(f"Library : {fmt.reader_library}")
        print(f"Tabular : {fmt.is_tabular}")
        print(f"Structured: {fmt.is_structured}")
        print(f"Needs LLM : {fmt.needs_llm}")