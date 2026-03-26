"""
pemex_config.py
---------------
Config for PEMEX weekly PDF pricing files.
The generic parser reads this — no Python parser code needed.
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "supplier_name":  "Pemex",
    "file_format":    "pdf",
    "contract_type":  "TAR",
    "country":        "MX",

    # ── File discovery ────────────────────────────────────────────────────
    "file_extensions": [".pdf"],

    # ── Date extraction ───────────────────────────────────────────────────
    # Pemex has Spanish date ranges in the document body
    # "del 1 al 5 de enero de 2024"
    "date_source": "content",
    "date_patterns": [
        # Same month: "del D al D de MONTH de YYYY"
        r"del\s+(\d{1,2})\s+al\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        # Cross month: "del D de MONTH de YYYY al D de MONTH de YYYY"
        r"del\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s+al\s+(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
    ],
    # Fallback: extract from filename pemex_tar_YYYYMMDD_YYYYMMDD.pdf
    "date_format_fallback": "%Y%m%d",
    "date_pattern_filename": r"(\d{8})_(\d{8})",

    # Spanish month name lookup — full and abbreviated
    "months_es": {
        "enero": 1,   "febrero": 2,  "marzo": 3,     "abril": 4,
        "mayo": 5,    "junio": 6,    "julio": 7,     "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
        "ene": 1, "feb": 2, "mar": 3, "abr": 4,
        "may": 5, "jun": 6, "jul": 7, "ago": 8,
        "sep": 9, "oct": 10, "nov": 11, "dic": 12,
    },

    # ── Document structure ────────────────────────────────────────────────
    # Pemex PDFs have no table headers — raw text lines are either:
    #   - Region names (ALL CAPS, no digits, length > 2)
    #   - Prices      (format: 19,645.6849 — digits,comma,digits.4decimals)
    "structure": "pdf_raw_text",

    # Price number pattern — specific to Pemex's MXN/m³ format
    "price_pattern": r"^\d{2,3},\d{3}\.\d{4}$",

    # ── Unit conversion ───────────────────────────────────────────────────
    # CRITICAL: Pemex quotes in MXN per CUBIC METER — must divide by 1000
    "price_unit":       "MXN_per_m3",
    "convert_to_unit":  "MXN_per_liter",
    "conversion_factor": 1000.0,      # divide by 1000
    "price_guardrails": {"min": 15.0, "max": 35.0},   # applied AFTER conversion

    # ── Header skip words ─────────────────────────────────────────────────
    # Lines to ignore when scanning for region names
    # This list drives resilience — adding new header variants here
    # handles format changes without touching any parser code
    "skip_words": [
        "PETRÓLEOS MEXICANOS", "PETROLEOS MEXICANOS",
        "REGIÓN", "REGION",
        "ZONA", "ZONA GEOGRÁFICA", "ZONA GEOGRAFICA", "LOCALIDAD",
        "PEMEX MAGNA", "PEMEX PREMIUM", "PEMEX DIESEL", "DIESEL",
        "PEMEX MAGNA REGULAR", "MAGNA REGULAR",
        "PEMEX PREMIUM PLUS",  "PREMIUM PLUS",
        "GASOLINA CON CONTENIDO MÍNIMO 87 OCTANOS",
        "GASOLINA CON CONTENIDO MÍNIMO 91 OCTANOS",
        "GASOLINA CON CONTENIDO MÍNIMO 87 OCTANOS - PEMEX MAGNA",
        "GASOLINA CON CONTENIDO MINIMO 87 OCTANOS",
        "GASOLINA CON CONTENIDO MINIMO 91 OCTANOS",
        "GASOLINA 87 OCTANOS", "GASOLINA 91 OCTANOS",
        "GASOLINA MAGNA REGULAR", "GASOLINA PREMIUM PLUS",
    ],

    # Additional substring filters for region name detection
    "skip_substrings": [
        "MÍNIMO", "OCTANOS", "CONTENIDO", "SINTÉTICOS",
        "DATOS", "ACADÉMICOS", "PETRÓLEOS", "MEXICANOS", "EJERCICIOS",
    ],

    # ── Product detection ─────────────────────────────────────────────────
    # Detected from first 200 chars of each page
    "product_map": {
        "87":      "Regular",
        "magna":   "Regular",
        "91":      "Premium",
        "premium": "Premium",
        "diesel":  "Diesel",
    },
}