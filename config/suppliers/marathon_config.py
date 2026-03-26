"""
marathon_config.py
------------------
Config for Marathon Petroleum daily TXT email pricing files.
The generic parser reads this — no Python parser code needed.
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "supplier_name":  "Marathon",
    "file_format":    "txt",
    "contract_type":  "Unbranded",
    "country":        "MX",

    # ── File discovery ────────────────────────────────────────────────────
    "file_extensions": [".txt"],

    # ── Date extraction ───────────────────────────────────────────────────
    # Marathon encodes date AND terminal in the filename
    # marathon_20240101_MX-IT-Chihuahua.txt
    "date_source":   "filename",
    "date_patterns": [
        r"(\d{8})",     # first 8-digit block is the date
    ],
    "date_format":   "%Y%m%d",

    # ── Terminal extraction from filename ─────────────────────────────────
    # Full pattern: marathon_YYYYMMDD_COUNTRY-IT-CityName.txt
    "terminal_from_filename": True,
    "terminal_pattern": r"(\d{8})_((MX|US)-IT-(.+))\.txt$",
    "terminal_groups": {
        "terminal_id":   2,    # full "MX-IT-Chihuahua"
        "country_code":  3,    # "MX" or "US"
        "terminal_name": 4,    # "Chihuahua"
    },

    # ── Document structure ────────────────────────────────────────────────
    # Marathon files are plain-text emails with labelled rows
    # Each label appears at the start of a line, followed by 3 price values
    # (one per product: Regular, Premium, Diesel)
    "structure": "txt_labelled_rows",

    # The label we use as the final price (Invoice Price after all taxes)
    # List in priority order — first one found in file wins
    "price_labels": [
        "Invoice Price",
        "invoice price",
        "Precio Factura",
        "precio factura",
        "Final Price",
        "precio final",
    ],

    # All labels to extract (kept for traceability, not used in optimization)
    "all_labels": [
        "Base Price",
        "Unit Price",
        "IVA",
        "Invoice Price",
        "IEPS 2D",
        "IEPS 2H",
        "IEPS 2A",
    ],

    # Product order in the 3-value rows (always: Regular, Premium, Diesel)
    "product_order": ["Regular", "Premium", "Diesel"],

    # ── Price extraction ──────────────────────────────────────────────────
    "price_unit":       "MXN_per_liter",
    "price_guardrails": {"min": 15.0, "max": 35.0},

    # ── Product normalization ─────────────────────────────────────────────
    # Marathon uses "UNBRANDED REGULAR" etc — normalize to standard types
    "product_map": {
        "regular":  "Regular",
        "87":       "Regular",
        "magna":    "Regular",
        "premium":  "Premium",
        "91":       "Premium",
        "diesel":   "Diesel",
    },
}
