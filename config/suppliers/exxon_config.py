"""
exxon_config.py
---------------
Config for ExxonMobil daily Excel pricing files.
The generic parser reads this — no Python parser code needed.
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "supplier_name":  "Exxon",
    "file_format":    "xlsx",
    "contract_type":  "Wholesale",
    "country":        "MX",

    # ── File discovery ────────────────────────────────────────────────────
    "file_extensions": [".xlsx", ".xlsm"],

    # ── Date extraction ───────────────────────────────────────────────────
    "date_source":   "title_row",     # date lives in row 0 of the sheet
    "date_patterns": [
        r"(\d{2}/\d{2}/\d{4})",       # 01/01/2024
        r"(\d{8})",                    # fallback: from filename 20240101
    ],
    "date_format_primary":  "%d/%m/%Y",
    "date_format_fallback":  "%Y%m%d",

    # ── Document structure ────────────────────────────────────────────────
    "structure":   "xlsx_table",
    "sheet_name":  "Precios Exxon",   # exact sheet name
    "sheet_index": 0,                 # fallback if sheet name not found
    "header_row":  2,                 # 0-indexed: row 2 (3rd row) has headers
    "data_start":  3,                 # data begins at row 3

    # ── Column mapping ────────────────────────────────────────────────────
    # Maps standard field → list of possible column name keywords
    # Matched by substring, case-insensitive, any language
    # First match wins — put most specific keywords first
    "column_map": {
        "terminal": [
            "terminal"
        ],
        "product": [
            "producto", "product", "fuel", "combustible"
        ],
        "price": [
            "facturaci",        # "Precio Facturación con Impuestos"
            "facturacion",      # without accent
            "invoice",          # English equivalent
            "precio final",     # possible future rename
            "net price",        # English equivalent
            "precio neto",      # Spanish equivalent
        ],
        "discount": [
            "descuento", "discount", "deduccion"
        ],
        "ref_price": [
            "referencia", "reference", "base price", "precio base"
        ],
    },

    # ── Price extraction ──────────────────────────────────────────────────
    "price_unit":       "MXN_per_liter",
    "price_guardrails": {"min": 15.0, "max": 35.0},

    # ── Terminal parsing ──────────────────────────────────────────────────
    # Exxon format: "0TJB - MTY MOBIL"
    "terminal_pattern": r"^(.+?)\s*-\s*(.+)$",
    "terminal_groups": {
        "terminal_id":   1,
        "terminal_name": 2,
    },

    # ── Product normalization ─────────────────────────────────────────────
    "product_map": {
        "regular":  "Regular",
        "87":       "Regular",
        "magna":    "Regular",
        "premium":  "Premium",
        "91":       "Premium",
        "diesel":   "Diesel",
    },
}
