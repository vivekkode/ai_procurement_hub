"""
valero_config.py
----------------
Config for Valero daily HTML pricing files.
The generic parser reads this — no Python parser code needed.
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "supplier_name":  "Valero",
    "file_format":    "html",
    "contract_type":  "Sin Marca",
    "country":        "MX",

    # ── File discovery ────────────────────────────────────────────────────
    # Any .html file in the supplier folder — not just valero_*.html
    "file_extensions": [".html", ".htm"],

    # ── Date extraction ───────────────────────────────────────────────────
    # Try filename first, then fall back to content
    "date_source":   "filename",
    "date_patterns": [
        r"(\d{8})",          # valero_20240101.html -> 20240101
    ],
    "date_format":   "%Y%m%d",

    # ── Document structure ────────────────────────────────────────────────
    "structure": "html_cards",   # tells generic parser to look for card blocks

    # HTML selectors — what to look for in the page
    "selectors": {
        "card":    {"tag": "div", "class": "terminal-card"},
        "header":  {"tag": "div", "class": "card-header"},
        "table":   {"tag": "table"},
    },

    # ── Terminal parsing ──────────────────────────────────────────────────
    # Valero header format: "Altamira, TMS, MX - TMX000001"
    "terminal_pattern": r"^(.+?),\s*([A-Z]{2,3}),\s*(MX|US)\s*-\s*(\S+)$",
    "terminal_groups":  {
        "terminal_name": 1,
        "state":         2,
        "country":       3,
        "terminal_id":   4,
    },
    # Fallback pattern: "Name - ID"
    "terminal_pattern_fallback": r"^(.+?)\s*-\s*(\S+)$",
    "terminal_groups_fallback": {
        "terminal_name": 1,
        "terminal_id":   2,
    },

    # ── Price extraction ──────────────────────────────────────────────────
    "price_unit":        "MXN_per_liter",
    "price_column_index": 3,    # 4th cell: Current Price (after Effective Since timestamp)
    "price_guardrails":  {"min": 15.0, "max": 35.0},

    # ── Product normalization ─────────────────────────────────────────────
    # Maps any string containing these keywords → standard product type
    "product_map": {
        "87":       "Regular",
        "regular":  "Regular",
        "magna":    "Regular",
        "91":       "Premium",
        "premium":  "Premium",
        "diesel":   "Diesel",
    },
}