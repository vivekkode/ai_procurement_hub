"""
parse_llm.py
------------
Step 4 of the generic ingestion layer — unstructured document parser.

Handles documents that have NO consistent table structure — emails,
surcharge notices, regulatory announcements, irregular PDFs written
in prose by humans for humans.

The LLM Config Generator (Step 3) handles structured files by
identifying which column maps to which field. This module handles
a completely different problem: extracting procurement-relevant
information from free text where there are no columns at all.

Roberto's requirement (Mar 13 email):
    "Can your agent read an unstructured email or PDF from Valero
    detailing a sudden weather surcharge, automatically extract that
    variable, update the landed cost, and autonomously route the
    purchase to Pemex instead?"

This module answers YES to that question.

What it extracts from unstructured text:
    - Base price changes       → updates price in recommendation engine
    - Surcharge amounts        → added ON TOP of base price in landed cost
    - Effective dates          → surcharge applied only during this window
    - Affected products        → surcharge may apply to Regular but not Diesel
    - Affected terminals       → surcharge is terminal-specific, not global
    - Surcharge reason         → stored for explainability in recommendation

Key difference from Step 3:
    Step 3 → structured file → LLM identifies columns → GenericParser extracts
    Step 4 → unstructured text → LLM reads prose → extracts fields directly
             No config generated. No GenericParser used.
             Fields extracted directly from human-written text.

Output:
    Two separate outputs — surcharges do NOT go into all_suppliers.csv
    because they are temporary adjustments, not base prices:

    1. SurchargeEvent — a dataclass representing one surcharge event
    2. surcharges.csv — persistent store of all active surcharge events,
                        read by the landed cost engine when calculating
                        true cost for a recommendation

How the landed cost engine uses this:
    base_price = all_suppliers.csv price for supplier/terminal/product/date
    surcharge  = surcharges.csv amount for same supplier/terminal/product/date
    landed_cost = base_price + surcharge + freight

Usage:
    from src.ingestion.parse_llm import UnstructuredParser

    parser = UnstructuredParser()

    # Parse a surcharge email
    events = parser.parse(
        filepath="/tmp/valero_surcharge_notice.txt",
        supplier_hint="Valero"       # optional — helps LLM if known
    )

    for event in events:
        print(event)
        # SurchargeEvent(supplier='Valero', terminal='Nuevo Laredo, TMS',
        #                product='Regular', surcharge_per_l=0.85,
        #                effective_from='2026-03-27', effective_to='2026-04-04',
        #                reason='Highway 85 weather closure', confidence='high')

    # Save to surcharges.csv for the landed cost engine
    parser.save_events(events)
"""

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Where active surcharge events are persisted
SURCHARGES_FILE = Path("data/processed/surcharges.csv")

# Known supplier name variants for normalisation
SUPPLIER_ALIASES = {
    "valero":    "Valero",
    "exxon":     "Exxon",
    "exxonmobil":"Exxon",
    "mobil":     "Exxon",
    "marathon":  "Marathon",
    "pemex":     "Pemex",
    "petróleos": "Pemex",
}


# ---------------------------------------------------------------------------
# SurchargeEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class SurchargeEvent:
    """
    One surcharge event extracted from an unstructured document.

    A surcharge is a temporary addition to the base price.
    It is terminal-specific, product-specific, and time-bounded.

    The landed cost engine adds surcharge_per_l to the base price
    for any order that matches supplier + terminal + product
    and falls within the effective date window.
    """
    supplier:        str              # e.g. "Valero"
    terminal:        str              # e.g. "Nuevo Laredo, TMS"
    product:         str              # "Regular", "Premium", "Diesel", or "All"
    surcharge_per_l: float            # MXN per liter
    effective_from:  str              # YYYY-MM-DD
    effective_to:    str              # YYYY-MM-DD
    reason:          str              # human-readable reason
    confidence:      str              # "high", "medium", "low"
    source_file:     str              # original document filename
    extracted_at:    str = field(     # when this was extracted
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def is_active(self, date: str = None) -> bool:
        """
        Check if this surcharge is active on a given date.
        Defaults to today if no date provided.
        """
        check_date = date or datetime.now().strftime("%Y-%m-%d")
        return self.effective_from <= check_date <= self.effective_to

    def applies_to(self, supplier: str, terminal: str,
                   product: str, date: str) -> bool:
        """
        Check if this surcharge applies to a specific order.
        Used by the landed cost engine.
        """
        supplier_match = self.supplier.lower() == supplier.lower()
        terminal_match = (
            self.terminal.lower() in terminal.lower() or
            terminal.lower() in self.terminal.lower()
        )
        product_match  = (
            self.product == "All" or
            self.product.lower() == product.lower()
        )
        date_match = self.is_active(date)

        return supplier_match and terminal_match and product_match and date_match

    def __str__(self):
        return (
            f"SurchargeEvent("
            f"supplier={self.supplier}, "
            f"terminal={self.terminal}, "
            f"product={self.product}, "
            f"+{self.surcharge_per_l} MXN/L, "
            f"{self.effective_from}→{self.effective_to}, "
            f"reason='{self.reason}', "
            f"confidence={self.confidence})"
        )


# ---------------------------------------------------------------------------
# UnstructuredParser
# ---------------------------------------------------------------------------

class UnstructuredParser:
    """
    Extracts procurement-relevant events from unstructured documents.

    Handles: supplier emails, surcharge notices, regulatory PDFs,
             weather alerts, logistics announcements — anything a
             human would write to communicate a price change.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse(self, filepath: str,
              supplier_hint: str = None) -> list:
        """
        Parse an unstructured document and extract surcharge events.

        Args:
            filepath:      Path to any text-based document
            supplier_hint: Supplier name if known (improves LLM accuracy)

        Returns:
            List of SurchargeEvent objects.
            Empty list if no procurement-relevant info found.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            self.logger.error("File not found: %s", filepath)
            return []

        # Read raw content
        content = self._read_content(filepath)
        if not content:
            self.logger.warning("Empty or unreadable file: %s", filepath.name)
            return []

        self.logger.info(
            "Parsing unstructured document: %s (%d chars)",
            filepath.name, len(content)
        )

        # Send to LLM for extraction
        raw_response = self._call_claude(content, supplier_hint, filepath.name)

        # Parse LLM response into SurchargeEvent objects
        events = self._parse_response(raw_response, filepath.name)

        if not events:
            self.logger.info(
                "No procurement-relevant events found in: %s",
                filepath.name
            )
        else:
            self.logger.info(
                "Extracted %d surcharge event(s) from: %s",
                len(events), filepath.name
            )
            for event in events:
                self.logger.info("  %s", event)

        return events

    def parse_text(self, text: str, supplier_hint: str = None,
                   source_label: str = "inline") -> list:
        """
        Parse raw text directly (no file needed).
        Useful for processing email bodies from an API or database.
        """
        raw_response = self._call_claude(text, supplier_hint, source_label)
        return self._parse_response(raw_response, source_label)

    def save_events(self, events: list):
        """
        Append surcharge events to surcharges.csv.

        The landed cost engine reads this file when calculating
        true cost. Events are deduplicated by
        supplier + terminal + product + effective_from.
        """
        if not events:
            return

        SURCHARGES_FILE.parent.mkdir(parents=True, exist_ok=True)

        new_rows = pd.DataFrame([asdict(e) for e in events])

        if SURCHARGES_FILE.exists():
            existing = pd.read_csv(SURCHARGES_FILE)
            combined = pd.concat([existing, new_rows], ignore_index=True)
            # Deduplicate — keep most recently extracted version
            combined = combined.drop_duplicates(
                subset=["supplier", "terminal", "product", "effective_from"],
                keep="last"
            )
        else:
            combined = new_rows

        combined.to_csv(SURCHARGES_FILE, index=False)
        self.logger.info(
            "Saved %d event(s) to %s (total: %d active surcharges)",
            len(events), SURCHARGES_FILE, len(combined)
        )

    def get_active_surcharges(self, date: str = None) -> pd.DataFrame:
        """
        Load all active surcharges for a given date.
        Used by the landed cost engine.

        Args:
            date: YYYY-MM-DD, defaults to today

        Returns:
            DataFrame of active SurchargeEvents
        """
        if not SURCHARGES_FILE.exists():
            return pd.DataFrame()

        check_date = date or datetime.now().strftime("%Y-%m-%d")
        df = pd.read_csv(SURCHARGES_FILE)

        # Filter to active date window
        active = df[
            (df["effective_from"] <= check_date) &
            (df["effective_to"]   >= check_date)
        ].copy()

        return active.reset_index(drop=True)

    def get_surcharge_for_order(self, supplier: str, terminal: str,
                                 product: str, date: str) -> float:
        """
        Get the total surcharge amount for a specific order.
        Returns 0.0 if no surcharge applies.

        Called by the landed cost engine:
            landed_cost = base_price + get_surcharge_for_order(...) + freight
        """
        active = self.get_active_surcharges(date)
        if active.empty:
            return 0.0

        total = 0.0
        for _, row in active.iterrows():
            supplier_match = row["supplier"].lower() == supplier.lower()
            terminal_match = (
                row["terminal"].lower() in terminal.lower() or
                terminal.lower() in row["terminal"].lower()
            )
            product_match = (
                row["product"] == "All" or
                row["product"].lower() == product.lower()
            )
            if supplier_match and terminal_match and product_match:
                total += float(row["surcharge_per_l"])

        return total

    # ── Content reading ──────────────────────────────────────────────────

    def _read_content(self, filepath: Path) -> str:
        """Read raw text content from any file type."""
        suffix = filepath.suffix.lower()

        try:
            if suffix == ".pdf":
                import pdfminer.high_level
                return pdfminer.high_level.extract_text(str(filepath)) or ""

            elif suffix in (".xlsx", ".xls"):
                # Excel sent as a notice (rare but possible)
                import openpyxl
                wb = openpyxl.load_workbook(str(filepath), data_only=True)
                lines = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        line = " ".join(str(c) for c in row if c is not None)
                        if line.strip():
                            lines.append(line)
                return "\n".join(lines)

            else:
                # TXT, HTML, any other text format
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()

        except Exception as e:
            self.logger.error(
                "Could not read %s: %s", filepath.name, e
            )
            return ""

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_claude(self, content: str, supplier_hint: str,
                     source_label: str) -> str:
        """
        Send document content to Claude and get structured extraction.
        """
        prompt = self._build_prompt(content, supplier_hint, source_label)

        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
                return data["content"][0]["text"]
        except Exception as e:
            self.logger.error("Claude API call failed: %s", e)
            raise ValueError(f"LLM API call failed: {e}")

    def _build_prompt(self, content: str, supplier_hint: str,
                      source_label: str) -> str:
        """Build the extraction prompt for Claude."""
        hint_line = (
            f"The document appears to be from supplier: {supplier_hint}."
            if supplier_hint else
            "The supplier name should be extracted from the document."
        )

        return f"""You are a fuel procurement analyst extracting pricing events
from supplier communications.

{hint_line}
Document source: {source_label}

Read this document carefully and extract ALL surcharge or price change events.
A surcharge is any temporary addition to the base price caused by weather,
logistics, regulatory changes, seasonal factors, or any other reason.

Return a JSON array. Each element represents one surcharge event.
If no surcharge or price change is mentioned, return an empty array [].
Return ONLY the JSON — no explanation, no markdown, no backticks.

Each event must have EXACTLY these fields:
{{
  "supplier":        "supplier name (normalize to: Valero/Exxon/Marathon/Pemex)",
  "terminal":        "terminal or location name (exactly as written)",
  "product":         "Regular or Premium or Diesel or All",
  "surcharge_per_l": surcharge amount as a float in MXN per liter,
  "effective_from":  "YYYY-MM-DD",
  "effective_to":    "YYYY-MM-DD",
  "reason":          "brief reason for the surcharge (1 sentence)",
  "confidence":      "high if amounts and dates are explicit, medium if inferred, low if guessed"
}}

Rules:
- If a surcharge applies to multiple products, create one event per product
- If effective_to is not stated, estimate 7 days from effective_from
- surcharge_per_l must always be positive (it is added to base price)
- If the document describes a BASE PRICE CHANGE (not a surcharge), set
  surcharge_per_l to 0.0 and explain in reason
- If no relevant procurement info exists in the document, return []

Document to analyze:
{content[:4000]}"""

    # ── Response parsing ─────────────────────────────────────────────────

    def _parse_response(self, raw_text: str, source_file: str) -> list:
        """Parse Claude's JSON response into SurchargeEvent objects."""
        # Strip markdown if present
        text = re.sub(r"```json\s*|\s*```", "", raw_text).strip()

        # Find JSON array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            self.logger.warning(
                "No JSON array in LLM response for %s — "
                "no events extracted", source_file
            )
            return []

        try:
            raw_events = json.loads(match.group())
        except json.JSONDecodeError as e:
            self.logger.error(
                "Invalid JSON from LLM for %s: %s", source_file, e
            )
            return []

        events = []
        for raw in raw_events:
            try:
                event = SurchargeEvent(
                    supplier        = self._normalise_supplier(raw.get("supplier", "")),
                    terminal        = raw.get("terminal", "Unknown"),
                    product         = self._normalise_product(raw.get("product", "All")),
                    surcharge_per_l = float(raw.get("surcharge_per_l", 0.0)),
                    effective_from  = raw.get("effective_from", ""),
                    effective_to    = raw.get("effective_to", ""),
                    reason          = raw.get("reason", ""),
                    confidence      = raw.get("confidence", "low"),
                    source_file     = source_file,
                )

                # Basic validation
                if not event.effective_from or not event.effective_to:
                    self.logger.warning(
                        "Event missing dates — skipping: %s", raw
                    )
                    continue

                if event.surcharge_per_l < 0:
                    self.logger.warning(
                        "Negative surcharge — skipping: %s", raw
                    )
                    continue

                events.append(event)

            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(
                    "Could not parse event from LLM response: %s — %s",
                    raw, e
                )
                continue

        return events

    def _normalise_supplier(self, raw: str) -> str:
        """Normalise supplier name to standard form."""
        lower = raw.lower().strip()
        for alias, standard in SUPPLIER_ALIASES.items():
            if alias in lower:
                return standard
        return raw.strip().title()

    def _normalise_product(self, raw: str) -> str:
        """Normalise product to Regular/Premium/Diesel/All."""
        lower = raw.lower()
        if "regular" in lower or "87" in lower or "magna" in lower:
            return "Regular"
        elif "premium" in lower or "91" in lower:
            return "Premium"
        elif "diesel" in lower:
            return "Diesel"
        elif "all" in lower or "todos" in lower or "all products" in lower:
            return "All"
        return raw.strip().title()


# ---------------------------------------------------------------------------
# Convenience function — main entry point for the pipeline
# ---------------------------------------------------------------------------

def parse_unstructured(filepath: str,
                       supplier_hint: str = None,
                       save: bool = True) -> list:
    """
    Parse an unstructured document and optionally save surcharge events.

    This is the main function called by run_ingestion.py when it
    encounters a file that the GenericParser cannot handle.

    Args:
        filepath:      Path to any unstructured document
        supplier_hint: Supplier name if known
        save:          If True, save events to surcharges.csv

    Returns:
        List of SurchargeEvent objects
    """
    parser = UnstructuredParser()
    events = parser.parse(filepath, supplier_hint)

    if save and events:
        parser.save_events(events)

    return events


# ---------------------------------------------------------------------------
# CLI — test on any document
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python parse_llm.py <filepath> [supplier_name]")
        print()
        print("Examples:")
        print("  python parse_llm.py emails/valero_surcharge.txt Valero")
        print("  python parse_llm.py notices/weather_alert.pdf")
        sys.exit(1)

    filepath      = sys.argv[1]
    supplier_hint = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nParsing unstructured document: {filepath}")
    if supplier_hint:
        print(f"Supplier hint: {supplier_hint}")
    print("-" * 55)

    try:
        events = parse_unstructured(filepath, supplier_hint, save=True)

        if not events:
            print("\nNo procurement-relevant events found in this document.")
        else:
            print(f"\nExtracted {len(events)} surcharge event(s):\n")
            for i, event in enumerate(events, 1):
                print(f"  Event {i}:")
                print(f"    Supplier:   {event.supplier}")
                print(f"    Terminal:   {event.terminal}")
                print(f"    Product:    {event.product}")
                print(f"    Surcharge:  +{event.surcharge_per_l} MXN/L")
                print(f"    Effective:  {event.effective_from} → {event.effective_to}")
                print(f"    Reason:     {event.reason}")
                print(f"    Confidence: {event.confidence}")
                print()

            # Show what the landed cost impact looks like
            print("-" * 55)
            print("LANDED COST IMPACT:")
            print()
            parser_obj = UnstructuredParser()
            for event in events:
                if event.surcharge_per_l > 0:
                    example_base = 20.14
                    adjusted     = example_base + event.surcharge_per_l
                    print(f"  {event.supplier} {event.terminal} ({event.product})")
                    print(f"    Base price:      {example_base:.2f} MXN/L")
                    print(f"    + Surcharge:     {event.surcharge_per_l:.2f} MXN/L")
                    print(f"    = Adjusted cost: {adjusted:.2f} MXN/L")
                    print()

            print(f"Saved to: {SURCHARGES_FILE}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)