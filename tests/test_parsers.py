"""
tests/test_parsers.py
---------------------
Tests for all 4 supplier parsers.

For each parser we test:
    1. Normal format     — standard file parses correctly
    2. Modified format   — intentionally changed structure still parses
    3. Edge cases        — empty file, missing values, bad prices

Each test prints a clear PASS / FAIL with what was tested and why.

Run with:
    python tests/test_parsers.py
"""

import sys
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.parse_valero   import parse_valero_file,   parse_valero_folder,   validate_prices as val_vp
from src.ingestion.parse_exxon    import parse_exxon_file,    parse_exxon_folder,    validate_prices as val_ep
from src.ingestion.parse_marathon import parse_marathon_file, parse_marathon_folder, validate_prices as val_mp
from src.ingestion.parse_pemex    import parse_pemex_file,    parse_pemex_folder,    validate_prices as val_pp

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
VALERO_FILE = ROOT / "data/raw/valero/valero_20240101.html"
EXXON_FILE  = ROOT / "data/raw/exxon/exxon_20240101.xlsx"
MARATHON_FILE = ROOT / "data/raw/marathon/marathon_20240101_MX-IT-Chihuahua.txt"
PEMEX_FILE  = ROOT / "data/raw/pemex/reportes_pdf/pemex_tar_20240101_20240105.pdf"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_temp_dir():
    """Create a temporary directory for test files."""
    return tempfile.mkdtemp()


# ══════════════════════════════════════════════════════════════════════════════
#  VALERO TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestValeroParser(unittest.TestCase):

    # ── Test 1: Normal format ─────────────────────────────────────────────────
    def test_normal_file_parses_correctly(self):
        """Standard Valero HTML file should produce 29 rows across 9 terminals."""
        df = parse_valero_file(str(VALERO_FILE))
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertEqual(len(df), 29, f"Expected 29 rows, got {len(df)}")
        self.assertEqual(df["terminal_id"].nunique(), 9,
                         f"Expected 9 terminals, got {df['terminal_id'].nunique()}")
        self.assertTrue((df["price_mxn_per_l"] > 0).all(),
                        "All prices should be positive")
        print(f"  ✓ Normal format: {len(df)} rows, {df['terminal_id'].nunique()} terminals")

    # ── Test 2: Modified format — extra div wrapper around terminal cards ──────
    def test_extra_html_wrapper_ignored(self):
        """Parser should still find terminal-cards even if wrapped in extra divs."""
        tmpdir = make_temp_dir()
        try:
            original = VALERO_FILE.read_text(encoding="utf-8", errors="replace")

            # Wrap entire body in an extra div — simulates CMS/template change
            modified = original.replace(
                "<body>",
                "<body><div id='wrapper'><div class='container'>"
            ).replace(
                "</body>",
                "</div></div></body>"
            )

            test_file = Path(tmpdir) / "valero_20240101.html"
            test_file.write_text(modified, encoding="utf-8")

            df = parse_valero_file(str(test_file))
            self.assertFalse(df.empty, "Should still parse with extra wrapper divs")
            self.assertEqual(len(df), 29,
                             f"Expected 29 rows even with wrapper, got {len(df)}")
            print(f"  ✓ Extra HTML wrapper: still got {len(df)} rows")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 3: Modified format — new terminal added ──────────────────────────
    def test_new_terminal_card_picked_up(self):
        """If Valero adds a new terminal, parser should pick it up automatically."""
        tmpdir = make_temp_dir()
        try:
            original = VALERO_FILE.read_text(encoding="utf-8", errors="replace")

            # Inject a brand new terminal card at the end
            new_card = """
            <div class="terminal-card">
              <div class="card-header">
                <span>Tijuana, BCN, MX - TMX000099</span>
                <span>Save / Print</span>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Contract Type</th><th>Product Description</th>
                    <th>Effective Since</th><th>Current Price</th>
                    <th>Previous Price</th><th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Sin Marca</td><td>Diesel</td>
                    <td>01/01/2024 12:00:00 AM</td>
                    <td class="price">21.500000</td>
                    <td>21.500000</td><td class="change-neu">-------</td>
                  </tr>
                </tbody>
              </table>
            </div>
            """
            modified = original.replace("</body>", new_card + "</body>")
            test_file = Path(tmpdir) / "valero_20240101.html"
            test_file.write_text(modified, encoding="utf-8")

            df = parse_valero_file(str(test_file))
            self.assertEqual(df["terminal_id"].nunique(), 10,
                             f"Expected 10 terminals after adding one, got {df['terminal_id'].nunique()}")
            self.assertIn("TMX000099", df["terminal_id"].values,
                          "New terminal TMX000099 should be in results")
            print(f"  ✓ New terminal added: found {df['terminal_id'].nunique()} terminals including TMX000099")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 4: Modified format — price changed to 2 decimal places ───────────
    def test_price_with_fewer_decimals(self):
        """Parser should handle prices with 2 decimal places not just 6."""
        tmpdir = make_temp_dir()
        try:
            original = VALERO_FILE.read_text(encoding="utf-8", errors="replace")
            # Replace a known price with 2 decimal version
            modified = original.replace("20.987243", "20.99")
            test_file = Path(tmpdir) / "valero_20240101.html"
            test_file.write_text(modified, encoding="utf-8")

            df = parse_valero_file(str(test_file))
            self.assertFalse(df.empty, "Should parse even with 2 decimal prices")
            self.assertIn(20.99, df["price_mxn_per_l"].values,
                          "Price 20.99 should appear in results")
            print(f"  ✓ 2-decimal price: parsed correctly as 20.99")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 5: Edge case — file does not exist ───────────────────────────────
    def test_missing_file_returns_empty(self):
        """Non-existent file should return empty DataFrame not crash."""
        df = parse_valero_file("data/raw/valero/does_not_exist.html")
        self.assertTrue(df.empty, "Missing file should return empty DataFrame")
        print(f"  ✓ Missing file: returned empty DataFrame safely")

    # ── Test 6: Price guardrail flags anomalous prices ────────────────────────
    def test_price_guardrail_flags_bad_prices(self):
        """Prices below 15 or above 35 MXN/L should be flagged."""
        tmpdir = make_temp_dir()
        try:
            original = VALERO_FILE.read_text(encoding="utf-8", errors="replace")
            # Inject an obviously wrong price
            modified = original.replace("20.987243", "0.01")
            test_file = Path(tmpdir) / "valero_20240101.html"
            test_file.write_text(modified, encoding="utf-8")

            df = parse_valero_file(str(test_file))
            df = val_vp(df)
            flagged = df["price_flag"].sum()
            self.assertGreater(flagged, 0,
                               "Price of 0.01 MXN/L should be flagged")
            print(f"  ✓ Price guardrail: flagged {flagged} anomalous price(s)")
        finally:
            shutil.rmtree(tmpdir)


# ══════════════════════════════════════════════════════════════════════════════
#  EXXON TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestExxonParser(unittest.TestCase):

    # ── Test 1: Normal format ─────────────────────────────────────────────────
    def test_normal_file_parses_correctly(self):
        """Standard Exxon xlsx file should produce 9 rows across 3 terminals."""
        df = parse_exxon_file(str(EXXON_FILE))
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertEqual(len(df), 9, f"Expected 9 rows, got {len(df)}")
        self.assertEqual(df["terminal_id"].nunique(), 3,
                         f"Expected 3 terminals, got {df['terminal_id'].nunique()}")
        self.assertTrue((df["discount"] < 0).all(),
                        "All discounts should be negative")
        print(f"  ✓ Normal format: {len(df)} rows, {df['terminal_id'].nunique()} terminals")

    # ── Test 2: Modified format — columns reordered ───────────────────────────
    def test_column_reorder_still_parses(self):
        """Inserting a new column at position 0 should not break parsing."""
        import openpyxl
        tmpdir = make_temp_dir()
        try:
            test_file = Path(tmpdir) / "exxon_20240101.xlsx"
            shutil.copy(str(EXXON_FILE), str(test_file))

            # Insert new column at position 1 — shifts all existing columns right
            wb = openpyxl.load_workbook(str(test_file))
            ws = wb.active
            ws.insert_cols(1)
            ws["A1"] = "Fecha"
            ws["A3"] = "Fecha"
            for row_idx in range(4, 13):
                ws.cell(row=row_idx, column=1, value="01/01/2024")
            wb.save(str(test_file))

            df = parse_exxon_file(str(test_file))
            self.assertFalse(df.empty,
                             "Should still parse after inserting new column")
            self.assertEqual(len(df), 9,
                             f"Expected 9 rows after reorder, got {len(df)}")
            # Verify prices are still numbers not strings
            self.assertTrue((df["price_mxn_per_l"] > 15).all(),
                            "Prices should still be valid after column reorder")
            print(f"  ✓ Column reorder: {len(df)} rows parsed correctly after inserting new column")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 3: Modified format — extra blank rows in middle ──────────────────
    def test_extra_blank_rows_skipped(self):
        """Blank rows inserted between data rows should be skipped."""
        import openpyxl
        tmpdir = make_temp_dir()
        try:
            test_file = Path(tmpdir) / "exxon_20240101.xlsx"
            shutil.copy(str(EXXON_FILE), str(test_file))

            # Insert 2 blank rows in middle of data
            wb = openpyxl.load_workbook(str(test_file))
            ws = wb.active
            ws.insert_rows(6)
            ws.insert_rows(6)
            wb.save(str(test_file))

            df = parse_exxon_file(str(test_file))
            self.assertEqual(len(df), 9,
                             f"Expected 9 rows despite blank rows, got {len(df)}")
            print(f"  ✓ Extra blank rows: {len(df)} rows, blanks correctly skipped")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 4: Modified format — new terminal added ──────────────────────────
    def test_new_terminal_row_picked_up(self):
        """Adding a new terminal block should increase row count."""
        import openpyxl
        tmpdir = make_temp_dir()
        try:
            test_file = Path(tmpdir) / "exxon_20240101.xlsx"
            shutil.copy(str(EXXON_FILE), str(test_file))

            wb = openpyxl.load_workbook(str(test_file))
            ws = wb.active
            # Append 3 new rows for a 4th terminal
            new_rows = [
                ("0TZZ - NEW TERMINAL", "Wholesale", "Regular", 21.0, -0.50, 20.50),
                ("0TZZ - NEW TERMINAL", "Wholesale", "Premium", 22.0, -0.40, 21.60),
                ("0TZZ - NEW TERMINAL", "Wholesale", "Diesel",  23.0, -0.80, 22.20),
            ]
            for row_data in new_rows:
                ws.append(row_data)
            wb.save(str(test_file))

            df = parse_exxon_file(str(test_file))
            self.assertEqual(len(df), 12,
                             f"Expected 12 rows after adding terminal, got {len(df)}")
            self.assertEqual(df["terminal_id"].nunique(), 4,
                             f"Expected 4 terminals, got {df['terminal_id'].nunique()}")
            print(f"  ✓ New terminal: {len(df)} rows, {df['terminal_id'].nunique()} terminals found")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 5: Edge case — deleted row reduces count ────────────────────────
    def test_deleted_row_skipped(self):
        """Deleting a data row should reduce count by 1 not crash."""
        import openpyxl
        tmpdir = make_temp_dir()
        try:
            test_file = Path(tmpdir) / "exxon_20240101.xlsx"
            shutil.copy(str(EXXON_FILE), str(test_file))

            # Delete row 4 entirely — simulates supplier removing a product
            wb = openpyxl.load_workbook(str(test_file))
            ws = wb.active
            ws.delete_rows(4)
            wb.save(str(test_file))

            df = parse_exxon_file(str(test_file))
            self.assertEqual(len(df), 8,
                             f"Expected 8 rows after deleting one, got {len(df)}")
            print(f"  ✓ Deleted row: {len(df)} rows returned correctly")
        finally:
            shutil.rmtree(tmpdir)


# ══════════════════════════════════════════════════════════════════════════════
#  MARATHON TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMarathonParser(unittest.TestCase):

    # ── Test 1: Normal format ─────────────────────────────────────────────────
    def test_normal_file_parses_correctly(self):
        """Standard Marathon TXT file should produce 3 rows."""
        df = parse_marathon_file(str(MARATHON_FILE))
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertEqual(len(df), 3,
                         f"Expected 3 rows (one per product), got {len(df)}")
        self.assertIn("Regular", df["product_type"].values)
        self.assertIn("Premium", df["product_type"].values)
        self.assertIn("Diesel",  df["product_type"].values)
        print(f"  ✓ Normal format: {len(df)} rows, all 3 products found")

    # ── Test 2: Modified format — rows reordered inside email ─────────────────
    def test_price_rows_reordered(self):
        """Moving price rows to different positions should not affect parsing."""
        tmpdir = make_temp_dir()
        try:
            original = MARATHON_FILE.read_text(encoding="utf-8", errors="replace")
            lines = original.splitlines()

            # Find and swap Invoice Price and Base Price lines
            invoice_idx = next((i for i, l in enumerate(lines)
                                if l.strip().startswith("Invoice Price")), None)
            base_idx    = next((i for i, l in enumerate(lines)
                                if l.strip().startswith("Base Price")), None)

            if invoice_idx and base_idx:
                lines[invoice_idx], lines[base_idx] = lines[base_idx], lines[invoice_idx]

            test_file = Path(tmpdir) / "marathon_20240101_MX-IT-Chihuahua.txt"
            test_file.write_text("\n".join(lines), encoding="utf-8")

            df = parse_marathon_file(str(test_file))
            # Invoice price should still be found correctly by label not position
            self.assertEqual(len(df), 3,
                             f"Expected 3 rows after row reorder, got {len(df)}")
            self.assertTrue((df["price_mxn_per_l"] > 15).all(),
                            "Invoice prices should still be valid after reorder")
            print(f"  ✓ Row reorder: invoice price still found correctly by label")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 3: Modified format — extra lines added to email header ────────────
    def test_extra_email_header_lines_ignored(self):
        """Additional email metadata lines should not affect price parsing."""
        tmpdir = make_temp_dir()
        try:
            original = MARATHON_FILE.read_text(encoding="utf-8", errors="replace")

            # Add extra email header fields at the top
            extra_headers = (
                "X-Spam-Score: 0.0\r\n"
                "X-Priority: 1\r\n"
                "Reply-To: pricing@marathon.com\r\n"
            )
            modified = extra_headers + original

            test_file = Path(tmpdir) / "marathon_20240101_MX-IT-Chihuahua.txt"
            test_file.write_text(modified, encoding="utf-8")

            df = parse_marathon_file(str(test_file))
            self.assertEqual(len(df), 3,
                             f"Expected 3 rows with extra headers, got {len(df)}")
            print(f"  ✓ Extra email headers: {len(df)} rows, extra lines ignored")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 4: Modified format — US terminal filename ────────────────────────
    def test_us_terminal_filename_parsed(self):
        """US terminal filenames should set country to US."""
        tmpdir = make_temp_dir()
        try:
            original = MARATHON_FILE.read_text(encoding="utf-8", errors="replace")
            # Copy file with US terminal filename
            test_file = Path(tmpdir) / "marathon_20240101_US-IT-El_Paso.txt"
            test_file.write_text(original, encoding="utf-8")

            df = parse_marathon_file(str(test_file))
            self.assertEqual(df["country"].iloc[0], "US",
                             "Country should be US for US-IT-* terminals")
            self.assertEqual(df["terminal_name"].iloc[0], "El Paso",
                             "Terminal name should be El Paso")
            print(f"  ✓ US terminal: country=US, terminal_name=El Paso")
        finally:
            shutil.rmtree(tmpdir)

    # ── Test 5: Edge case — missing invoice price block ───────────────────────
    def test_missing_invoice_price_returns_empty(self):
        """File with no Invoice Price line should return empty DataFrame."""
        tmpdir = make_temp_dir()
        try:
            original = MARATHON_FILE.read_text(encoding="utf-8", errors="replace")
            # Remove Invoice Price line entirely
            modified = "\n".join(
                l for l in original.splitlines()
                if not l.strip().startswith("Invoice Price")
            )
            test_file = Path(tmpdir) / "marathon_20240101_MX-IT-Chihuahua.txt"
            test_file.write_text(modified, encoding="utf-8")

            df = parse_marathon_file(str(test_file))
            self.assertTrue(df.empty,
                            "No Invoice Price line should return empty DataFrame")
            print(f"  ✓ Missing invoice price: returned empty DataFrame safely")
        finally:
            shutil.rmtree(tmpdir)


# ══════════════════════════════════════════════════════════════════════════════
#  PEMEX TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPemexParser(unittest.TestCase):

    # ── Test 1: Normal format ─────────────────────────────────────────────────
    def test_normal_file_parses_correctly(self):
        """Standard Pemex PDF should produce 234 rows across 78 regions."""
        df = parse_pemex_file(str(PEMEX_FILE))
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertEqual(len(df), 234,
                         f"Expected 234 rows, got {len(df)}")
        self.assertEqual(df["terminal_id"].nunique(), 78,
                         f"Expected 78 regions, got {df['terminal_id'].nunique()}")
        print(f"  ✓ Normal format: {len(df)} rows, {df['terminal_id'].nunique()} regions")

    # ── Test 2: Price conversion MXN/m³ → MXN/L ──────────────────────────────
    def test_price_converted_from_m3_to_liter(self):
        """Prices should be divided by 1000 — original ~19000, converted ~19.0."""
        df = parse_pemex_file(str(PEMEX_FILE))
        df = val_pp(df)

        # All converted prices should be in realistic MXN/L range
        self.assertTrue((df["price_mxn_per_l"] < 35).all(),
                        "Converted prices should be under 35 MXN/L")
        self.assertTrue((df["price_mxn_per_l"] > 15).all(),
                        "Converted prices should be over 15 MXN/L")

        # Original m³ prices should be ~1000x larger
        self.assertTrue((df["price_mxn_per_m3"] > 15000).all(),
                        "Original m³ prices should be over 15,000")
        self.assertEqual(df["price_flag"].sum(), 0,
                         "No prices should be flagged after correct conversion")

        sample = df[df["terminal_id"] == "CHIHUAHUA"].iloc[0]
        print(f"  ✓ Price conversion: Chihuahua Regular = "
              f"{sample['price_mxn_per_m3']:,.4f} MXN/m³ → "
              f"{sample['price_mxn_per_l']:.4f} MXN/L")

    # ── Test 3: All 3 product types detected ─────────────────────────────────
    def test_all_product_types_detected(self):
        """Should detect Regular, Premium and Diesel — 78 rows each."""
        df = parse_pemex_file(str(PEMEX_FILE))
        counts = df["product_type"].value_counts()
        self.assertEqual(counts.get("Regular", 0), 78,
                         f"Expected 78 Regular rows, got {counts.get('Regular', 0)}")
        self.assertEqual(counts.get("Premium", 0), 78,
                         f"Expected 78 Premium rows, got {counts.get('Premium', 0)}")
        self.assertEqual(counts.get("Diesel", 0), 78,
                         f"Expected 78 Diesel rows, got {counts.get('Diesel', 0)}")
        print(f"  ✓ Product types: Regular=78, Premium=78, Diesel=78")

    # ── Test 4: Date range parsed from Spanish text ───────────────────────────
    def test_spanish_date_range_parsed(self):
        """Date 'del 1 al 5 de enero de 2024' should parse to 2024-01-01."""
        df = parse_pemex_file(str(PEMEX_FILE))
        self.assertEqual(str(df["date"].iloc[0].date()), "2024-01-01",
                         "Start date should be 2024-01-01")
        self.assertEqual(str(df["date_end"].iloc[0]), "2024-01-05",
                         "End date should be 2024-01-05")
        print(f"  ✓ Spanish date: 'del 1 al 5 de enero de 2024' → "
              f"{df['date'].iloc[0].date()} to {df['date_end'].iloc[0]}")

    # ── Test 5: Edge case — missing file ─────────────────────────────────────
    def test_missing_file_returns_empty(self):
        """Non-existent PDF should return empty DataFrame not crash."""
        df = parse_pemex_file("data/raw/pemex/reportes_pdf/does_not_exist.pdf")
        self.assertTrue(df.empty, "Missing file should return empty DataFrame")
        print(f"  ✓ Missing file: returned empty DataFrame safely")


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests():
    """Run all tests with clean formatted output."""

    test_classes = [
        ("VALERO  (HTML)",    TestValeroParser),
        ("EXXON   (Excel)",   TestExxonParser),
        ("MARATHON (TXT)",    TestMarathonParser),
        ("PEMEX   (PDF)",     TestPemexParser),
    ]

    total_passed = 0
    total_failed = 0
    total_errors = 0

    for label, cls in test_classes:
        print()
        print(f"{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        suite  = unittest.TestLoader().loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(
            verbosity=0,
            stream=open(os.devnull, "w")
        )
        result = runner.run(suite)

        # Re-run with our custom print output
        for test in unittest.TestLoader().loadTestsFromTestCase(cls):
            if test is None:
                continue
            try:
                test.debug()
                total_passed += 1
            except AssertionError as e:
                name = getattr(test, '_testMethodName', str(test))
                print(f"  ✗ {name}: FAILED — {e}")
                total_failed += 1
            except Exception as e:
                name = getattr(test, '_testMethodName', str(test))
                print(f"  ✗ {name}: ERROR — {e}")
                total_errors += 1

    print()
    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Passed : {total_passed}")
    print(f"  Failed : {total_failed}")
    print(f"  Errors : {total_errors}")
    print(f"  Total  : {total_passed + total_failed + total_errors}")
    print(f"{'='*60}")
    print()

    if total_failed == 0 and total_errors == 0:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED — check output above")
    print()


if __name__ == "__main__":
    run_tests()