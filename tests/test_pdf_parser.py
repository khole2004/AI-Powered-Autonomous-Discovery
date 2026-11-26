"""
Unit and integration tests for src/multimodal/pdf_parser.py.

Run from the project root:

    python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import pdf_fixtures as fx  # also puts src/ on sys.path

from multimodal.pdf_parser import extract_tables_from_pdf, extract_text_from_pdf


class TestExtractTextSynthetic(unittest.TestCase):
    """Plain text extraction on PDFs whose text we wrote ourselves."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_record_per_page_with_1_based_numbers(self):
        pdf = fx.make_text_pdf(self.dir / "t.pdf", ["alpha page", "beta page"])
        pages = extract_text_from_pdf(pdf)
        self.assertEqual([p["page_number"] for p in pages], [1, 2])

    def test_text_lands_on_the_right_page(self):
        pdf = fx.make_text_pdf(self.dir / "t.pdf", ["alpha page", "beta page"])
        pages = extract_text_from_pdf(pdf)
        self.assertIn("alpha", pages[0]["text"])
        self.assertIn("beta", pages[1]["text"])
        self.assertNotIn("beta", pages[0]["text"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf(self.dir / "does_not_exist.pdf")


class TestExtractTablesSynthetic(unittest.TestCase):
    """Text-table extraction against a ruled table with known cell values."""

    DATA = [["Model", "Acc", "F1"],
            ["A", "0.91", "87.9%"],
            ["B", "0.87", "1,030,671"]]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cells_match_exactly(self):
        pdf = fx.make_grid_table_pdf(self.dir / "g.pdf", self.DATA)
        tables = extract_tables_from_pdf(pdf)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["rows"], self.DATA)

    def test_decimals_and_percentages_survive(self):
        # Regression: PyMuPDF's table.extract() returned "091 ." for "0.91"
        # and "879% ." for "87.9%". The numbers must come back intact.
        pdf = fx.make_grid_table_pdf(self.dir / "g.pdf", self.DATA)
        cells = [c for row in extract_tables_from_pdf(pdf)[0]["rows"] for c in row]
        for value in ("0.91", "0.87", "87.9%", "1,030,671"):
            self.assertIn(value, cells)

    def test_metadata_fields(self):
        pdf = fx.make_grid_table_pdf(self.dir / "g.pdf", self.DATA)
        table = extract_tables_from_pdf(pdf)[0]
        self.assertEqual(table["page_number"], 1)
        self.assertEqual(table["table_index"], 0)
        self.assertEqual((table["row_count"], table["col_count"]), (3, 3))
        self.assertEqual(len(table["bbox"]), 4)
        x0, y0, x1, y1 = table["bbox"]
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)

    def test_pdf_without_tables_returns_empty_list(self):
        self.assertEqual(extract_tables_from_pdf(fx.make_blank_pdf(self.dir / "b.pdf")), [])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            extract_tables_from_pdf(self.dir / "nope.pdf")

    def test_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            extract_tables_from_pdf(self.dir)

    def test_non_pdf_raises_value_error(self):
        # Regression: a .txt file used to be opened and "found" to contain a
        # 28-row garbage table.
        with self.assertRaises(ValueError):
            extract_tables_from_pdf(fx.make_non_pdf(self.dir / "notes.txt"))


@unittest.skipUnless(fx.SAMPLE_PDF.exists(), "sample_paper1.pdf not available")
class TestSamplePaper(unittest.TestCase):
    """Integration checks on the real sample paper (values verified by eye)."""

    @classmethod
    def setUpClass(cls):
        cls.pages = extract_text_from_pdf(fx.SAMPLE_PDF)
        cls.tables = extract_tables_from_pdf(fx.SAMPLE_PDF)

    def test_page_count_and_numbering(self):
        self.assertEqual([p["page_number"] for p in self.pages], list(range(1, 11)))

    def test_captions_appear_on_expected_pages(self):
        self.assertIn("Figure 1:", self.pages[3]["text"])   # page 4
        self.assertIn("Table 2:", self.pages[5]["text"])    # page 6

    def test_three_text_tables_on_pages_5_6_8(self):
        self.assertEqual([t["page_number"] for t in self.tables], [5, 6, 8])
        for table in self.tables:
            self.assertEqual(table["col_count"], 3)

    def test_table_numbers_are_not_corrupted(self):
        def cells(page):
            table = next(t for t in self.tables if t["page_number"] == page)
            return " ".join(c for row in table["rows"] for c in row)

        self.assertIn("0.95", cells(6))
        self.assertIn("87.9%", cells(8))
        self.assertIn("PDFFigures 2.0 [7]", cells(8))
        self.assertIn("242,041", cells(5))
        # Regression: a hidden typesetting glyph used to turn "104" into "104i".
        self.assertIn("104", cells(5).split())
        self.assertNotIn("104i", cells(5))


if __name__ == "__main__":
    unittest.main()
