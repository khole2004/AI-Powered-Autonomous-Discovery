"""
Unit and integration tests for src/multimodal/figure_extractor.py.

Run from the project root:

    python -m unittest discover -s tests -v

All output files go to temporary folders, so running the tests never touches
the real sample_pdfs/ output folders.
"""

import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pdf_fixtures as fx  # also puts src/ on sys.path

from multimodal import figure_extractor as fe


def quiet(func, *args, **kwargs):
    """Call `func` with its [note]/[warning] console output suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


class TempDirTestCase(unittest.TestCase):
    """Gives every test a fresh temporary folder in `self.dir`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# Caption detection
# ---------------------------------------------------------------------------

class TestCaptionPattern(unittest.TestCase):

    def test_accepts_real_caption_styles(self):
        for text in ("Figure 3: Accuracy", "Fig. 3. Accuracy", "FIG. 5: RG",
                     "Table 2: Precision", "Figure 1a: Left panel",
                     "Figure 12. Long caption"):
            with self.subTest(text=text):
                self.assertIsNotNone(fe.CAPTION_PATTERN.match(text))

    def test_rejects_in_text_references(self):
        # Regression: these used to be treated as captions.
        for text in ("Figure 3 shows that accuracy improves",
                     "Table 2 lists the results",
                     "Figures are common in papers"):
            with self.subTest(text=text):
                self.assertIsNone(fe.CAPTION_PATTERN.match(text))


# ---------------------------------------------------------------------------
# Embedded images and duplicate placements
# ---------------------------------------------------------------------------

class TestEmbeddedImages(TempDirTestCase):

    def setUp(self):
        super().setUp()
        pdf = fx.make_duplicate_images_pdf(self.dir / "d.pdf", self.dir)
        self.out = self.dir / "images"
        self.records = quiet(fe.extract_embedded_images, pdf, output_dir=self.out)

    def test_every_placement_is_recorded_and_tiny_image_skipped(self):
        # 2x A on page 1 + A and B on page 2; the 20x20 image is filtered out.
        self.assertEqual(len(self.records), 4)
        self.assertEqual([r["page_number"] for r in self.records], [1, 1, 2, 2])
        self.assertTrue(all(r["width"] >= 50 for r in self.records))

    def test_duplicates_point_at_the_first_copy(self):
        originals = [r for r in self.records if r["duplicate_of"] is None]
        duplicates = [r for r in self.records if r["duplicate_of"] is not None]
        self.assertEqual(len(originals), 2)      # A and B
        self.assertEqual(len(duplicates), 2)     # the two extra A placements
        first_a = self.records[0]["file_name"]
        for dup in duplicates:
            self.assertEqual(dup["duplicate_of"], first_a)
            self.assertEqual(Path(dup["file_path"]).name, first_a)

    def test_only_unique_bytes_are_written(self):
        written = sorted(p.name for p in self.out.iterdir())
        expected = sorted(r["file_name"] for r in self.records
                          if r["duplicate_of"] is None)
        self.assertEqual(written, expected)
        for record in self.records:
            self.assertTrue(Path(record["file_path"]).exists())

    def test_filenames_are_unique_per_placement(self):
        names = [r["file_name"] for r in self.records]
        self.assertEqual(len(names), len(set(names)))

    def test_metadata_is_complete_and_consistent(self):
        expected_keys = {
            "page_number", "placement_index", "file_name", "file_path",
            "width", "height", "format", "colorspace", "bits_per_component",
            "size_bytes", "bbox", "duplicate_of", "classification",
            "fragment_of",
        }
        for record in self.records:
            self.assertEqual(set(record), expected_keys)
            self.assertEqual(len(record["bbox"]), 4)
            self.assertEqual(record["classification"], "standalone_image")
            self.assertIsNone(record["fragment_of"])

    def test_dimensions_are_the_stored_pixel_size(self):
        sizes = {(r["width"], r["height"]) for r in self.records}
        self.assertEqual(sizes, {(200, 150), (180, 120)})

    def test_pdf_without_images_returns_empty_list(self):
        blank = fx.make_blank_pdf(self.dir / "b.pdf")
        self.assertEqual(quiet(fe.extract_embedded_images, blank,
                               output_dir=self.dir / "x"), [])


# ---------------------------------------------------------------------------
# Figure-region cropping
# ---------------------------------------------------------------------------

class TestFigureRegions(TempDirTestCase):

    def test_vector_only_figure_is_cropped(self):
        pdf = fx.make_vector_figure_pdf(self.dir / "v.pdf")
        regions = quiet(fe.extract_figure_regions, pdf, output_dir=self.dir / "r")

        # One real caption; the "Figure 1 shows ..." paragraph is not one.
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region["label"], "Figure 1")
        self.assertEqual(region["figure_number"], 1)
        self.assertEqual(region["page_number"], 1)
        self.assertEqual(region["region_confidence"], "detected")
        self.assertEqual(region["extraction_method"], "region_crop")
        self.assertTrue(region["caption"].startswith("Figure 1: A vector bar chart"))

        out = Path(region["file_path"])
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 0)

    def test_region_sits_between_paragraph_and_caption(self):
        pdf = fx.make_vector_figure_pdf(self.dir / "v.pdf")
        x0, y0, x1, y1 = quiet(fe.extract_figure_regions, pdf,
                               output_dir=self.dir / "r")[0]["bbox"]
        # Chart is drawn at y 160-300; paragraph ends ~140; caption at ~310.
        self.assertGreaterEqual(y0, 140)
        self.assertLessEqual(y0, 160)
        self.assertGreaterEqual(y1, 300)
        self.assertLessEqual(y1, 312)

    def test_same_label_twice_on_a_page_does_not_overwrite(self):
        pdf = fx.make_two_same_label_figures_pdf(self.dir / "two.pdf")
        regions = quiet(fe.extract_figure_regions, pdf, output_dir=self.dir / "r")
        self.assertEqual(len(regions), 2)
        names = [r["file_name"] for r in regions]
        self.assertEqual(len(set(names)), 2)
        for region in regions:
            self.assertTrue(Path(region["file_path"]).exists())

    def test_tables_are_not_cropped_as_figures(self):
        pdf = fx.make_grid_table_pdf(self.dir / "g.pdf", [["a", "b"], ["1", "2"]])
        self.assertEqual(quiet(fe.extract_figure_regions, pdf,
                               output_dir=self.dir / "r"), [])

    def test_pdf_without_captions_returns_empty_list(self):
        blank = fx.make_blank_pdf(self.dir / "b.pdf")
        self.assertEqual(quiet(fe.extract_figure_regions, blank,
                               output_dir=self.dir / "r"), [])


# ---------------------------------------------------------------------------
# Table triage
# ---------------------------------------------------------------------------

class TestTableTriage(TempDirTestCase):

    def test_ruled_text_table_is_routed_to_pdf_parser(self):
        pdf = fx.make_grid_table_pdf(self.dir / "g.pdf", [["a", "b"], ["1", "2"]])
        review = self.dir / "review"
        notes = quiet(fe.detect_tables, pdf, review_dir=review)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["status"], "text_based")
        self.assertEqual(notes[0]["handled_by"], "pdf_parser")
        self.assertIsNone(notes[0]["file_path"])
        self.assertFalse(review.exists(), "review folder created needlessly")

    def test_image_table_is_flagged_for_ocr_and_saved(self):
        pdf = fx.make_image_table_pdf(self.dir / "it.pdf", self.dir)
        notes = quiet(fe.detect_tables, pdf, review_dir=self.dir / "review")
        self.assertEqual(len(notes), 1)
        note = notes[0]
        self.assertEqual(note["status"], "image_based_requires_ocr")
        self.assertEqual(note["handled_by"], "ocr_or_manual_review")
        self.assertTrue(Path(note["file_path"]).exists())

    def test_caption_without_table_is_kept_for_manual_review(self):
        pdf = fx.make_caption_only_table_pdf(self.dir / "c.pdf")
        notes = quiet(fe.detect_tables, pdf, review_dir=self.dir / "review")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["status"], "unresolved_manual_review")
        self.assertEqual(notes[0]["label"], "Table 2")
        self.assertEqual(notes[0]["table_number"], 2)


# ---------------------------------------------------------------------------
# Error handling, shared by every public function
# ---------------------------------------------------------------------------

class TestInvalidInputs(TempDirTestCase):

    FUNCTIONS = ("extract_embedded_images", "extract_figure_regions",
                 "detect_tables", "extract_all", "extract_figures_from_pdf")

    def test_missing_file(self):
        for name in self.FUNCTIONS:
            with self.subTest(function=name):
                with self.assertRaises(FileNotFoundError):
                    getattr(fe, name)(self.dir / "missing.pdf")

    def test_directory_instead_of_file(self):
        for name in self.FUNCTIONS:
            with self.subTest(function=name):
                with self.assertRaises(FileNotFoundError):
                    getattr(fe, name)(self.dir)

    def test_non_pdf_file(self):
        txt = fx.make_non_pdf(self.dir / "notes.txt")
        for name in self.FUNCTIONS:
            with self.subTest(function=name):
                with self.assertRaises(ValueError):
                    getattr(fe, name)(txt)


# ---------------------------------------------------------------------------
# Module boundaries
# ---------------------------------------------------------------------------

class TestModulesStayDecoupled(unittest.TestCase):
    """The team decided pdf_parser and figure_extractor must not import each other."""

    @staticmethod
    def imported_modules(path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    def test_figure_extractor_does_not_import_pdf_parser(self):
        names = self.imported_modules(fx.SRC_DIR / "multimodal" / "figure_extractor.py")
        self.assertFalse(any("pdf_parser" in n for n in names), names)

    def test_pdf_parser_does_not_import_figure_extractor(self):
        names = self.imported_modules(fx.SRC_DIR / "multimodal" / "pdf_parser.py")
        self.assertFalse(any("figure_extractor" in n for n in names), names)


# ---------------------------------------------------------------------------
# Integration on the real sample paper
# ---------------------------------------------------------------------------

@unittest.skipUnless(fx.SAMPLE_PDF.exists(), "sample_paper1.pdf not available")
class TestSamplePaper(unittest.TestCase):
    """Expected values below were verified by viewing the extracted images."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.image_dir = base / "images"
        cls.result = quiet(fe.extract_all, fx.SAMPLE_PDF,
                           image_dir=cls.image_dir,
                           region_dir=base / "regions",
                           review_dir=base / "review")
        cls.review_dir = base / "review"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_result_keys(self):
        self.assertEqual(set(self.result),
                         {"embedded_images", "figure_regions", "tables", "pdf_path"})

    def test_embedded_image_placements(self):
        images = self.result["embedded_images"]
        self.assertEqual(len(images), 7)
        self.assertEqual([i["page_number"] for i in images], [4, 4, 4, 7, 7, 7, 8])
        self.assertEqual(len(list(self.image_dir.iterdir())), 5)

    def test_repeated_page_thumbnail_is_kept_as_duplicates(self):
        by_name = {i["file_name"]: i for i in self.result["embedded_images"]}
        # Drawn twice on page 7 and reused on page 8 (verified byte-identical).
        for name in ("page07_img02.jpeg", "page08_img01.jpeg"):
            self.assertEqual(by_name[name]["duplicate_of"], "page07_img01.jpeg")
        self.assertIsNone(by_name["page07_img01.jpeg"]["duplicate_of"])

    def test_every_image_is_a_fragment_of_its_figure(self):
        expected = {4: "page04_figure1.png", 7: "page07_figure3.png",
                    8: "page08_figure4.png"}
        for image in self.result["embedded_images"]:
            with self.subTest(image=image["file_name"]):
                self.assertEqual(image["classification"], "image_fragment")
                self.assertEqual(image["fragment_of"], expected[image["page_number"]])

    def test_five_figures_on_expected_pages(self):
        regions = self.result["figure_regions"]
        self.assertEqual([(r["label"], r["page_number"]) for r in regions],
                         [("Figure 1", 4), ("Figure 2", 6), ("Figure 3", 7),
                          ("Figure 4", 8), ("Figure 5", 9)])
        for region in regions:
            with self.subTest(figure=region["label"]):
                self.assertEqual(region["region_confidence"], "detected")
                self.assertTrue(Path(region["file_path"]).exists())
                self.assertTrue(region["caption"].startswith(region["label"] + ":"))

    def test_tall_figure_4_is_not_truncated(self):
        # Regression: Figure 4 (a full-column diagram, y ~84-532) was once cut
        # down to y 476-538 by the neighbouring column's paragraphs.
        fig4 = next(r for r in self.result["figure_regions"] if r["label"] == "Figure 4")
        _, y0, _, y1 = fig4["bbox"]
        self.assertLessEqual(y0, 90)
        self.assertGreaterEqual(y1, 530)

    def test_three_text_tables_triaged_to_pdf_parser(self):
        tables = self.result["tables"]
        self.assertEqual([(t["label"], t["page_number"]) for t in tables],
                         [("Table 1", 5), ("Table 2", 6), ("Table 3", 8)])
        self.assertTrue(all(t["status"] == "text_based" for t in tables))
        self.assertFalse(self.review_dir.exists())


if __name__ == "__main__":
    unittest.main()
