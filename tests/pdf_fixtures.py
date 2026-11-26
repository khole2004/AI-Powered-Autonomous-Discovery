"""
Shared helpers for the Module 4 ingestion tests.

Each `make_*` function writes a small synthetic PDF whose contents we choose
ourselves, so the tests can assert exact expected values instead of trusting
whatever the extractor happens to return. The real sample paper is used for
integration tests on top of these.

This file is not a test module itself (its name does not start with "test"),
so `unittest discover` imports it only when a test file does.
"""

import sys
from pathlib import Path

import pymupdf

# Make `src/` importable as the package root, the same way the modules are
# laid out in the project: `from multimodal.pdf_parser import ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SAMPLE_PDF = PROJECT_ROOT / "sample_pdfs" / "sample_paper1.pdf"

# A paragraph long enough (>= 150 chars) to count as body text, i.e. a hard
# boundary that a figure region must not cross.
LONG_PARAGRAPH = (
    "This is ordinary body text describing the experiment in detail. It is "
    "deliberately long so that the extractor treats it as a paragraph of prose "
    "rather than a short label inside a figure."
)


def make_png(path, width, height, rgb):
    """Write a solid-colour PNG. Different colours give different bytes."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, rgb)
    pix.save(str(path))
    return Path(path)


def make_text_pdf(path, pages):
    """One page per string in `pages`."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 400), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_blank_pdf(path):
    """A single page with a few words and nothing else."""
    return make_text_pdf(path, ["Nothing to extract here."])


def make_grid_table_pdf(path, data, caption="Table 1: Synthetic results."):
    """
    A ruled table (every cell boxed) with known cell values and a caption
    above it. `data` is a list of rows, each a list of strings.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    if caption:
        page.insert_text((72, 90), caption, fontsize=10)
    x0, y0, col_w, row_h = 72, 100, 120, 20
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            cell = pymupdf.Rect(x0 + c * col_w, y0 + r * row_h,
                                x0 + (c + 1) * col_w, y0 + (r + 1) * row_h)
            page.draw_rect(cell, color=(0, 0, 0), width=0.8)
            page.insert_text((cell.x0 + 4, cell.y1 - 6), value, fontsize=10)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_duplicate_images_pdf(path, work_dir):
    """
    Two pages exercising duplicate handling:

      page 1: image A drawn twice, plus a tiny 20x20 image (should be filtered)
      page 2: image A again, plus a different image B

    Expected: 4 kept placements, 2 unique files (A and B), and the 2 extra
    placements of A marked `duplicate_of` the first copy of A.
    """
    work_dir = Path(work_dir)
    img_a = make_png(work_dir / "a.png", 200, 150, (200, 30, 30))
    img_b = make_png(work_dir / "b.png", 180, 120, (30, 30, 200))
    tiny = make_png(work_dir / "tiny.png", 20, 20, (30, 200, 30))

    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_image(pymupdf.Rect(72, 72, 272, 222), filename=str(img_a))
    page1.insert_image(pymupdf.Rect(320, 72, 520, 222), filename=str(img_a))
    page1.insert_image(pymupdf.Rect(72, 300, 92, 320), filename=str(tiny))
    page2 = doc.new_page()
    page2.insert_image(pymupdf.Rect(72, 72, 272, 222), filename=str(img_a))
    page2.insert_image(pymupdf.Rect(72, 300, 252, 420), filename=str(img_b))
    doc.save(str(path))
    doc.close()
    return Path(path)


def _draw_bar_chart(page, left, top, width, height):
    """Draw a small vector-only bar chart (axes + bars), no bitmap at all."""
    bottom = top + height
    page.draw_line((left, top), (left, bottom), color=(0, 0, 0), width=1)
    page.draw_line((left, bottom), (left + width, bottom), color=(0, 0, 0), width=1)
    bar_w = width / 7
    for i, frac in enumerate([0.4, 0.8, 0.6]):
        x = left + bar_w * (1 + 2 * i)
        page.draw_rect(pymupdf.Rect(x, bottom - height * frac, x + bar_w, bottom),
                       color=(0, 0, 0), fill=(0.2, 0.4, 0.8))


def make_vector_figure_pdf(path):
    """
    Left-column layout:

        y  60-140  body paragraph (a boundary the figure must not cross)
        y 160-300  vector bar chart (no embedded image)
        y 310      caption "Figure 1: ..."
        y 340+     a paragraph that only *mentions* "Figure 1 shows ..."

    Expected: exactly one figure region, confidence "detected", lying between
    the top paragraph and the caption; the in-text mention is NOT a caption.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 60, 290, 140), LONG_PARAGRAPH, fontsize=9)
    _draw_bar_chart(page, 90, 160, 180, 140)
    page.insert_text((72, 318), "Figure 1: A vector bar chart.", fontsize=9)
    page.insert_textbox(
        pymupdf.Rect(72, 340, 290, 460),
        "Figure 1 shows that the middle bar is tallest, which is the result we "
        "expected from the synthetic data used in this test document here.",
        fontsize=9)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_two_same_label_figures_pdf(path):
    """
    Two vector charts on one page, both captioned "Figure 1:" (as happens with
    a figure continued). Each crop must get its own file - no overwriting.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    _draw_bar_chart(page, 90, 80, 180, 120)
    page.insert_text((72, 218), "Figure 1: First part.", fontsize=9)
    _draw_bar_chart(page, 90, 260, 180, 120)
    page.insert_text((72, 398), "Figure 1: Continued.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_image_table_pdf(path, work_dir):
    """A 'table' that is only a picture (no selectable cells) + its caption."""
    picture = make_png(Path(work_dir) / "table_picture.png", 400, 200, (220, 220, 220))
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 100, 300, 200), filename=str(picture))
    page.insert_text((72, 215), "Table 1: Results as a scanned image.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_caption_only_table_pdf(path):
    """A table caption with neither a ruled table nor an image near it."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Table 2: Caption with no table.", fontsize=9)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_non_pdf(path):
    """A plain text file - which PyMuPDF will happily open if not guarded."""
    Path(path).write_text("Table 1: not really a PDF\n1 2 3\n4 5 6\n",
                          encoding="utf-8")
    return Path(path)
