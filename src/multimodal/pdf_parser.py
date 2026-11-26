"""
Module 4 - Multimodal Ingestion: PDF TEXT and TEXT-TABLE extraction.

This module is responsible for the *textual* content of a research PDF:

  * `extract_text_from_pdf`   - plain page text (unchanged original behaviour).
  * `extract_tables_from_pdf` - text-based tables (rows and cells) detected with
                                PyMuPDF's built-in table finder.

Figures, embedded images and vector charts are handled by the companion module
`figure_extractor.py`. Keeping the two apart means each file has one clear job:
this file turns a PDF into text and table rows; the other turns it into images.

Tables that are actually scanned/screenshot images (no selectable text) cannot
be read here - `figure_extractor.detect_tables` records those separately as
needing OCR or manual review, so they are never silently lost.
"""

import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a research PDF.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extracted_pages = []

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text()

            extracted_pages.append({
                "page_number": page_number,
                "text": text
            })

    return extracted_pages


def _clean_cell(cell):
    """
    Tidy one table cell.

    PyMuPDF returns a cell as a string (or None for an empty cell) and often
    keeps the hard line breaks that existed in the PDF. We turn None into an
    empty string and collapse any internal whitespace into single spaces so the
    rows are easy to read and store.
    """

    if cell is None:
        return ""

    return " ".join(cell.split())


def _read_table_cells(page, table, page_words):
    """
    Read every cell of a detected table as clean text.

    Why not just call `table.extract()`? In the installed PyMuPDF version it
    assembles cell characters in the wrong order and splits decimal points
    off numbers - "0.91" comes back as "091 ." and "87.9%" as "879% .".
    That silently corrupts exactly the numbers the statistical critic needs.

    Instead we use the table finder only for the cell GEOMETRY (each cell's
    rectangle) and fill the cells ourselves from the page's word list:

      * `page_words` comes from ONE call to `page.get_text("words")`, the same
        extraction that produces the page's plain text, so a cell reads exactly
        like the page text does. (Reading each cell with `get_text(clip=...)`
        was tried and rejected: on the sample paper it leaks an invisible
        typesetting glyph, turning "104" into "104i".)
      * A word belongs to a cell when the word's CENTRE lies inside the cell,
        so a word that merely touches a cell border never leaks into the
        neighbouring cell.
      * Words keep their original reading order (block, line, word number).

    If the cell geometry is unavailable we fall back to `table.extract()`.

    Returns a list of rows, each a list of cell strings ("" for empty cells).
    """

    try:
        rows = []
        for row in table.rows:
            cells = []
            for cell_box in row.cells:
                if cell_box is None:
                    # Merged/empty cell - keep the column count intact.
                    cells.append("")
                    continue

                cell_rect = pymupdf.Rect(cell_box)
                inside = []
                for word in page_words:
                    # word = (x0, y0, x1, y1, text, block_no, line_no, word_no)
                    centre = pymupdf.Point((word[0] + word[2]) / 2,
                                           (word[1] + word[3]) / 2)
                    if cell_rect.contains(centre):
                        inside.append(word)

                inside.sort(key=lambda w: (w[5], w[6], w[7]))
                cells.append(_clean_cell(" ".join(w[4] for w in inside)))
            rows.append(cells)
        return rows

    except Exception:
        # Fallback: less accurate for decimals, but better than nothing.
        return [[_clean_cell(cell) for cell in row] for row in table.extract()]


def extract_tables_from_pdf(pdf_path):
    """
    Extract TEXT-BASED tables from a research PDF.

    Uses PyMuPDF's `page.find_tables()`, which locates tables by their ruling
    lines and text alignment and reads back the cell contents. This only works
    for tables whose numbers and labels are real, selectable text - a table
    stored as a flat image has no text to find and will simply not be detected
    here (see `figure_extractor.detect_tables` for how those are flagged).

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    list[dict]
        One dictionary per detected table, in page order:

            {
                "page_number": 5,              # 1-based
                "table_index": 0,              # 0-based, within that page
                "bbox": [325.0, 86.0, 549.0, 141.0],   # on-page box, points
                "row_count": 2,
                "col_count": 3,
                "rows": [["Dataset", "LaTeX", "XML"], ...],  # cleaned cells
            }

        An empty list means no text-based table was found, which is a normal
        result (the paper may have none, or its tables may be images).

    Raises
    ------
    FileNotFoundError
        If `pdf_path` does not exist or is not a file.
    ValueError
        If the file cannot be opened, or is not a PDF. (PyMuPDF will happily
        open a plain .txt file and "find" nonsense tables in it, so we refuse
        anything that is not a real PDF.)
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Not a file: {pdf_path}")

    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as error:
        raise ValueError(f"Could not open PDF: {pdf_path} ({error})") from error

    if not pdf.is_pdf:
        pdf.close()
        raise ValueError(f"File is not a PDF document: {pdf_path}")

    extracted_tables = []

    with pdf:

        for page_number, page in enumerate(pdf, start=1):

            # find_tables() can fail on an unusual page; one bad page should
            # never stop the whole document, so we guard it.
            try:
                found = page.find_tables()
            except Exception:
                continue

            # Only pay for word extraction on pages that have a table.
            page_words = page.get_text("words") if found.tables else []

            for table_index, table in enumerate(found.tables):

                clean_rows = _read_table_cells(page, table, page_words)

                extracted_tables.append({
                    "page_number": page_number,
                    "table_index": table_index,
                    "bbox": [round(value, 1) for value in table.bbox],
                    "row_count": table.row_count,
                    "col_count": table.col_count,
                    "rows": clean_rows,
                })

    return extracted_tables


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    pdf_path = project_root / "sample_pdfs" / "sample_paper1.pdf"

    # ----- 1: plain page text (original behaviour, unchanged) --------------
    pages = extract_text_from_pdf(pdf_path)

    output_path = project_root / "sample_pdfs" / "extracted_text.txt"

    with open(output_path, "w", encoding="utf-8") as file:

        for page in pages:
            file.write(f"\n--- Page {page['page_number']} ---\n")
            file.write(page["text"])
            file.write("\n")

    print(f"Text extraction completed!")
    print(f"Output saved to: {output_path}")

    # ----- 2: text-based tables (new) --------------------------------------
    tables = extract_tables_from_pdf(pdf_path)

    tables_path = project_root / "sample_pdfs" / "extracted_tables.txt"

    with open(tables_path, "w", encoding="utf-8") as file:

        for table in tables:
            file.write(f"\n--- Table on page {table['page_number']} "
                       f"(index {table['table_index']}, "
                       f"{table['row_count']}x{table['col_count']}) ---\n")
            for row in table["rows"]:
                file.write(" | ".join(row))
                file.write("\n")

    print()
    print(f"Table extraction completed!")
    print(f"Text-based tables found: {len(tables)}")
    for table in tables:
        print(f"  page {table['page_number']}: "
              f"{table['row_count']} rows x {table['col_count']} cols")
    print(f"Output saved to: {tables_path}")
