"""
Module 4 - Multimodal Ingestion: PDF figure and image extraction.

This module offers TWO complementary ways to pull visual content out of a
research PDF, because neither alone is enough for real papers:

  1. Embedded-image extraction (`extract_embedded_images`)
     ------------------------------------------------------
     Pulls the raw raster images (XObjects) that are stored inside the PDF and
     writes them to disk byte-for-byte. Fast and lossless, BUT it only sees
     bitmaps. A chart drawn with vector lines, or a composite figure made of
     arrows + boxes + text + small bitmaps, is invisible to it, and a single
     figure can arrive as several unrelated fragments.

  2. Figure-region cropping (`extract_figure_regions`)
     -------------------------------------------------
     Finds each "Figure N:" / "Table N:" caption, works out the rectangle the
     figure occupies (using caption position + vector-drawing geometry), and
     RENDERS that rectangle to a PNG with `get_pixmap(clip=...)`. Because it
     renders, it captures everything that is visually there - vector charts,
     arrows, labels and bitmaps - as one complete picture.

`extract_all` runs both and cross-links them: an embedded image that sits
inside a cropped region is flagged as a *fragment* of that figure, so the two
views stay consistent.

TABLES are deliberately NOT cropped as figures here. Responsibilities are kept
apart:

  * Text-based tables (real selectable text) belong to `pdf_parser.py`, whose
    `extract_tables_from_pdf` reads their rows and cells.
  * `detect_tables` in THIS module looks at every "Table N:" caption and decides
    which bucket it falls in - text-based (hand off to pdf_parser) or image-based
    (a scanned/screenshot table with no text). Image-based tables are recorded as
    needing OCR / manual review and their region is saved for inspection, so they
    are never silently ignored.

This stage uses ONLY PyMuPDF geometry + text. No LLM and no computer-vision
model - a later stage (`vlm_client.py`) is where a vision model such as LLaVA
reads the pictures this module produces.

Run directly to test against the bundled sample paper:

    python src/multimodal/figure_extractor.py
"""

import hashlib
import re
from pathlib import Path

import pymupdf


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Images smaller than this (pixels) are almost always logos, rules or bullets
# rather than real figures, so the embedded-image pass skips them.
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50

# Resolution for rendered figure crops. 150 DPI is a good balance between
# sharp enough for a vision model and small enough on disk.
CROP_DPI = 150

# Page margin (points) used as the default top/side boundary of a figure when
# nothing better is found. 72 points = 1 inch.
PAGE_MARGIN = 45

# A caption block must start like "Figure 3:", "Fig. 3.", "Table 2:" or
# "FIG. 5:". The number MUST be followed by ":" or "." - without that rule a
# body paragraph that merely starts "Figure 3 shows that ..." would be
# mistaken for a caption. An optional sub-figure letter ("Figure 1a:") is
# allowed.
CAPTION_PATTERN = re.compile(
    r"^\s*(figure|fig\.?|table|chart)\s*\.?\s*(\d+)[a-z]?\s*[:.]",
    re.IGNORECASE)

# A text block counts as a "body paragraph" (and therefore a hard boundary
# that a figure cannot cross) once it is at least this many characters long.
# Figure-internal labels such as "3x3 max pool, stride 2" are far shorter.
BODY_TEXT_MIN_CHARS = 150

# How far (points) from a caption we are willing to look for its figure. A
# figure can fill almost a whole column, so this is deliberately generous; the
# real boundaries come from body paragraphs and the page margin.
MAX_REGION_REACH = 700

# A drawing whose on-page rectangle covers more than this fraction of the page
# is treated as a clip/transform mask artifact and ignored.
PAGE_MASK_AREA_FRACTION = 0.8

# An "uncertain" region shorter than this (points) is treated as a failure to
# locate the figure rather than rendered as a near-empty sliver. This mostly
# affects text-only tables that have no vector rules to anchor on.
MIN_REGION_HEIGHT = 24


# ---------------------------------------------------------------------------
# Small text / geometry helpers
# ---------------------------------------------------------------------------

def _normalise_whitespace(text):
    """Collapse newlines and repeated spaces into single spaces."""
    return " ".join(text.split())


def _console_safe(text):
    """
    Make a string printable on a Windows terminal.

    Research PDFs contain typographic characters (ligatures like "fi", curly
    quotes, en dashes) the default Windows console cannot encode, which would
    crash `print()`. We swap anything unprintable for "?" FOR DISPLAY ONLY -
    the stored metadata keeps the original characters.
    """
    return text.encode("ascii", errors="replace").decode("ascii")


def _horizontal_overlap(rect_a, rect_b):
    """Return True if two rectangles overlap along the x-axis at all."""
    return rect_a.x0 < rect_b.x1 and rect_b.x0 < rect_a.x1


def _midpoint_in_column(rect, column):
    """
    Return True if a rectangle's horizontal centre lies inside a column band.

    This is a STRICTER test than `_horizontal_overlap`, used when deciding
    whether a body paragraph belongs to the same column as a figure. A loose
    "any overlap" test wrongly treats the neighbouring column's prose (which
    just touches the midline) as a ceiling and chops the figure short.
    """
    centre = (rect.x0 + rect.x1) / 2
    return column[0] <= centre <= column[1]


def _clip_rect(rect, bounds):
    """
    Intersect `rect` with `bounds`, KEEPING straight lines.

    PyMuPDF's own `rect & bounds` treats a zero-width or zero-height rectangle
    as "empty" - and every perfectly horizontal or vertical line (a chart axis,
    a table rule, an arrow shaft) has exactly such a rectangle. Using `&` and
    `is_empty` therefore silently threw those lines away. This helper clamps
    the coordinates by hand and only returns None when the rectangle really
    lies outside `bounds`.
    """
    x0 = max(rect.x0, bounds.x0)
    y0 = max(rect.y0, bounds.y0)
    x1 = min(rect.x1, bounds.x1)
    y1 = min(rect.y1, bounds.y1)
    if x0 > x1 or y0 > y1:
        return None
    return pymupdf.Rect(x0, y0, x1, y1)


def _unique_image_xrefs(page):
    """
    Return each embedded image xref on a page exactly once, in listing order.

    `page.get_images()` lists an xref once per *reference* in the page's
    resources, so an image drawn twice can appear twice. Combined with
    `get_image_rects` (which already returns every placement of that xref),
    iterating the raw list would record each placement twice.
    """
    seen = set()
    xrefs = []
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref not in seen:
            seen.add(xref)
            xrefs.append(xref)
    return xrefs


def _containment_fraction(inner, outer):
    """
    Fraction of `inner`'s area that lies inside `outer` (0.0 - 1.0).

    Used to decide whether an embedded image is "inside" a figure region.
    """
    intersection = inner & outer          # PyMuPDF rectangle intersection
    if inner.get_area() <= 0:
        return 0.0
    return intersection.get_area() / inner.get_area()


def _slugify_label(label):
    """Turn 'Figure 2' into 'figure2' for use in a filename."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


# ---------------------------------------------------------------------------
# Page-level geometry collection
# ---------------------------------------------------------------------------

def _collect_captions(page):
    """
    Find every caption on a page.

    Returns a list of dicts, one per caption:
        {"label": "Figure 2", "kind": "figure"|"table",
         "number": 2, "rect": Rect, "text": "Figure 2: ..."}
    """
    captions = []

    for block in page.get_text("blocks"):
        text = _normalise_whitespace(block[4])
        match = CAPTION_PATTERN.match(text)
        if not match:
            continue

        word = match.group(1).lower()
        kind = "table" if word.startswith("table") else "figure"
        number = int(match.group(2))

        captions.append({
            "label": f"{kind.capitalize()} {number}",
            "kind": kind,
            "number": number,
            "rect": pymupdf.Rect(block[:4]),
            "text": text,
        })

    return captions


def _collect_graphics(page):
    """
    Collect the bounding boxes of every graphic element on a page.

    "Graphic" means embedded images plus vector drawings (lines, curves,
    filled shapes). These boxes are used only to work out WHERE a figure sits;
    the actual pixels come later from rendering.

    Clip/transform mask artifacts (drawings whose box spills far outside the
    page or blankets almost the whole page) are discarded, because PyMuPDF
    sometimes reports them with absurd coordinates.

    Returns a list of PyMuPDF Rect objects, all clamped to the page.
    """
    page_rect = page.rect
    page_area = page_rect.get_area()
    graphics = []

    # --- embedded images: use their on-page placement rectangles -----------
    for xref in _unique_image_xrefs(page):
        try:
            for rect in page.get_image_rects(xref):
                graphics.append(pymupdf.Rect(rect))
        except Exception:
            # A malformed image reference should never sink the whole page.
            continue

    # --- vector drawings ---------------------------------------------------
    for drawing in page.get_drawings():
        # Clamp to the visible page to tame off-page artifact coordinates.
        # `_clip_rect` (not `&`) so straight axis/rule lines are kept.
        rect = _clip_rect(pymupdf.Rect(drawing["rect"]), page_rect)
        if rect is None:
            continue

        # Drop full-page masks (used internally for clipping), but keep thin
        # lines (height or width ~0) because table rules and axes matter.
        if rect.get_area() > PAGE_MASK_AREA_FRACTION * page_area:
            continue

        graphics.append(rect)

    return graphics


def _embedded_image_rects(page):
    """
    Return just the on-page rectangles of embedded raster images.

    `_collect_graphics` mixes images and vector drawings together, but to tell
    an IMAGE-based table from a text/vector one we need the images on their own.
    """
    rects = []
    for xref in _unique_image_xrefs(page):
        try:
            for rect in page.get_image_rects(xref):
                rects.append(pymupdf.Rect(rect))
        except Exception:
            continue
    return rects


def _vertical_gap(rect_a, rect_b):
    """
    Vertical distance (points) between two rectangles; 0 if they overlap in y.
    """
    if rect_a.y1 <= rect_b.y0:
        return rect_b.y0 - rect_a.y1
    if rect_b.y1 <= rect_a.y0:
        return rect_a.y0 - rect_b.y1
    return 0.0


def _collect_body_paragraphs(page):
    """
    Return the rectangles of substantial body-text paragraphs on a page.

    These act as hard ceilings/floors: a figure region is not allowed to grow
    across a real paragraph of prose. Short blocks (axis labels, captions) are
    excluded by the length threshold.
    """
    paragraphs = []
    for block in page.get_text("blocks"):
        text = _normalise_whitespace(block[4])
        if len(text) < BODY_TEXT_MIN_CHARS:
            continue
        if CAPTION_PATTERN.match(text):
            continue
        paragraphs.append(pymupdf.Rect(block[:4]))
    return paragraphs


def _caption_column(caption_rect, page):
    """
    Decide which column a caption belongs to and return that column's x-range.

    A wide caption spans the full page; otherwise it is the left or right
    column. Returns (x0, x1) in points.
    """
    page_width = page.rect.width
    margin = PAGE_MARGIN

    # Full-width figure: caption stretches across most of the page.
    if caption_rect.width > 0.55 * page_width:
        return (margin, page_width - margin)

    centre = (caption_rect.x0 + caption_rect.x1) / 2
    if centre < page_width / 2:
        # Left column, with a little slack past the midline.
        return (margin, page_width / 2 + 12)
    # Right column.
    return (page_width / 2 - 12, page_width - margin)


# ---------------------------------------------------------------------------
# Region computation (the heart of the vector/composite support)
# ---------------------------------------------------------------------------

def _grow_region(caption, direction, column, graphics, boundaries, page):
    """
    Compute the figure rectangle on one side of a caption.

    Parameters
    ----------
    caption : dict         (from `_collect_captions`)
    direction : "up" | "down"
        Where the figure body lies relative to the caption. Figures in this
        paper sit ABOVE their caption; table bodies can go either way.
    column : (x0, x1)      horizontal band to stay within
    graphics : list of Rect
        Bounding boxes of images + vector drawings (from `_collect_graphics`).
    boundaries : list of Rect
        Rectangles the region must not cross: body paragraphs plus the captions
        of other figures/tables.
    page : the PDF page

    Returns
    -------
    (rect_or_None, confidence)
        `rect` is the region to render; `confidence` is:
          "detected"  - graphics were found and tightly bound the region
          "uncertain" - no graphics found; we fall back to a plain slab and
                        cannot promise the crop is a clean figure
    """
    cap = caption["rect"]
    col_rect = pymupdf.Rect(column[0], page.rect.y0, column[1], page.rect.y1)
    page_rect = page.rect

    # --- 1: find the boundary set by the nearest paragraph OR caption ------
    # `boundaries` holds body paragraphs AND the captions of other figures and
    # tables. Stopping at another caption is what stops a table from reaching
    # up and swallowing the figure above it.
    if direction == "up":
        # Default ceiling is the top page margin.
        boundary = page_rect.y0 + PAGE_MARGIN
        for other in boundaries:
            # Only something in THIS column can act as a ceiling.
            if not _midpoint_in_column(other, column):
                continue
            # A block that ends above the caption pushes the ceiling down.
            if other.y1 <= cap.y0 + 2:
                boundary = max(boundary, other.y1 + 2)
        region_top = max(boundary, cap.y0 - MAX_REGION_REACH)
        region_bottom = cap.y0 - 1
    else:  # direction == "down"
        boundary = page_rect.y1 - PAGE_MARGIN
        for other in boundaries:
            if not _midpoint_in_column(other, column):
                continue
            if other.y0 >= cap.y1 - 2:
                boundary = min(boundary, other.y0 - 2)
        region_top = cap.y1 + 1
        region_bottom = min(boundary, cap.y1 + MAX_REGION_REACH)

    if region_bottom <= region_top:
        return None, "uncertain"

    search_band = pymupdf.Rect(column[0], region_top, column[1], region_bottom)

    # --- 2: tighten to the graphics that fall inside the search band -------
    inside = []
    for rect in graphics:
        if not _horizontal_overlap(rect, col_rect):
            continue
        # Require the graphic to lie (mostly) within the vertical search band.
        if rect.y1 <= region_top + 2 or rect.y0 >= region_bottom - 2:
            continue
        clipped = _clip_rect(rect, search_band)
        if clipped is not None:
            inside.append(clipped)

    if inside:
        x0 = min(r.x0 for r in inside)
        y0 = min(r.y0 for r in inside)
        x1 = max(r.x1 for r in inside)
        y1 = max(r.y1 for r in inside)
        region = pymupdf.Rect(x0, y0, x1, y1)

        # Pad a little so we don't shave off axis ticks or outer strokes.
        region += (-6, -6, 6, 6)
        # Keep the region inside its own column so a single-column figure does
        # not bleed into the neighbouring column's text, then clamp to page.
        region = region & col_rect & page_rect
        return region, "detected"

    # --- 3: nothing graphical found -> fall back to the plain slab ---------
    fallback = search_band & page_rect
    if fallback.is_empty or fallback.get_area() <= 0:
        return None, "uncertain"
    return fallback, "uncertain"


def _compute_figure_region(caption, all_captions, graphics, paragraphs, page):
    """
    Work out the region for one caption, choosing a search direction.

    Figures: look above the caption (confirmed layout for this paper).
    Tables : look both ways and keep whichever side has more graphic content,
             since table captions sit above or below depending on the paper.

    `all_captions` lets a region stop at a neighbouring caption instead of
    crossing into another figure's or table's content.
    """
    column = _caption_column(caption["rect"], page)

    # Boundaries = body paragraphs + the OTHER captions on this page.
    boundaries = list(paragraphs)
    for other in all_captions:
        if other is not caption:
            boundaries.append(other["rect"])

    if caption["kind"] == "figure":
        region, confidence = _grow_region(
            caption, "up", column, graphics, boundaries, page)
        return region, confidence, "up"

    # Tables: compare both directions by graphic area captured.
    up_region, up_conf = _grow_region(
        caption, "up", column, graphics, boundaries, page)
    down_region, down_conf = _grow_region(
        caption, "down", column, graphics, boundaries, page)

    up_area = up_region.get_area() if (up_region and up_conf == "detected") else 0
    down_area = down_region.get_area() if (down_region and down_conf == "detected") else 0

    if down_area > up_area:
        return down_region, down_conf, "down"
    if up_area > 0:
        return up_region, up_conf, "up"
    # Neither side had graphics; fall back to whichever slab exists.
    if up_region:
        return up_region, up_conf, "up"
    return down_region, down_conf, "down"


# ===========================================================================
# PUBLIC API 1: embedded-image extraction
# ===========================================================================

def extract_embedded_images(
    pdf_path,
    output_dir=None,
    min_width=MIN_IMAGE_WIDTH,
    min_height=MIN_IMAGE_HEIGHT,
):
    """
    Extract the raw embedded raster images from a PDF, one record per PLACEMENT.

    Unlike a naive extractor, this does NOT throw away repeats. The same image
    object can be drawn in several spots (or reused on several pages); each
    drawing is recorded as its own placement. When a placement's pixels are
    byte-identical to one already written, we set `duplicate_of` to the first
    file's name and point `file_path` at that file instead of writing a copy.

    Returns
    -------
    list[dict], one per placement, in page then placement order:

        {
            "page_number": 7,
            "placement_index": 2,            # 1-based, within the page
            "file_name": "page07_img02.jpeg",
            "file_path": ".../page07_img01.jpeg",   # original, if duplicate
            "width": 775, "height": 1075,    # pixels of the stored image
            "format": "jpeg",
            "colorspace": "ICCBased(RGB,...)",
            "bits_per_component": 8,
            "size_bytes": 146813,
            "bbox": [57.0, 104.0, 166.0, 257.0],     # on-page placement, pts
            "duplicate_of": "page07_img01.jpeg" | None,
            "classification": "standalone_image",    # refined by extract_all
        }

    Raises
    ------
    FileNotFoundError : path missing or not a file.
    ValueError        : file exists but is not a readable PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Not a file: {pdf_path}")

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "sample_pdfs" / "extracted_figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as error:
        raise ValueError(f"Could not open PDF: {pdf_path} ({error})") from error

    if not pdf.is_pdf:
        pdf.close()
        raise ValueError(f"File is not a PDF document: {pdf_path}")

    records = []
    # Maps an image fingerprint -> the file name it was first written as.
    seen_hashes = {}

    with pdf:
        for page_number, page in enumerate(pdf, start=1):

            placement_index = 0

            # Each xref once per page; get_image_rects then yields every
            # placement of it, so no placement is counted twice.
            for xref in _unique_image_xrefs(page):

                try:
                    image_data = pdf.extract_image(xref)
                except Exception as error:
                    print(f"  [warning] page {page_number}: cannot extract "
                          f"xref {xref} ({error})")
                    continue
                if not image_data or not image_data.get("image"):
                    continue

                image_bytes = image_data["image"]
                width = image_data.get("width", 0)
                height = image_data.get("height", 0)
                extension = image_data.get("ext", "png")

                # Skip tiny decorations.
                if width < min_width or height < min_height:
                    continue

                # On-page placement rectangles (one image may be drawn twice).
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                if not rects:
                    # No geometry (rare) - record a single placement with an
                    # empty bbox so the occurrence is not lost.
                    rects = [None]

                fingerprint = hashlib.sha256(image_bytes).hexdigest()

                for rect in rects:
                    placement_index += 1
                    file_name = (
                        f"page{page_number:02d}_img{placement_index:02d}.{extension}"
                    )

                    if fingerprint in seen_hashes:
                        # Duplicate pixels: keep the occurrence, reuse the file.
                        original_name = seen_hashes[fingerprint]
                        file_path = output_dir / original_name
                        duplicate_of = original_name
                    else:
                        file_path = output_dir / file_name
                        try:
                            with open(file_path, "wb") as handle:
                                handle.write(image_bytes)
                        except OSError as error:
                            print(f"  [warning] page {page_number}: cannot write "
                                  f"{file_name} ({error})")
                            placement_index -= 1
                            continue
                        seen_hashes[fingerprint] = file_name
                        duplicate_of = None

                    bbox = ([rect.x0, rect.y0, rect.x1, rect.y1]
                            if rect is not None else None)

                    records.append({
                        "page_number": page_number,
                        "placement_index": placement_index,
                        "file_name": file_name,
                        "file_path": str(file_path),
                        "width": width,
                        "height": height,
                        "format": extension,
                        "colorspace": image_data.get("cs-name", "unknown"),
                        "bits_per_component": image_data.get("bpc", 0),
                        "size_bytes": len(image_bytes),
                        "bbox": bbox,
                        "duplicate_of": duplicate_of,
                        # Refined to "image_fragment" by extract_all when the
                        # image turns out to sit inside a cropped figure.
                        "classification": "standalone_image",
                        # Set by extract_all; always present so every record
                        # has the same keys whichever function produced it.
                        "fragment_of": None,
                    })

    return records


# ===========================================================================
# PUBLIC API 2: figure-region cropping (vector + composite support)
# ===========================================================================

def extract_figure_regions(pdf_path, output_dir=None, dpi=CROP_DPI):
    """
    Detect caption-anchored FIGURE regions and render each to a PNG.

    Because the region is RENDERED (not copied), this captures vector charts,
    arrows, labels and composite figures that embedded-image extraction cannot
    see. Each "Figure N:" caption yields at most one crop.

    Tables are intentionally skipped here - they are routed by `detect_tables`
    (text tables to `pdf_parser`, image tables to OCR/manual review).

    Returns
    -------
    list[dict], one per detected region, in page then caption order:

        {
            "page_number": 6,
            "label": "Figure 2",
            "figure_number": 2,
            "kind": "figure" | "table",
            "caption": "Figure 2: Distributions of figures ...",
            "file_name": "page06_figure2.png",
            "file_path": ".../page06_figure2.png",
            "bbox": [45.0, 84.0, 567.0, 237.0],      # rendered region, points
            "search_direction": "up" | "down",
            "extraction_method": "region_crop",
            "region_type": "figure_region" | "table_region",
            "region_confidence": "detected" | "uncertain" | "not_located",
        }

        region_confidence meanings:
          "detected"    - vector/image geometry bounded the region; reliable.
          "uncertain"   - no geometry found; the crop is a best-effort slab and
                          may include surrounding whitespace or text. Not exact.
          "not_located" - the figure could not be placed at all (e.g. a
                          text-only table with no rules). No image is written;
                          `file_name`, `file_path` and `bbox` are None, but the
                          caption, page and figure number are still recorded.

    Raises the same errors as `extract_embedded_images`.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Not a file: {pdf_path}")

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "sample_pdfs" / "extracted_figure_regions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as error:
        raise ValueError(f"Could not open PDF: {pdf_path} ({error})") from error
    if not pdf.is_pdf:
        pdf.close()
        raise ValueError(f"File is not a PDF document: {pdf_path}")

    regions = []
    used_names = set()   # crop filenames already written in this run

    with pdf:
        for page_number, page in enumerate(pdf, start=1):

            captions = _collect_captions(page)
            if not captions:
                continue  # Pages with no caption produce no regions - fine.

            # We still keep ALL captions (figures + tables) so a figure can use
            # a neighbouring table caption as a boundary, but we only CROP
            # figures. Tables are handled by `detect_tables`.
            figure_captions = [c for c in captions if c["kind"] == "figure"]
            if not figure_captions:
                continue

            graphics = _collect_graphics(page)
            paragraphs = _collect_body_paragraphs(page)

            for caption in figure_captions:
                region, confidence, direction = _compute_figure_region(
                    caption, captions, graphics, paragraphs, page)

                # A caption whose figure cannot be located (no region, or only
                # a degenerate uncertain sliver - typical of a text-only table)
                # is still recorded so its number and caption are preserved, but
                # no misleading image file is written.
                degenerate = (region is not None
                              and confidence == "uncertain"
                              and region.height < MIN_REGION_HEIGHT)

                if region is None or region.get_area() <= 0 or degenerate:
                    print(f"  [note] page {page_number}: could not reliably "
                          f"locate a region for {caption['label']} "
                          f"(caption kept, no crop saved)")
                    regions.append({
                        "page_number": page_number,
                        "label": caption["label"],
                        "figure_number": caption["number"],
                        "kind": caption["kind"],
                        "caption": caption["text"],
                        "file_name": None,
                        "file_path": None,
                        "bbox": None,
                        "search_direction": direction,
                        "extraction_method": "region_crop",
                        "region_type": f"{caption['kind']}_region",
                        "region_confidence": "not_located",
                    })
                    continue

                # Render the region to an image. If two captions on one page
                # share a label (e.g. a figure continued), add a suffix so the
                # second crop never overwrites the first.
                base_name = (
                    f"page{page_number:02d}_{_slugify_label(caption['label'])}"
                )
                file_name = f"{base_name}.png"
                suffix = 2
                while file_name in used_names:
                    file_name = f"{base_name}_{suffix}.png"
                    suffix += 1
                used_names.add(file_name)
                file_path = output_dir / file_name

                try:
                    pixmap = page.get_pixmap(clip=region, dpi=dpi)
                    pixmap.save(file_path)
                except Exception as error:
                    print(f"  [warning] page {page_number}: cannot render "
                          f"{caption['label']} ({error})")
                    continue

                regions.append({
                    "page_number": page_number,
                    "label": caption["label"],
                    "figure_number": caption["number"],
                    "kind": caption["kind"],
                    "caption": caption["text"],
                    "file_name": file_name,
                    "file_path": str(file_path),
                    "bbox": [region.x0, region.y0, region.x1, region.y1],
                    "search_direction": direction,
                    "extraction_method": "region_crop",
                    "region_type": f"{caption['kind']}_region",
                    "region_confidence": confidence,
                })

    return regions


# ===========================================================================
# PUBLIC API 2b: table caption triage (text vs image-based)
# ===========================================================================

def detect_tables(pdf_path, review_dir=None, dpi=CROP_DPI):
    """
    Classify every "Table N:" caption as text-based or image-based.

    This module does NOT read table contents - that is `pdf_parser`'s job. Its
    only responsibility for tables is TRIAGE, so that no table is lost:

      * text-based  : PyMuPDF's table finder locates a real (selectable-text)
                      table near the caption. Status "text_based"; the caller
                      should read it with `pdf_parser.extract_tables_from_pdf`.
      * image-based : no text table is found, but an embedded image covers the
                      caption's area - the table is a scan/screenshot. Status
                      "image_based_requires_ocr"; the region is rendered to
                      `review_dir` so a human (or a later OCR step) can handle it.
      * unresolved  : a caption with neither a detected table nor a covering
                      image. Status "unresolved_manual_review"; recorded, not
                      dropped.

    Parameters
    ----------
    pdf_path : str | Path
    review_dir : str | Path, optional
        Where crops of image-based tables are saved. Defaults to
        `<project_root>/sample_pdfs/tables_requiring_ocr/`. Created only if an
        image-based table is actually found.
    dpi : int
        Render resolution for image-based table crops.

    Returns
    -------
    list[dict], one per table caption:

        {
            "page_number": 5,
            "label": "Table 1",
            "table_number": 1,
            "caption": "Table 1: Number of papers ...",
            "status": "text_based" | "image_based_requires_ocr"
                      | "unresolved_manual_review",
            "handled_by": "pdf_parser" | "ocr_or_manual_review"
                          | "manual_review",
            "bbox": [x0, y0, x1, y1] | None,   # table/image region if known
            "file_name": str | None,           # crop, only for image-based
            "file_path": str | None,
        }

    Raises the same errors as `extract_embedded_images`.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Not a file: {pdf_path}")

    if review_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        review_dir = project_root / "sample_pdfs" / "tables_requiring_ocr"
    review_dir = Path(review_dir)

    try:
        pdf = pymupdf.open(pdf_path)
    except Exception as error:
        raise ValueError(f"Could not open PDF: {pdf_path} ({error})") from error
    if not pdf.is_pdf:
        pdf.close()
        raise ValueError(f"File is not a PDF document: {pdf_path}")

    table_notes = []

    with pdf:
        for page_number, page in enumerate(pdf, start=1):

            captions = _collect_captions(page)
            table_captions = [c for c in captions if c["kind"] == "table"]
            if not table_captions:
                continue

            # Real text tables on this page (by their bounding boxes).
            try:
                found_tables = page.find_tables().tables
            except Exception:
                found_tables = []
            image_rects = _embedded_image_rects(page)

            for caption in table_captions:
                cap_rect = caption["rect"]
                column = _caption_column(cap_rect, page)
                col_rect = pymupdf.Rect(column[0], page.rect.y0,
                                        column[1], page.rect.y1)

                # --- look for a text table in the same column, nearby --------
                text_table_box = None
                best_gap = None
                for table in found_tables:
                    tbox = pymupdf.Rect(table.bbox)
                    if not _horizontal_overlap(tbox, col_rect):
                        continue
                    gap = _vertical_gap(tbox, cap_rect)
                    if gap <= MAX_REGION_REACH and (best_gap is None or gap < best_gap):
                        best_gap = gap
                        text_table_box = tbox

                if text_table_box is not None:
                    table_notes.append({
                        "page_number": page_number,
                        "label": caption["label"],
                        "table_number": caption["number"],
                        "caption": caption["text"],
                        "status": "text_based",
                        "handled_by": "pdf_parser",
                        "bbox": [text_table_box.x0, text_table_box.y0,
                                 text_table_box.x1, text_table_box.y1],
                        "file_name": None,
                        "file_path": None,
                    })
                    continue

                # --- no text table: is a sizeable image covering the area? ---
                covering_image = None
                for rect in image_rects:
                    if not _horizontal_overlap(rect, col_rect):
                        continue
                    if _vertical_gap(rect, cap_rect) > MAX_REGION_REACH:
                        continue
                    # Only a reasonably large image is plausibly a whole table.
                    if rect.get_area() >= 5000 and (covering_image is None
                            or rect.get_area() > covering_image.get_area()):
                        covering_image = rect

                if covering_image is not None:
                    # Save the image region so the table is available for OCR or
                    # manual review - it is flagged, never silently ignored.
                    review_dir.mkdir(parents=True, exist_ok=True)
                    file_name = (f"page{page_number:02d}_"
                                 f"{_slugify_label(caption['label'])}_requires_ocr.png")
                    file_path = review_dir / file_name
                    try:
                        page.get_pixmap(clip=covering_image, dpi=dpi).save(file_path)
                        saved_name, saved_path = file_name, str(file_path)
                    except Exception as error:
                        print(f"  [warning] page {page_number}: cannot render "
                              f"image table {caption['label']} ({error})")
                        saved_name, saved_path = None, None

                    table_notes.append({
                        "page_number": page_number,
                        "label": caption["label"],
                        "table_number": caption["number"],
                        "caption": caption["text"],
                        "status": "image_based_requires_ocr",
                        "handled_by": "ocr_or_manual_review",
                        "bbox": [covering_image.x0, covering_image.y0,
                                 covering_image.x1, covering_image.y1],
                        "file_name": saved_name,
                        "file_path": saved_path,
                    })
                    continue

                # --- neither: record for a human to look at ------------------
                table_notes.append({
                    "page_number": page_number,
                    "label": caption["label"],
                    "table_number": caption["number"],
                    "caption": caption["text"],
                    "status": "unresolved_manual_review",
                    "handled_by": "manual_review",
                    "bbox": None,
                    "file_name": None,
                    "file_path": None,
                })

    return table_notes


# ===========================================================================
# PUBLIC API 3: run both and cross-link
# ===========================================================================

def extract_all(pdf_path, image_dir=None, region_dir=None, review_dir=None):
    """
    Run the image, figure and table-triage passes and cross-link them.

    An embedded image whose placement lies mostly inside a cropped figure
    region is reclassified as "image_fragment" and gains `fragment_of`
    (the region file it belongs to). Images outside every region stay
    "standalone_image".

    Table CONTENTS are not read here (that is `pdf_parser`'s job); `tables`
    holds the triage result from `detect_tables` so a caller knows which tables
    to read with pdf_parser and which need OCR.

    Returns
    -------
    dict:
        {
            "embedded_images": [ ...records from extract_embedded_images... ],
            "figure_regions":  [ ...records from extract_figure_regions... ],
            "tables":          [ ...records from detect_tables... ],
            "pdf_path": "<the input path>",
        }
    """
    images = extract_embedded_images(pdf_path, output_dir=image_dir)
    regions = extract_figure_regions(pdf_path, output_dir=region_dir)
    tables = detect_tables(pdf_path, review_dir=review_dir)

    # Index regions by page for a quick lookup.
    regions_by_page = {}
    for region in regions:
        regions_by_page.setdefault(region["page_number"], []).append(region)

    for image in images:
        if image["bbox"] is None:
            continue
        image_rect = pymupdf.Rect(image["bbox"])

        for region in regions_by_page.get(image["page_number"], []):
            # A "not_located" region has no bbox, so it cannot contain anything.
            if region["bbox"] is None:
                continue
            region_rect = pymupdf.Rect(region["bbox"])
            # "Mostly inside" = 70% of the image area falls within the region.
            if _containment_fraction(image_rect, region_rect) >= 0.70:
                image["classification"] = "image_fragment"
                image["fragment_of"] = region["file_name"]
                break
        else:
            image["fragment_of"] = None

    return {
        "embedded_images": images,
        "figure_regions": regions,
        "tables": tables,
        "pdf_path": str(pdf_path),
    }


# ===========================================================================
# Backward-compatible alias
# ===========================================================================

def extract_figures_from_pdf(pdf_path, output_dir=None, **kwargs):
    """
    Backward-compatible name for the embedded-image extractor.

    Kept so earlier code/tests that call `extract_figures_from_pdf` keep
    working. New code should prefer `extract_embedded_images`,
    `extract_figure_regions`, or `extract_all`.
    """
    return extract_embedded_images(pdf_path, output_dir=output_dir, **kwargs)


# ===========================================================================
# Test / demo block
# ===========================================================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    pdf_path = project_root / "sample_pdfs" / "sample_paper1.pdf"
    image_dir = project_root / "sample_pdfs" / "extracted_figures"
    region_dir = project_root / "sample_pdfs" / "extracted_figure_regions"

    print(f"Reading PDF : {pdf_path}")
    print(f"Images  -> {image_dir}")
    print(f"Regions -> {region_dir}")
    print()

    try:
        result = extract_all(pdf_path, image_dir=image_dir, region_dir=region_dir)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

    images = result["embedded_images"]
    regions = result["figure_regions"]
    tables = result["tables"]

    # ----- headline counts (kept deliberately separate) --------------------
    unique_images = len({img["file_name"] for img in images
                         if img["duplicate_of"] is None})
    fragments = sum(1 for img in images
                    if img["classification"] == "image_fragment")
    standalone = sum(1 for img in images
                     if img["classification"] == "standalone_image")
    # Figure crops are counted apart from embedded images, so we never present
    # "number of images" as "number of figures".
    rendered = [r for r in regions if r["file_name"] is not None]
    confident = sum(1 for r in rendered if r["region_confidence"] == "detected")
    uncertain = sum(1 for r in rendered if r["region_confidence"] == "uncertain")
    not_located = sum(1 for r in regions if r["region_confidence"] == "not_located")
    # Table triage counts.
    text_tables = sum(1 for t in tables if t["status"] == "text_based")
    ocr_tables = sum(1 for t in tables
                     if t["status"] == "image_based_requires_ocr")
    unresolved_tables = sum(1 for t in tables
                            if t["status"] == "unresolved_manual_review")

    print("=" * 72)
    print("SUMMARY (images, figures and tables counted separately)")
    print("=" * 72)
    print(f"Embedded image placements found : {len(images)}")
    print(f"  - unique image files written  : {unique_images}")
    print(f"  - duplicate placements        : {len(images) - unique_images}")
    print(f"  - standalone images           : {standalone}")
    print(f"  - fragments of a larger figure: {fragments}")
    print(f"Figure captions detected        : {len(regions)}")
    print(f"Figure-region crops rendered    : {len(rendered)}")
    print(f"  - confidently bounded         : {confident}")
    print(f"  - best-effort (uncertain)     : {uncertain}")
    print(f"  - caption kept, no crop saved : {not_located}")
    print(f"Table captions detected         : {len(tables)}")
    print(f"  - text-based (-> pdf_parser)  : {text_tables}")
    print(f"  - image-based (-> OCR/review) : {ocr_tables}")
    print(f"  - unresolved (manual review)  : {unresolved_tables}")
    print()

    # ----- embedded images -------------------------------------------------
    print("-" * 72)
    print("EMBEDDED IMAGES (raw bitmaps stored in the PDF)")
    print("-" * 72)
    if not images:
        print("None found.")
    for img in images:
        note = ""
        if img["duplicate_of"]:
            note = f"  [duplicate of {img['duplicate_of']}]"
        elif img["classification"] == "image_fragment":
            note = f"  [fragment of {img['fragment_of']}]"
        print(f"page {img['page_number']}  {img['file_name']}  "
              f"{img['width']}x{img['height']}px  "
              f"{img['format'].upper()}{note}")
    print()

    # ----- figure regions --------------------------------------------------
    print("-" * 72)
    print("FIGURE-REGION CROPS (rendered - includes vector/composite figures)")
    print("-" * 72)
    if not regions:
        print("None found.")
    for region in regions:
        print(f"{region['label']} (page {region['page_number']})  "
              f"[{region['region_confidence']}]")
        if region["file_name"] is None:
            # Caption detected but the figure could not be reliably located.
            print("  Saved   : (no crop - figure region could not be located)")
        else:
            print(f"  Saved   : {region['file_path']}")
            bbox = region["bbox"]
            print(f"  Region  : x {bbox[0]:.0f}-{bbox[2]:.0f}  "
                  f"y {bbox[1]:.0f}-{bbox[3]:.0f} pt  "
                  f"(searched {region['search_direction']})")
        print(f"  Caption : {_console_safe(region['caption'])[:100]}")
        print()

    # ----- table triage ----------------------------------------------------
    print("-" * 72)
    print("TABLE TRIAGE (contents are read by pdf_parser, not cropped here)")
    print("-" * 72)
    if not tables:
        print("No table captions found.")
    for table in tables:
        print(f"{table['label']} (page {table['page_number']})  "
              f"[{table['status']}]")
        print(f"  Route   : {table['handled_by']}")
        if table["status"] == "text_based":
            print("  Action  : read rows via "
                  "pdf_parser.extract_tables_from_pdf()")
        elif table["status"] == "image_based_requires_ocr":
            print(f"  Saved   : {table['file_path']}")
            print("  Action  : needs OCR / manual review (table is an image)")
        else:
            print("  Action  : manual review (no table or image detected)")
        print(f"  Caption : {_console_safe(table['caption'])[:100]}")
        print()

    print("=" * 72)
    print("NOTE: Figure crops are best-effort regions, not a guarantee of exact")
    print("figure boundaries; crops marked 'uncertain' may include extra")
    print("whitespace or text. Table CONTENTS are extracted by pdf_parser.py;")
    print("image-based tables are flagged for OCR, never silently dropped.")
    print("Review the saved PNGs before downstream use.")
    print("Figure extraction completed!")
