#!/usr/bin/env python
"""Convert a document (PDF, EPUB, XPS, MOBI, FB2, CBZ, images, ... -
anything PyMuPDF can open) into readable UTF-8 HTML, Markdown, or plain
text, extracting embedded images along the way.

Handles the mixed/broken text encodings common in scanned/converted PDFs
(custom font subsets, private-use-area glyph codes, ligatures), falling
back to OCR (with its own layout reconstruction - see cluster_rows/
find_callout_boxes) when a page's text layer turns out to be unusable.
Supports converting a page range, streams output to disk page by page, and
automatically resumes an interrupted run.

Each page is extracted ONCE into a format-independent structure (see
extract_page_blocks) and cached to disk as JSON; HTML/Markdown/txt
rendering happens from that structure, so switching --out-format on a
later run never requires redoing OCR.
"""

import argparse
import base64
import io
import json
import re
import shutil
import sys
import unicodedata
import urllib.error
import urllib.request
from html import escape
from pathlib import Path

import pymupdf

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
OCR_LANGUAGES = ["ru", "en"]
OCR_RENDER_DPI = 300
IMAGE_RENDER_DPI = 200
# Share of control characters (below 0x20, excluding whitespace) in a page's
# extracted text above which we treat its text layer as unusable.
GARBLED_TEXT_THRESHOLD = 0.2
GARBLED_TEXT_MIN_LENGTH = 20
# Text confined to the top/bottom N% of a page and short enough is treated as
# a running header/footer (chapter title, page number) rather than body text.
MARGIN_ZONE_RATIO = 0.09
MARGIN_TEXT_MAX_LEN = 70
# OCR paragraphs that are mostly digits are almost always a residual mangled
# fragment of a page-number column (see module docstring / README) - flagged,
# not discarded.
NUMERIC_PARAGRAPH_RATIO = 0.6
NUMERIC_PARAGRAPH_MIN_CHARS = 4
LMSTUDIO_DEFAULT_URL = "http://localhost:1234/v1"
LMSTUDIO_TIMEOUT = 120

# A hyphen at a line-wrap point ("бо-\nлее" -> extracted as "бо- лее") should
# collapse back into one word ("более"). Requires a letter on both sides so we
# don't touch numeric ranges ("2020-2021") or list markers.
HYPHEN_WRAP_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])-\s+(?=[A-Za-zА-Яа-яЁё])")

# --- OCR layout reconstruction tuning ---
# Two raw OCR detections are the same printed row if their y-ranges overlap
# by at least this fraction (intersection / union of [y0,y1]) - NOT just
# center proximity. A table-of-contents title and its far-off page number
# (split apart by a sparse, undetected dot leader) sit almost exactly within
# each other's height despite a huge horizontal gap, so they clear this bar.
# Note this does NOT reliably separate two columns running in parallel at
# the same height (e.g. a margin callout box laid out on the same baseline
# grid as the main text overlaps just as much) - that ambiguity is why
# callout boxes are found and OCR'd separately via find_callout_boxes()
# instead of being told apart from text geometry alone.
ROW_MIN_Y_OVERLAP_RATIO = 0.5
# Two consecutive rows merge into one paragraph if the gap between them is
# under this multiple of the page's median row height.
PARAGRAPH_GAP_RATIO = 1.3
# A trailing "<space><1-4 digits>" is treated as a page-number reference that
# closes out a table-of-contents/index entry, so the next row always starts a
# new paragraph instead of running on into it. Also used to tell a page
# number apart from a running head in the page margin (see
# dedup_running_heads): only the latter is suppressed on repeat pages.
PAGE_NUMBER_TOKEN_RE = re.compile(r"^\d{1,4}[.,]?$")
TRAILING_PAGE_NUMBER_RE = re.compile(r"\s\d{1,4}[.,]?$")
PARAGRAPH_TERMINATOR_CHARS = ".!?:;»)”"
# split_merged_row(): the internal x-gap within a merged row must be at
# least this many pixels (at OCR_RENDER_DPI) AND this many times the row's
# other internal gaps before it's even considered a possible column split -
# ordinary inter-word/inter-fragment gaps are far smaller than either.
COLUMN_SPLIT_MIN_GAP = 100
COLUMN_SPLIT_GAP_RATIO = 3.0
# detect_column_boundary_x(): same idea as COLUMN_SPLIT_MIN_GAP/_GAP_RATIO
# above, but applied to PDF text-BLOCK x0 positions in point space (not OCR
# pixel space) - hence the separate, smaller absolute threshold.
COLUMN_BOUNDARY_MIN_GAP_PT = 24
# _blocks_run_in_parallel(): fraction of the smaller candidate column's
# blocks that must have a same-height counterpart in the other candidate
# column before the split is trusted as genuine side-by-side columns.
COLUMN_SIBLING_MATCH_MIN_RATIO = 0.5
# A vector-graphics region (see find_callout_boxes) spanning most of the
# page width is a full-width rule (e.g. a header underline), not a margin
# callout box; anything smaller than this is noise, not a real box.
CALLOUT_MAX_WIDTH_RATIO = 0.85
CALLOUT_MIN_SIZE_PT = 20
CALLOUT_IMAGE_OVERLAP_MAX_RATIO = 0.5
# extract_page_blocks() (non-OCR path only - see find_page_tables): a text
# block counts as "consumed by" a detected table when this much of its own
# area sits inside the table's bbox, so it's rendered once as part of the
# table instead of also appearing as a separate, duplicated paragraph.
TABLE_BLOCK_OVERLAP_MIN_RATIO = 0.5
# page.find_tables() often over-segments into spurious all-empty columns
# and turns a wrapped multi-line cell into an extra almost-empty row - a
# detected table is only trusted once cleaned down to at least this many
# real columns and rows (see clean_table_rows).
TABLE_MIN_COLUMNS = 2
TABLE_MIN_ROWS = 2

# --- Heading detection tuning ---
# A text block/paragraph is promoted to a heading if its dominant font size
# is at least this many times the page's own body-text size (see
# compute_body_font_size) - or, on the OCR path where no font metadata
# survives, if a single OCR row's height is at least this many times the
# page's median row height (larger glyphs measure taller).
HEADING_SIZE_RATIO_H3 = 1.15
HEADING_SIZE_RATIO_H2 = 1.35
HEADING_BOLD_RATIO_MIN = 0.8
HEADING_BOLD_SIZE_TOLERANCE = 0.05
HEADING_MAX_CHARS = 120
# The page's body size is only trusted once measured across a reasonable
# amount of running text - a title/colophon page has no real "body size" to
# compare against, and guessing wrong there would misclassify everything on
# the page instead of nothing.
HEADING_BODY_SIZE_MIN_CHARS = 200
HEADING_OCR_HEIGHT_RATIO_H3 = 1.25
HEADING_OCR_HEIGHT_RATIO_H2 = 1.5
HEADING_OCR_MAX_CHARS = 80

# ocr_region_paragraphs(): converts a paragraph's pixel indent (relative to
# its own column's leftmost row) into a CSS "em" unit, so a nested sub-item
# under a column's own entries (e.g. bullets under a table-of-contents
# title) keeps a visible indent in HTML even without real list markup. Only
# ever computed for column-tagged paragraphs (a genuine PDF-geometry-
# detected column) - an ordinary single-column paragraph's first line is
# often indented by normal book typography, which would otherwise be
# misread as a nested sub-item.
OCR_INDENT_PX_PER_EM = OCR_RENDER_DPI / 72 * 11
OCR_INDENT_MIN_EM = 0.3
OCR_INDENT_MAX_EM = 4.0

_caption_model = None
_caption_processor = None
_caption_device = "cpu"
_ocr_reader = None


def resolve_use_gpu(preference):
    """None means auto-detect; True/False forces the choice."""
    if preference is not None:
        return preference
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def check_blip_available():
    """transformers is an optional dependency (see requirements-blip.txt) -
    only needed for --generate-alt blip."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def get_captioner(use_gpu):
    """Lazily load the offline BLIP captioning model (only when needed)."""
    global _caption_model, _caption_processor, _caption_device
    if _caption_model is None:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise SystemExit(
                "Пакет transformers не установлен (нужен для --generate-alt blip). "
                "Поставьте: conda run -n pdf2html python -m pip install -r requirements-blip.txt "
                "- либо используйте --generate-alt lmstudio или --generate-alt off."
            )

        print(
            f"Loading image captioning model '{CAPTION_MODEL_NAME}' "
            "(first run downloads the weights)...",
            file=sys.stderr,
        )
        _caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
        _caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
        _caption_device = "cuda" if use_gpu else "cpu"
        _caption_model.to(_caption_device)
    return _caption_processor, _caption_model, _caption_device


def caption_image_blip(image_bytes, use_gpu):
    """Return a short English description of the image, or None on failure."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    # Skip tiny decorative images (bullets, rules, icons) - captioning them is noise.
    if image.width < 32 or image.height < 32:
        return None

    processor, model, device = get_captioner(use_gpu)
    inputs = processor(image, return_tensors="pt").to(device)
    # no_repeat_ngram_size guards against BLIP's greedy decoding getting stuck
    # in a repetition loop on flat/low-detail images (e.g. "menu menu menu...").
    output = model.generate(**inputs, max_new_tokens=30, no_repeat_ngram_size=3)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip() or None


def caption_image_lmstudio(image_bytes, base_url, model_name):
    """Caption an image via a local LM Studio server (OpenAI-compatible API)."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    if image.width < 32 or image.height < 32:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in one short factual sentence, "
                            "suitable for an HTML alt attribute. No preamble, "
                            "no quotes, just the description."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 60,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LMSTUDIO_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
        caption = result["choices"][0]["message"]["content"].strip()
        return caption or None
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as exc:
        print(f"Не удалось получить описание от LM Studio: {exc}", file=sys.stderr)
        return None


def caption_image(image_bytes, backend, use_gpu, lmstudio_url, lmstudio_model):
    if backend == "lmstudio":
        return caption_image_lmstudio(image_bytes, lmstudio_url, lmstudio_model)
    return caption_image_blip(image_bytes, use_gpu)


def check_lmstudio(base_url, model_name):
    """Fail fast with a clear message if LM Studio isn't reachable/ready."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return False, (
            f"Не удалось подключиться к LM Studio по адресу {base_url} ({exc}). "
            "Убедитесь, что в LM Studio включён Local Server и загружена "
            "vision-модель."
        )
    available = [item.get("id") for item in data.get("data", [])]
    if not model_name:
        return False, (
            "Укажите --lmstudio-model. Сейчас в LM Studio доступны: "
            + (", ".join(available) if available else "(нет моделей)")
        )
    if available and model_name not in available:
        return False, f"Модель '{model_name}' не найдена в LM Studio. Доступны: {', '.join(available)}"
    return True, ""


def get_ocr_reader(use_gpu):
    """Lazily load the offline OCR model (only when a garbled text layer is hit)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        print(
            f"Loading OCR model for languages {OCR_LANGUAGES} (gpu={use_gpu}) "
            "(first run downloads the weights)...",
            file=sys.stderr,
        )
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu)
    return _ocr_reader


def vertical_overlap_ratio(a_y0, a_y1, b_y0, b_y1):
    overlap = min(a_y1, b_y1) - max(a_y0, b_y0)
    if overlap <= 0:
        return 0.0
    union = max(a_y1, b_y1) - min(a_y0, b_y0)
    return overlap / union if union > 0 else 0.0


def split_merged_row(row, gutter=None):
    """A row built by vertical-overlap merging can accidentally fuse two
    DIFFERENT columns' entries that happen to share a baseline (e.g. a page
    laid out in two text columns on a shared line grid) instead of
    genuinely being one entry's title plus a far-off trailing page number
    (e.g. a table-of-contents title separated from "27" by a sparse dot
    leader). Both look identical as "one wide gap inside the row" - the
    only reliable way to tell them apart is what's on the far side of that
    gap: a bare page number is the normal single-column case and stays
    merged; a whole second title means this was really two columns, so it
    gets split off as its own row (recursively, in case a row fused three
    or more columns).

    gutter, if given, is a (left, right) pixel-space interval independently
    derived from real PDF block geometry (see detect_column_boundary_x) -
    when an internal gap in this row actually spans that interval, it is
    trusted as the column split point regardless of its raw size, which is
    what catches a genuine but narrow gutter (e.g. ~11pt in print, far
    below COLUMN_SPLIT_MIN_GAP) that the size/ratio heuristic alone would
    leave fused. Without a gutter (or when no gap spans it), the original
    size/ratio heuristic is the only signal.
    """
    boxes = sorted(row["boxes"], key=lambda b: b["x0"])
    if len(boxes) < 2:
        return [row]

    gaps = [boxes[i]["x0"] - boxes[i - 1]["x1"] for i in range(1, len(boxes))]

    split_at = None
    if gutter is not None:
        gutter_left, gutter_right = gutter
        for i, gap in enumerate(gaps):
            if gap > 0 and boxes[i]["x1"] <= gutter_right and boxes[i + 1]["x0"] >= gutter_left:
                split_at = i + 1
                break

    if split_at is None:
        max_gap = max(gaps)
        candidate = gaps.index(max_gap) + 1
        if max_gap < COLUMN_SPLIT_MIN_GAP:
            return [row]
        other_gaps = gaps[:candidate - 1] + gaps[candidate:]
        if other_gaps:
            median_other = sorted(other_gaps)[len(other_gaps) // 2]
            if median_other > 0 and max_gap < median_other * COLUMN_SPLIT_GAP_RATIO:
                return [row]
        split_at = candidate

    right_boxes = boxes[split_at:]
    if is_page_number_like(" ".join(b["text"] for b in right_boxes).strip()):
        return [row]

    left_boxes = boxes[:split_at]
    left_row = {
        "y0": min(b["y0"] for b in left_boxes), "y1": max(b["y1"] for b in left_boxes),
        "boxes": left_boxes,
    }
    right_row = {
        "y0": min(b["y0"] for b in right_boxes), "y1": max(b["y1"] for b in right_boxes),
        "boxes": right_boxes,
    }
    return [left_row] + split_merged_row(right_row, gutter=gutter)


def cluster_rows(line_boxes, gutter=None):
    """Merge raw OCR detections into visual rows by vertical (y-range)
    OVERLAP rather than center proximity or a fixed gap.

    This is what correctly reattaches a table-of-contents title to its page
    number - "Title......." and "27" are split into two boxes because the
    sparse, undotted dot leader between them isn't recognized as text, but
    both boxes span almost the same y-range (the digits are shorter/smaller
    but still sit within the title's height), so they overlap heavily
    despite the huge horizontal gap. Two genuinely different lines at a
    similar height - including two columns running in parallel, e.g. a
    two-column page on a shared line grid - can overlap just as heavily;
    split_merged_row() below resolves that ambiguity afterward.

    The y-overlap merge step itself deliberately does NOT take the column
    gutter into account: a detector on justified text can emit one box per
    WORD instead of per line (the stretched inter-word spacing looks like a
    gap to it), and those words can individually land on either side of a
    gutter computed from unrelated blocks even though they all belong to
    the same, single-column printed line - filtering merges by gutter side
    at this stage previously fragmented ordinary justified paragraphs into
    a handful of disconnected words. gutter is only ever consulted
    afterward, in split_merged_row(), once rows are already fully formed.
    """
    if not line_boxes:
        return []
    ordered = sorted(line_boxes, key=lambda b: (b["y0"] + b["y1"]) / 2)
    rows = []
    for box in ordered:
        merged = False
        for row in rows:
            if vertical_overlap_ratio(box["y0"], box["y1"], row["y0"], row["y1"]) >= ROW_MIN_Y_OVERLAP_RATIO:
                row["boxes"].append(box)
                row["y0"] = min(row["y0"], box["y0"])
                row["y1"] = max(row["y1"], box["y1"])
                merged = True
                break
        if not merged:
            rows.append({"y0": box["y0"], "y1": box["y1"], "boxes": [box]})

    split_rows = []
    for row in rows:
        split_rows.extend(split_merged_row(row, gutter=gutter))

    result = []
    for row in split_rows:
        boxes = sorted(row["boxes"], key=lambda b: b["x0"])
        text = " ".join(b["text"] for b in boxes if b["text"].strip())
        if not text.strip():
            continue
        result.append({
            "x0": min(b["x0"] for b in boxes),
            "x1": max(b["x1"] for b in boxes),
            "y0": min(b["y0"] for b in boxes),
            "y1": max(b["y1"] for b in boxes),
            "text": text,
        })
    result.sort(key=lambda r: (r["y0"] + r["y1"]) / 2)
    return result


def find_callout_boxes(page, page_dict):
    """Detect rectangular decorative regions (e.g. a margin callout/tip box)
    from the page's own vector graphics (border lines/fills) - independent
    of the (possibly broken) text layer, and far more reliable than trying
    to infer columns from OCR text geometry: some books' callout boxes share
    the SAME baseline grid as the main column, so a box's side text can
    align almost exactly with a main-column line at the same height, which
    defeats any purely position-based text heuristic.

    A box's border is typically drawn as several separate thin line/fill
    segments; segments that touch or nearly touch are merged into one
    region. Returns a list of pymupdf.Rect, filtered to plausible partial-
    width boxes - a full-width rect (e.g. a header underline) or a tiny
    stray mark isn't a callout box - and with anything that's mostly a
    frame drawn around an embedded image (e.g. a screenshot border)
    excluded, since OCR'ing that region would just read the picture's own
    on-screen text as if it were callout prose.
    """
    drawings = page.get_drawings()
    page_width = page.rect.width
    rects = [pymupdf.Rect(d["rect"]) for d in drawings if d.get("rect") is not None]
    rects = [r for r in rects if r.width >= 0.5 or r.height >= 0.5]
    if not rects:
        return []

    tolerance = 3
    n = len(rects)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        a = rects[i]
        expanded = pymupdf.Rect(a.x0 - tolerance, a.y0 - tolerance, a.x1 + tolerance, a.y1 + tolerance)
        for j in range(i + 1, n):
            if expanded.intersects(rects[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root in groups:
            groups[root] |= rects[i]
        else:
            groups[root] = pymupdf.Rect(rects[i])

    image_rects = [pymupdf.Rect(b["bbox"]) for b in page_dict["blocks"] if b["type"] == 1]

    boxes = []
    for rect in groups.values():
        if rect.width >= page_width * CALLOUT_MAX_WIDTH_RATIO:
            continue
        if rect.height < CALLOUT_MIN_SIZE_PT or rect.width < CALLOUT_MIN_SIZE_PT:
            continue
        box_area = rect.width * rect.height
        is_image_frame = False
        for img_rect in image_rects:
            intersection = rect & img_rect
            if intersection.is_empty:
                continue
            overlap_area = intersection.width * intersection.height
            if box_area > 0 and overlap_area / box_area > CALLOUT_IMAGE_OVERLAP_MAX_RATIO:
                is_image_frame = True
                break
        if is_image_frame:
            continue
        boxes.append(rect)
    return boxes


LEADING_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}\s")

# Real bullet glyphs seen across sampled books (► is common in Cyrillic
# technical books; the others are the usual Unicode bullet family) - a
# bare "-"/"*" is deliberately excluded, since those are far too likely to
# be an ordinary hyphen/em-dash-like usage or a code comment marker rather
# than a genuine list item.
LIST_BULLET_CHARS = "►•‣▪●○◦"
LIST_BULLET_RE = re.compile(rf"^([{re.escape(LIST_BULLET_CHARS)}])\s+(\S.*)$")
# A numbered list marker needs '.'/')' right after the digit(s) plus real
# content after that - "28 Chapter Name" (TOC-style, no punctuation) is a
# different pattern already handled by LEADING_PAGE_NUMBER_RE elsewhere.
LIST_NUMBER_RE = re.compile(r"^(\d{1,2})[.)]\s+(\S.*)$")


def strip_list_marker(text):
    """(item_text_without_marker, ordered) if text opens with a bullet or
    numbered-list marker, else None."""
    match = LIST_BULLET_RE.match(text)
    if match:
        return match.group(2), False
    match = LIST_NUMBER_RE.match(text)
    if match:
        return match.group(2), True
    return None


def _finalize_ocr_paragraph(current, text, row_count, median_height):
    """A paragraph that never merged with a neighboring row (row_count == 1)
    and is noticeably taller than the page's median row height is very
    likely a heading, not a coincidentally short paragraph - font size
    metadata doesn't survive OCR, so height is the only proxy available.
    Deliberately conservative (single row, short, no terminal punctuation):
    missing a real heading is far less damaging than turning an ordinary
    short paragraph into one.
    """
    para = {**current, "text": text}
    height = current["y1"] - current["y0"]
    stripped = text.rstrip()
    looks_like_heading = (
        row_count == 1
        and median_height > 0
        and height >= median_height * HEADING_OCR_HEIGHT_RATIO_H3
        and len(text) <= HEADING_OCR_MAX_CHARS
        and stripped
        and stripped[-1] not in PARAGRAPH_TERMINATOR_CHARS
        and not is_numeric_heavy(text)
        # A leading or trailing page number means this is a table-of-
        # contents/index entry (see reflow_rows_into_paragraphs), never a
        # section title, even if it happens to sit taller than the page's
        # median row (e.g. a bolded top-level entry in a nested TOC).
        and TRAILING_PAGE_NUMBER_RE.search(stripped) is None
        and LEADING_PAGE_NUMBER_RE.match(stripped) is None
    )
    if looks_like_heading:
        para["heading_level"] = 2 if height >= median_height * HEADING_OCR_HEIGHT_RATIO_H2 else 3
    return para


def reflow_rows_into_paragraphs(rows):
    """Merge consecutive rows into paragraphs: close vertical spacing plus a
    row whose text doesn't already "look finished" (no terminal punctuation,
    no trailing page number) continues the paragraph; otherwise a new one
    starts. The trailing-page-number check is what keeps table-of-contents
    entries as separate one-line paragraphs instead of one giant run-on
    block, even though they're single-spaced like a normal paragraph - the
    leading-page-number check does the same for a table laid out with the
    number BEFORE the title ("28 Chapter Name") instead of after: a row
    that opens with a page number is always a new entry, never a
    continuation of the previous one.
    """
    if not rows:
        return []
    heights = [r["y1"] - r["y0"] for r in rows] or [10]
    median_height = sorted(heights)[len(heights) // 2] or 10
    gap_threshold = median_height * PARAGRAPH_GAP_RATIO

    paragraphs = []
    current_text = rows[0]["text"]
    current = dict(rows[0])
    current_row_count = 1
    for row in rows[1:]:
        gap = row["y0"] - current["y1"]
        prev_text = current_text.rstrip()
        prev_height = current["y1"] - current["y0"]
        # A short, isolated, noticeably taller row (a section title's own
        # font is bigger than body text) never continues into the next row
        # even without terminal punctuation - unlike an ordinary paragraph
        # opener, a title is never "unfinished". Without this, a heading
        # sitting right above body text with normal paragraph spacing gets
        # silently absorbed into that paragraph before heading detection
        # (see _finalize_ocr_paragraph) ever runs on it.
        prev_looks_like_heading = (
            current_row_count == 1
            and median_height > 0
            and prev_height >= median_height * HEADING_OCR_HEIGHT_RATIO_H3
            and len(prev_text) <= HEADING_OCR_MAX_CHARS
            and not is_numeric_heavy(prev_text)
            and TRAILING_PAGE_NUMBER_RE.search(prev_text) is None
        )
        prev_finished = (
            not prev_text
            or prev_text[-1] in PARAGRAPH_TERMINATOR_CHARS
            or TRAILING_PAGE_NUMBER_RE.search(prev_text) is not None
            or prev_looks_like_heading
        )
        next_starts_new_entry = (
            bool(LEADING_PAGE_NUMBER_RE.match(row["text"]))
            or strip_list_marker(row["text"]) is not None
        )
        if gap <= gap_threshold and not prev_finished and not next_starts_new_entry:
            current_text = prev_text + " " + row["text"]
            current["y1"] = row["y1"]
            current["x1"] = max(current["x1"], row["x1"])
            current_row_count += 1
        else:
            paragraphs.append(_finalize_ocr_paragraph(current, current_text, current_row_count, median_height))
            current_text = row["text"]
            current = dict(row)
            current_row_count = 1
    paragraphs.append(_finalize_ocr_paragraph(current, current_text, current_row_count, median_height))
    return paragraphs


def ocr_region_paragraphs(reader, image, offset_x=0.0, offset_y=0.0, gutter=None):
    """OCR an already-rendered (and already-masked, if relevant) region
    image and return reflowed paragraphs. offset_x/offset_y translate a
    cropped sub-image's local coordinates back into full-page pixel space,
    so a callout box's paragraphs are directly comparable (e.g. for
    header/footer classification) to the main flow's. gutter (already in
    this same pixel space) is a genuine column gutter, if one was found for
    the page - reading order needs the left column's rows fully
    top-to-bottom before the right column's, not interleaved by height, or
    reflow would stitch rows from different columns back together even
    after cluster_rows kept them as separate rows. Column-tagged paragraphs
    also carry a "column" index and, for rows indented past their column's
    own leftmost row, an "indent_em" hint - both purely for HTML rendering
    (see group_blocks_for_layout); md/txt ignore them since reading order
    is already correct without them.
    """
    import numpy as np

    raw_results = reader.readtext(np.array(image), detail=1, paragraph=False)
    line_boxes = []
    for bbox, text, _confidence in raw_results:
        text = text.strip()
        if not text:
            continue
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        line_boxes.append({
            "x0": min(xs) + offset_x, "x1": max(xs) + offset_x,
            "y0": min(ys) + offset_y, "y1": max(ys) + offset_y,
            "text": text,
        })

    rows = cluster_rows(line_boxes, gutter=gutter)

    if gutter is None:
        column_groups = [(None, rows)]
    else:
        gutter_left, gutter_right = gutter
        left = sorted((r for r in rows if r["x0"] < gutter_right), key=lambda r: (r["y0"] + r["y1"]) / 2)
        right = sorted((r for r in rows if r["x0"] >= gutter_right), key=lambda r: (r["y0"] + r["y1"]) / 2)
        column_groups = [(idx, group) for idx, group in enumerate((left, right)) if group]

    paragraphs = []
    for col_index, group_rows in column_groups:
        if not group_rows:
            continue
        group_min_x0 = min(r["x0"] for r in group_rows)
        for para in reflow_rows_into_paragraphs(group_rows):
            text = clean_text(para["text"])
            if not text.strip():
                continue
            entry = {"text": text, "y0": para["y0"], "y1": para["y1"]}
            if para.get("heading_level"):
                entry["heading_level"] = para["heading_level"]
            if col_index is not None:
                entry["column"] = col_index
                indent_em = round((para["x0"] - group_min_x0) / OCR_INDENT_PX_PER_EM, 1)
                if OCR_INDENT_MIN_EM <= indent_em <= OCR_INDENT_MAX_EM:
                    entry["indent_em"] = indent_em
            paragraphs.append(entry)
    return paragraphs


def ocr_page_regions(page, page_dict, use_gpu):
    """Render a page to an image and OCR it, for documents whose text layer
    is broken, reconstructing rows/paragraphs from the raw detections (see
    cluster_rows/reflow_rows_into_paragraphs).

    Embedded raster images AND any detected callout boxes (find_callout_
    boxes) are blanked out of the MAIN pass and OCR'd separately instead:
    images are already extracted/captioned by extract_image_block(), and a
    callout box's text needs its own isolated OCR pass rather than being
    read together with the main column - see find_callout_boxes() for why
    that matters. Leaving either in the main pass would have OCR read
    on-screen UI chrome / a sidebar's unrelated text as if it were body text.

    Returns (paragraphs, page_height_px) where each paragraph is
    {"text": str, "y0": float, "y1": float, "region": int} in the same pixel
    space as page_height_px (for header/footer position classification).
    region 0 is the main flow; region >= 1 is a callout box.
    """
    from PIL import Image, ImageDraw

    reader = get_ocr_reader(use_gpu)
    scale = OCR_RENDER_DPI / 72.0
    callout_boxes = find_callout_boxes(page, page_dict)
    boundary_pt = detect_column_boundary_x(page_dict, page.rect.height)
    gutter = (boundary_pt[0] * scale, boundary_pt[1] * scale) if boundary_pt is not None else None

    pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
    main_image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(main_image)
    for block in page_dict["blocks"]:
        if block["type"] == 1:
            x0, y0, x1, y1 = block["bbox"]
            draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill="white")
    for box in callout_boxes:
        draw.rectangle([box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale], fill="white")

    paragraphs = []
    for para in ocr_region_paragraphs(reader, main_image, gutter=gutter):
        para["region"] = 0
        paragraphs.append(para)

    for box in callout_boxes:
        clip_pixmap = page.get_pixmap(clip=box, dpi=OCR_RENDER_DPI)
        clip_image = Image.open(io.BytesIO(clip_pixmap.tobytes("png"))).convert("RGB")
        for para in ocr_region_paragraphs(reader, clip_image, box.x0 * scale, box.y0 * scale):
            para["region"] = 1
            paragraphs.append(para)

    return paragraphs, main_image.height


def is_text_garbled(raw_text_blocks):
    """Detect a broken font/ToUnicode mapping: the text layer still yields
    *a* character per glyph, but with no real CMap those characters are
    meaningless control codes rather than actual letters - the classic
    symptom of the "wrong encoding" problem this script exists to work
    around.

    Must run on RAW (pre-clean_text) extracted text: clean_text() strips
    exactly these control characters, so checking after cleaning would
    always see zero junk.
    """
    combined = "".join(raw_text_blocks)
    if len(combined) < GARBLED_TEXT_MIN_LENGTH:
        return False
    junk = sum(1 for ch in combined if ord(ch) < 0x20 and ch not in "\n\r\t")
    return (junk / len(combined)) > GARBLED_TEXT_THRESHOLD


def is_in_margin_zone(y0, y1, page_height):
    top_limit = page_height * MARGIN_ZONE_RATIO
    bottom_limit = page_height * (1 - MARGIN_ZONE_RATIO)
    return y1 <= top_limit or y0 >= bottom_limit


def is_margin_text(y0, y1, page_height, text):
    """Short text confined to the page's top/bottom margin: a running header,
    chapter title, or lone page number rather than a body paragraph."""
    if not text or len(text) > MARGIN_TEXT_MAX_LEN:
        return False
    return is_in_margin_zone(y0, y1, page_height)


FOOTNOTE_MARKER_RE = re.compile(r"^(\d{1,2})\s+\S")


def collect_superscript_digit_markers(page_dict):
    """Digit strings that appear as a genuine superscript span (flags&1)
    somewhere on the page - a footnote at the bottom of the page starts
    with its own reference number restated in normal size, and that
    number matching one of these is what tells it apart from an ordinary
    running footer/page number: is_margin_text()'s generic "short text
    near the edge" heuristic alone can't (a short footnote and a running
    footer look geometrically identical to it), and mistaking one for the
    other silently drops the footnote's content into page metadata (see
    extract_page_blocks)."""
    markers = set()
    for block in page_dict["blocks"]:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("flags", 0) & 1:
                    text = span.get("text", "").strip()
                    if text.isdigit():
                        markers.add(text)
    return markers


def _blocks_run_in_parallel(left, right):
    """The real signature of two columns running side by side is that most
    blocks on one side sit at the same height as SOME block on the other
    side - not just that the two sides' overall y-ranges happen to
    overlap. A title/colophon page with varied indentation (a centered
    title block, then a wide byline, then a run of body blocks, then a
    narrower imprint block near the bottom) can produce a big x0 gap and
    an overlapping min/max y-range too, purely by coincidence, without any
    of its blocks actually being side-by-side with another at the same
    height - checking each block for a same-height counterpart catches
    that difference.
    """
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    matches = 0
    for block in smaller:
        b_y0, b_y1 = block["bbox"][1], block["bbox"][3]
        for other in larger:
            o_y0, o_y1 = other["bbox"][1], other["bbox"][3]
            if min(b_y1, o_y1) - max(b_y0, o_y0) > 0:
                matches += 1
                break
    return (matches / len(smaller)) >= COLUMN_SIBLING_MATCH_MIN_RATIO


def detect_column_boundary_x(page_dict, page_height):
    """Look at the page's own text-block bounding boxes for a genuine
    column boundary - independent of (and far more reliable than) the OCR
    text this function's caller is about to produce, because glyph
    positions come from font metrics and stay valid even when a broken
    ToUnicode CMap makes the extracted characters meaningless.

    A single-column page (even a table-of-contents with a title and its
    page number far apart on the same line) has each *block* spanning
    close to the full text width - title and number are one continuous
    run in the content stream. A genuinely multi-column page has separate,
    narrower blocks per column, so their bounding boxes cluster into two
    clearly-separated x0 groups instead. Running headers/footers are
    excluded first since their bboxes are often oddly shaped (e.g. one
    wide block spanning both a page number and a chapter title) and would
    just add noise.

    Returns a (left, right) PDF-point interval - the true empty gutter
    between the two columns' blocks, NOT their midpoint, since a block's
    own width is not symmetric around any single split point - or None for
    what looks like a single-column page.
    """
    text_blocks = [
        b for b in page_dict["blocks"]
        if b["type"] == 0 and not is_in_margin_zone(b["bbox"][1], b["bbox"][3], page_height)
    ]
    if len(text_blocks) < 4:
        return None
    ordered = sorted(text_blocks, key=lambda b: b["bbox"][0])
    gaps = [(ordered[i]["bbox"][0] - ordered[i - 1]["bbox"][0], i) for i in range(1, len(ordered))]
    max_gap, split_at = max(gaps)
    if max_gap < COLUMN_BOUNDARY_MIN_GAP_PT:
        return None
    other_gaps = [g for g, i in gaps if i != split_at]
    if other_gaps:
        median_other = sorted(other_gaps)[len(other_gaps) // 2]
        if median_other > 0 and max_gap < median_other * COLUMN_SPLIT_GAP_RATIO:
            return None

    left, right = ordered[:split_at], ordered[split_at:]
    # A genuine second column can be a single block of continuous running
    # text (one block flowing down 24 lines, say) rather than several - the
    # real protection against false positives (a sequential single-column
    # page with varied indentation) is _blocks_run_in_parallel below, not
    # the block count on either side.
    if not left or not right:
        return None
    if not _blocks_run_in_parallel(left, right):
        return None

    gutter_left = max(b["bbox"][2] for b in left)
    gutter_right = min(b["bbox"][0] for b in right)
    if gutter_right <= gutter_left:
        return None
    return (gutter_left, gutter_right)


def is_page_number_like(text):
    """A margin item that's purely a page number - unlike a running head
    (chapter/section title), it legitimately changes on every page, so
    dedup_running_heads() never suppresses it."""
    return bool(PAGE_NUMBER_TOKEN_RE.match(text.strip()))


LEADING_MARGIN_NUMBER_RE = re.compile(r"^(\d{1,4})\s+(\S.*)$")
TRAILING_MARGIN_NUMBER_RE = re.compile(r"^(.*\S)\s+(\d{1,4})$")


def split_margin_number(text):
    """A margin OCR paragraph sometimes glues a running head and its page
    number into one string ("8 Содержание" / "Содержание 9") since both are
    short and land close together - reflow_rows_into_paragraphs() has no
    reason to keep them apart. Split them back into separate margin items
    so dedup_running_heads() can recognize a repeated head regardless of
    which page number happens to be attached to it that page.
    """
    text = text.strip()
    match = LEADING_MARGIN_NUMBER_RE.match(text)
    if match:
        return [match.group(1), match.group(2)]
    match = TRAILING_MARGIN_NUMBER_RE.match(text)
    if match:
        return [match.group(1), match.group(2)]
    return [text]


def is_numeric_heavy(text):
    """Flags OCR paragraphs that are mostly digits - typically a table-of-
    contents page-number column that got separated from its titles (see
    README for why this can't be reliably re-paired)."""
    alnum = [ch for ch in text if ch.isalnum()]
    if len(alnum) < NUMERIC_PARAGRAPH_MIN_CHARS:
        return False
    digits = sum(1 for ch in alnum if ch.isdigit())
    return (digits / len(alnum)) >= NUMERIC_PARAGRAPH_RATIO


def clean_text(text):
    """Normalize extracted text so it is well-formed, readable UTF-8.

    Documents frequently reference fonts with broken or missing ToUnicode
    CMaps. The text layer still maps such glyphs to *something*, but that
    something is often a Private-Use-Area codepoint with no real meaning
    (renders as tofu boxes) - those are stripped. Ligatures and other
    compatibility characters are folded to their plain equivalents via
    NFKC. Hyphenated line-wraps ("бо- лее") are rejoined into whole words.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text
        if not (0xE000 <= ord(ch) <= 0xF8FF)
        and not (ord(ch) < 0x20 and ch not in "\n\t")
    )
    text = HYPHEN_WRAP_RE.sub("", text)
    return text


def extract_block_raw_text(block):
    lines = []
    for line in block.get("lines", []):
        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
        if line_text.strip():
            lines.append(line_text)
    return " ".join(lines)


def extract_block_text(block):
    return clean_text(extract_block_raw_text(block))


def _span_href(span, link_rects):
    """The href of whichever link (see find_page_links) this span's own
    bbox mostly sits inside, or None - a link's "from" rect is normally
    sized to cover exactly the linked text, so most-of-the-span-inside is
    a reliable match without needing exact bbox equality."""
    if not link_rects:
        return None
    bbox = span.get("bbox")
    if not bbox:
        return None
    span_rect = pymupdf.Rect(bbox)
    span_area = span_rect.width * span_rect.height
    if span_area <= 0:
        return None
    for rect, href in link_rects:
        overlap = span_rect & rect
        if overlap.is_empty:
            continue
        if (overlap.width * overlap.height) / span_area > 0.5:
            return href
    return None


def extract_block_runs(block, link_rects=None):
    """Per-span (text, bold, italic, href) runs for a block, adjacent runs
    of identical style+link merged - mirrors extract_block_raw_text's line
    filtering/joining exactly (same non-blank lines, same " " between
    them) so the concatenation of these runs' text matches extract_block_
    text()'s plain string, just with style/link info preserved alongside
    instead of discarded. link_rects (see find_page_links) is optional -
    without it every run's href is None."""
    kept_lines = [
        line for line in block.get("lines", [])
        if "".join(s.get("text", "") for s in line.get("spans", [])).strip()
    ]
    raw_runs = []
    for i, line in enumerate(kept_lines):
        if i > 0:
            raw_runs.append([" ", False, False, False, None])
        for span in line.get("spans", []):
            text = span.get("text", "")
            if not text:
                continue
            flags = span.get("flags", 0)
            href = _span_href(span, link_rects)
            raw_runs.append([text, bool(flags & 2 ** 4), bool(flags & 2), bool(flags & 1), href])

    merged = []
    for text, bold, italic, superscript, href in raw_runs:
        if (
            merged
            and merged[-1][1] == bold
            and merged[-1][2] == italic
            and merged[-1][3] == superscript
            and merged[-1][4] == href
        ):
            merged[-1][0] += text
        else:
            merged.append([text, bold, italic, superscript, href])
    return merged


def block_emphasis_runs(block, link_rects=None):
    """None for a uniformly-plain, unlinked block (the common case - not
    worth the extra JSON/render-time work), or a list of {"text", "bold",
    "italic", "superscript", "href"} runs when the block genuinely mixes/
    contains bold/italic/superscript spans or overlaps a real PDF link -
    each run's text cleaned the same way clean_code_text() cleans a code
    line (NFKC + strip PUA/control), skipping HYPHEN_WRAP_RE since a
    hyphen at a style-run boundary is rare and unwrapping across runs
    would need cross-run lookahead this function deliberately doesn't do.
    """
    runs = extract_block_runs(block, link_rects)
    if not any(bold or italic or superscript or href for _, bold, italic, superscript, href in runs):
        return None
    cleaned = [
        {"text": clean_code_text(text), "bold": bold, "italic": italic, "superscript": superscript, "href": href}
        for text, bold, italic, superscript, href in runs
    ]
    return [r for r in cleaned if r["text"]]


LISTING_CAPTION_RE = re.compile(r"^(?:Лістинг|Листинг)\s+\d")
MONOSPACE_FONT_HINTS = ("courier", "consolas", "mono", "menlo", "cascadia", "roboto mono")


def clean_code_text(text):
    """Like clean_text(), but for a code block: preserves newlines/leading
    whitespace (indentation IS the content) and skips HYPHEN_WRAP_RE - a
    stray "-\\n" in code is far more likely a real hyphen/operator at a
    line break than a hyphenated word wrap, and rejoining it would corrupt
    the code."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in text
        if not (0xE000 <= ord(ch) <= 0xF8FF)
        and not (ord(ch) < 0x20 and ch not in "\n\t")
    )


def extract_block_code_raw_text(block):
    """Like extract_block_raw_text(), but joins lines with '\\n' instead of
    a single space and never strips a line - a prose block's line breaks
    are just wrapping and don't matter, but in code they (and the leading
    whitespace PyMuPDF preserves as literal space characters) are the
    difference between readable code and a useless run-on string."""
    return "\n".join(
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    )


def dominant_font_name(block):
    """Character-weighted most common font name across a block's spans -
    used to recognize a monospace-flagged code listing (see
    MONOSPACE_FONT_HINTS)."""
    counts = {}
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "")
            if not text.strip():
                continue
            name = span.get("font", "")
            counts[name] = counts.get(name, 0) + len(text)
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def is_monospace_font_name(name):
    lowered = (name or "").lower()
    return any(hint in lowered for hint in MONOSPACE_FONT_HINTS)


def block_font_stats(block):
    """Character-weighted average font size and bold fraction for a text
    block's spans - a block usually has one dominant size/weight, but
    weighting by character count (rather than just averaging span sizes)
    keeps a short trailing superscript or footnote marker from skewing the
    result of an otherwise uniform line."""
    total_chars = 0
    size_sum = 0.0
    bold_chars = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "")
            n = len(text)
            if n == 0:
                continue
            total_chars += n
            size_sum += span.get("size", 0.0) * n
            if span.get("flags", 0) & 2 ** 4:
                bold_chars += n
    if total_chars == 0:
        return 0.0, 0.0
    return size_sum / total_chars, bold_chars / total_chars


def compute_body_font_size(page_dict, page_height):
    """The page's dominant (character-weighted modal) body font size, used
    as the baseline headings are measured against - only trusted once seen
    across a substantial amount of text (HEADING_BODY_SIZE_MIN_CHARS); a
    title/colophon page has no real "body" to compare against, and a wrong
    guess there would misclassify the whole page instead of nothing."""
    sizes = {}
    for block in page_dict["blocks"]:
        if block["type"] != 0:
            continue
        if is_in_margin_zone(block["bbox"][1], block["bbox"][3], page_height):
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                size = round(span.get("size", 0.0), 1)
                sizes[size] = sizes.get(size, 0) + len(text)
    if not sizes:
        return None
    size, count = max(sizes.items(), key=lambda item: item[1])
    if count < HEADING_BODY_SIZE_MIN_CHARS:
        return None
    return size


def classify_heading_level(text, block_size, bold_ratio, body_size):
    """None, or a heading level (2/3) for a non-margin text block, based on
    its font size relative to the page's body size (see
    compute_body_font_size) - or, absent a size difference, a block that's
    fully bold, close to body size, and short (a common styling for minor
    subheadings that don't get a larger point size)."""
    if body_size is None or block_size <= 0:
        return None
    if len(text) > HEADING_MAX_CHARS:
        return None
    stripped = text.rstrip()
    if not stripped or stripped[-1] in PARAGRAPH_TERMINATOR_CHARS:
        return None
    ratio = block_size / body_size
    if ratio >= HEADING_SIZE_RATIO_H2:
        return 2
    if ratio >= HEADING_SIZE_RATIO_H3:
        return 3
    if (
        bold_ratio >= HEADING_BOLD_RATIO_MIN
        and abs(ratio - 1.0) < HEADING_BOLD_SIZE_TOLERANCE
        and len(text) <= HEADING_OCR_MAX_CHARS
    ):
        return 3
    return None


def dedup_running_heads(ordered_pages):
    """Suppress a running head (a non-page-number margin item, e.g. a
    chapter/section title) that's identical to the one on the immediately
    preceding page - repeating it on every single page is just noise once
    it's already been shown. A page number is never suppressed since it
    genuinely changes each page.

    Runs at RENDER time over the full in-order page sequence (not at
    extraction time), so it gives the same result whether all pages were
    produced in one run or resumed piecemeal across several, and works
    identically for every --out-format since it only touches the shared
    "margin_display" field each renderer reads. Mutates each page dict in
    place; the original "margin" is left untouched for future re-renders.
    """
    previous_heads = set()
    for page_data in ordered_pages:
        margin = page_data.get("margin", [])
        heads = [m for m in margin if not is_page_number_like(m)]
        page_data["margin_display"] = [
            m for m in margin if is_page_number_like(m) or m not in previous_heads
        ]
        previous_heads = set(heads)
    return ordered_pages


# A URL that's visible as plain text but has no PDF link object behind it
# (common when a document was scanned/OCR'd, or the source just didn't
# bother making it clickable) - matched at RENDER time from whatever text
# each path already extracted, so it works identically for both the clean
# text-layer path (page.get_links() catches most but not all real links -
# see find_page_links) and the OCR fallback path (no PDF link objects
# survive a broken text layer, but the URL text itself often does).
BARE_URL_RE = re.compile(r"(https?://[^\s<>\"'«»]+|www\.[^\s<>\"'«»]+\.[a-zA-Z]{2,}[^\s<>\"'«»]*)")


def _linkify_url_target(matched):
    url = matched.rstrip(").,;:!?»")
    trailing = matched[len(url):]
    href = url if url.startswith("http") else f"https://{url}"
    return url, href, trailing


def linkify_html(text):
    parts = []
    pos = 0
    for m in BARE_URL_RE.finditer(text):
        if m.start() > pos:
            parts.append(escape(text[pos:m.start()]))
        url, href, trailing = _linkify_url_target(m.group(0))
        parts.append(f'<a href="{escape(href)}">{escape(url)}</a>{escape(trailing)}')
        pos = m.end()
    parts.append(escape(text[pos:]))
    return "".join(parts)


def linkify_markdown(text):
    parts = []
    pos = 0
    for m in BARE_URL_RE.finditer(text):
        if m.start() > pos:
            parts.append(escape_markdown_inline(text[pos:m.start()]))
        url, href, trailing = _linkify_url_target(m.group(0))
        parts.append(f"[{url}]({href})")
        parts.append(escape_markdown_inline(trailing))
        pos = m.end()
    parts.append(escape_markdown_inline(text[pos:]))
    return escape_markdown_leading_guard("".join(parts))


def render_runs_html(runs):
    parts = []
    for run in runs:
        text = escape(run["text"])
        if run["bold"] and run["italic"]:
            text = f"<strong><em>{text}</em></strong>"
        elif run["bold"]:
            text = f"<strong>{text}</strong>"
        elif run["italic"]:
            text = f"<em>{text}</em>"
        if run.get("superscript"):
            text = f"<sup>{text}</sup>"
        href = run.get("href")
        if href:
            text = f'<a href="{escape(href)}">{text}</a>'
        parts.append(text)
    return "".join(parts)


def render_block_html(block):
    block_type = block["type"]
    if block_type in ("p", "p_numeric"):
        css_class = "ocr-numbers" if block_type == "p_numeric" else None
        style = f' style="margin-left:{block["indent_em"]}em"' if block.get("indent_em") else ""
        class_attr = f' class="{css_class}"' if css_class else ""
        inner = render_runs_html(block["runs"]) if block.get("runs") else linkify_html(block["text"])
        return f"<p{class_attr}{style}>{inner}</p>"
    if block_type == "h":
        level = block["level"]
        return f"<h{level}>{escape(block['text'])}</h{level}>"
    if block_type == "img":
        return f'<img src="{block["src"]}" alt="{escape(block["alt"])}" loading="lazy">'
    if block_type == "img_placeholder":
        if block["caption"]:
            return f'<p class="image-placeholder">[Изображение: {escape(block["caption"])}]</p>'
        return '<p class="image-placeholder">[Изображение]</p>'
    if block_type == "table":
        header, *body = block["rows"]
        parts = ["<table>", "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr>"]
        for row in body:
            parts.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
        parts.append("</table>")
        return "\n".join(parts)
    if block_type == "code":
        return f"<pre><code>{escape(block['text'])}</code></pre>"
    if block_type == "li":
        # Standalone fallback (a lone list item with no neighbors) - the
        # normal path for consecutive items is render_flow_html, which
        # groups them into one shared <ul>/<ol> so ordered numbering
        # doesn't restart at every single item.
        tag = "ol" if block.get("ordered") else "ul"
        return f"<{tag}><li>{linkify_html(block['text'])}</li></{tag}>"
    raise ValueError(f"unknown block type: {block_type}")


def render_flow_html(blocks):
    """Render a flat block list to HTML parts, grouping any run of
    consecutive "li" blocks (same ordered/unordered kind) into one shared
    <ul>/<ol> - rendering each item independently would restart an ordered
    list's numbering at "1." every time instead of counting up."""
    parts = []
    i, n = 0, len(blocks)
    while i < n:
        block = blocks[i]
        if block["type"] == "li":
            ordered = block.get("ordered", False)
            j = i
            items = []
            while j < n and blocks[j]["type"] == "li" and blocks[j].get("ordered", False) == ordered:
                items.append(blocks[j])
                j += 1
            tag = "ol" if ordered else "ul"
            parts.append(f"<{tag}>")
            parts.extend(f"<li>{linkify_html(item['text'])}</li>" for item in items)
            parts.append(f"</{tag}>")
            i = j
        else:
            parts.append(render_block_html(block))
            i += 1
    return parts


def group_blocks_for_layout(blocks):
    """Split a page's flat block list into render segments: consecutive
    blocks with no "column" tag stay a single flowing sequence, while a
    consecutive run of column-tagged blocks (see ocr_region_paragraphs)
    becomes one side-by-side group, each column's blocks kept in their own
    relative order. Blocks are already emitted in reading order (one
    column fully, then the next), so grouping by tag alone - without
    touching that order - is enough to lay genuinely parallel columns out
    side by side instead of as one linear stream.

    Returns a list of ("flow", [block, ...]) | ("columns", {index: [block, ...]}).
    """
    segments = []
    i, n = 0, len(blocks)
    while i < n:
        if blocks[i].get("column") is None:
            j = i
            while j < n and blocks[j].get("column") is None:
                j += 1
            segments.append(("flow", blocks[i:j]))
            i = j
        else:
            j = i
            columns = {}
            while j < n and blocks[j].get("column") is not None:
                columns.setdefault(blocks[j]["column"], []).append(blocks[j])
                j += 1
            segments.append(("columns", columns))
            i = j
    return segments


def render_page_html(page_data):
    parts = [f'<section class="page" id="page-{page_data["page_number"]}">']
    margin = page_data.get("margin_display", page_data["margin"])
    if margin:
        parts.append(f'<p class="page-meta">{escape(" · ".join(margin))}</p>')
    for kind, payload in group_blocks_for_layout(page_data["blocks"]):
        if kind == "flow":
            parts.extend(render_flow_html(payload))
        else:
            parts.append('<div class="cols">')
            for col_index in sorted(payload):
                parts.append('<div class="col">')
                parts.extend(render_flow_html(payload[col_index]))
                parts.append("</div>")
            parts.append("</div>")
    if page_data["callout"]:
        parts.append('<aside class="callout">')
        parts.extend(f"<p>{escape(t)}</p>" for t in page_data["callout"])
        parts.append("</aside>")
    parts.append("</section>")
    return "\n".join(parts)


def build_html_document(title, body_html):
    parts = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        "<style>",
        "body{font-family:Georgia,'Times New Roman',serif;max-width:860px;"
        "margin:2rem auto;padding:0 1rem;line-height:1.6;color:#222}",
        "img{max-width:100%;height:auto;display:block;margin:1rem auto}",
        ".page{margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid #eee}",
        ".image-placeholder{font-style:italic;color:#666}",
        ".page-meta{font-size:0.8em;color:#999;margin:0 0 0.9em;"
        "text-transform:uppercase;letter-spacing:0.03em}",
        ".ocr-numbers{color:#aaa;font-style:italic;font-size:0.85em}",
        ".callout{border-left:3px solid #ccc;margin:1.2rem 0;padding:0.1rem 1rem;"
        "background:#fafafa;font-size:0.95em;color:#444}",
        ".cols{display:flex;gap:2rem;margin:1rem 0}",
        ".col{flex:1;min-width:0}",
        "@media (max-width:700px){.cols{display:block}.col+.col{margin-top:1rem}}",
        "table{border-collapse:collapse;margin:1rem 0;width:100%;font-size:0.92em}",
        "table th,table td{border:1px solid #ddd;padding:0.4em 0.7em;text-align:left;vertical-align:top}",
        "table th{background:#f5f5f5}",
        "pre{background:#282c34;color:#e6e6e6;padding:1em;border-radius:6px;overflow-x:auto;"
        "font-size:0.88em;line-height:1.5}",
        "pre code{font-family:Consolas,'Courier New',monospace}",
        "ul,ol{margin:0.8rem 0;padding-left:1.6rem}",
        "li{margin:0.25rem 0}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        body_html,
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


MARKDOWN_INLINE_ESCAPE_RE = re.compile(r"([\\`*_\[\]<>&])")
MARKDOWN_LEADING_DIGIT_RE = re.compile(r"^(\d+)\.(\s|$)")


def escape_markdown_inline(text):
    """Escape only the inline syntax characters (emphasis, links, code
    spans, raw HTML/entities) - deliberately WITHOUT the leading-character
    guard escape_markdown() adds, since this is meant to be safe to call
    on a run/fragment that isn't necessarily at the start of a line (see
    render_runs_markdown) - a mid-sentence run that happens to start with
    "-" (e.g. the second half of a bold/plain split on "test-driven")
    must not be treated as if it opened a new line."""
    return MARKDOWN_INLINE_ESCAPE_RE.sub(r"\\\1", text)


def escape_markdown_leading_guard(text):
    """Prefix a backslash if text's own first character would otherwise
    be parsed as block-level Markdown syntax (heading/list marker/ordered-
    list start) when it's actually just the first character of extracted
    prose."""
    if text[:1] in "#-+":
        return "\\" + text
    match = MARKDOWN_LEADING_DIGIT_RE.match(text)
    if match:
        digits = match.group(1)
        return text[:len(digits)] + "\\." + text[len(digits) + 1:]
    return text


def escape_markdown(text):
    """Escape characters CommonMark/GFM treats as syntax (emphasis, links,
    code spans) or raw HTML/entities. Text pulled out of a document is not
    markup - and an unescaped "<tag>" is the dangerous case: a book that
    happens to mention an HTML tag by name (e.g. "<NOFRAMES>") produces a
    literal, unclosed tag in the Markdown source, and most renderers then
    treat *everything after it* as being inside that (never-closed)
    element - for <noframes> specifically, browsers hide that content
    outright, so the rendered page silently ends right there.
    """
    return escape_markdown_leading_guard(escape_markdown_inline(text))


def escape_markdown_image_alt(text):
    # Convert brackets instead of escaping them - a backslash-escaped "]"
    # inside ![alt](src) risks confusing parsers about where alt ends.
    text = text.replace("[", "(").replace("]", ")")
    return re.sub(r"([\\`*_<>&])", r"\\\1", text)


def render_runs_markdown(runs):
    parts = []
    for run in runs:
        text = escape_markdown_inline(run["text"])
        if run["bold"] and run["italic"]:
            text = f"***{text}***"
        elif run["bold"]:
            text = f"**{text}**"
        elif run["italic"]:
            text = f"*{text}*"
        href = run.get("href")
        if href:
            text = f"[{text}]({href})"
        parts.append(text)
    return escape_markdown_leading_guard("".join(parts))


def render_block_markdown(block):
    block_type = block["type"]
    if block_type in ("p", "p_numeric"):
        return render_runs_markdown(block["runs"]) if block.get("runs") else linkify_markdown(block["text"])
    if block_type == "h":
        return ("#" * block["level"]) + " " + escape_markdown(block["text"])
    if block_type == "li":
        # "1." for every ordered item is valid CommonMark - renderers
        # number from the first item's value and increment themselves.
        marker = "1." if block.get("ordered") else "-"
        return f"{marker} {linkify_markdown(block['text'])}"
    if block_type == "img":
        return f'![{escape_markdown_image_alt(block["alt"])}]({block["src"]})'
    if block_type == "img_placeholder":
        caption = escape_markdown(block["caption"]) if block["caption"] else None
        return f'*[Изображение: {caption}]*' if caption else "*[Изображение]*"
    if block_type == "table":
        def cell(text):
            # A literal "|" would otherwise be read as a new column
            # boundary by any Markdown table renderer - on top of the
            # usual inline escaping, it needs its own explicit escape.
            return escape_markdown(text).replace("|", "\\|") or " "
        header, *body = block["rows"]
        lines = ["| " + " | ".join(cell(c) for c in header) + " |"]
        lines.append("|" + "|".join(" --- " for _ in header) + "|")
        for row in body:
            lines.append("| " + " | ".join(cell(c) for c in row) + " |")
        return "\n".join(lines)
    if block_type == "code":
        text = block["text"]
        # Content inside a fenced code block is taken literally by
        # CommonMark - no inline escaping needed, and this is what avoids
        # the raw-HTML-tag risk escape_markdown() exists for elsewhere.
        fence = "~~~" if "```" in text else "```"
        return f"{fence}\n{text}\n{fence}"
    raise ValueError(f"unknown block type: {block_type}")


def render_page_markdown(page_data):
    parts = []
    margin = page_data.get("margin_display", page_data["margin"])
    if margin:
        parts.append(f'*{escape_markdown(" · ".join(margin))}*')
    parts.extend(render_block_markdown(b) for b in page_data["blocks"])
    if page_data["callout"]:
        parts.append("\n".join(f"> {escape_markdown(t)}" for t in page_data["callout"]))
    return "\n\n".join(p for p in parts if p and p.strip())


def build_markdown_document(title, page_mds):
    body = "\n\n---\n\n".join(page_mds)
    return f"# {title}\n\n{body}\n"


def render_block_txt(block):
    block_type = block["type"]
    if block_type in ("p", "p_numeric", "h"):
        return block["text"]
    if block_type == "li":
        return f"  - {block['text']}"
    if block_type == "img":
        return f'[Изображение: {block["alt"]}]' if block["alt"] else "[Изображение]"
    if block_type == "img_placeholder":
        return f'[Изображение: {block["caption"]}]' if block["caption"] else "[Изображение]"
    if block_type == "table":
        rows = block["rows"]
        widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
        lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)
    if block_type == "code":
        return block["text"]
    raise ValueError(f"unknown block type: {block_type}")


def render_page_txt(page_data):
    parts = []
    margin = page_data.get("margin_display", page_data["margin"])
    if margin:
        parts.append(f'[{" · ".join(margin)}]')
    parts.extend(render_block_txt(b) for b in page_data["blocks"])
    if page_data["callout"]:
        parts.append("\n".join(f"    {t}" for t in page_data["callout"]))
    return "\n\n".join(p for p in parts if p and p.strip())


def build_txt_document(title, page_txts):
    underline = "=" * len(title)
    body = "\n\n----------\n\n".join(page_txts)
    return f"{title}\n{underline}\n\n{body}\n"


def extract_image_block(page, block, page_number, image_index, images_dir, save_images,
                         generate_alt, caption_backend, use_gpu, lmstudio_url, lmstudio_model):
    # The raw bytes in block["image"] are the image XObject as stored in the
    # document, BEFORE the placement matrix the page applies to it - for
    # images under a flip/rotate/mirror transform those raw bytes come out
    # upside down or mirrored. Rendering the page's own bbox instead
    # reproduces exactly what a reader sees, transform included.
    bbox = pymupdf.Rect(block["bbox"])
    if bbox.is_empty or bbox.width < 1 or bbox.height < 1:
        return None
    pixmap = page.get_pixmap(clip=bbox, dpi=IMAGE_RENDER_DPI)
    image_bytes = pixmap.tobytes("png")

    caption = None
    if generate_alt:
        caption = caption_image(image_bytes, caption_backend, use_gpu, lmstudio_url, lmstudio_model)

    if save_images:
        filename = f"image_{page_number:04d}_{image_index:04d}.png"
        (images_dir / filename).write_bytes(image_bytes)
        return {"type": "img", "src": f"images/{filename}", "alt": caption or ""}
    return {"type": "img_placeholder", "caption": caption}


def _with_layout_fields(block, para):
    """Carry an OCR paragraph's column/indent hints (see
    ocr_region_paragraphs) onto its extracted block, if present - purely
    additive rendering metadata that md/txt renderers ignore."""
    if para.get("column") is not None:
        block["column"] = para["column"]
    if para.get("indent_em"):
        block["indent_em"] = para["indent_em"]
    return block


def clean_table_rows(raw_rows):
    """page.find_tables()'s raw extract() is noisy in two specific ways
    that would otherwise corrupt a table's meaning if rendered as-is:

    1. It routinely over-segments columns - a genuine 4-column table often
       comes back as 9-12 columns, most of them empty/None in every row
       (whitespace inside a cell misread as a column boundary). Columns
       that are empty in every row carry no information and are dropped.
    2. A wrapped multi-line cell (the source table's row text spans two
       printed lines) comes back as an extra row of its own, with only
       that one column populated and every other cell empty/None. Such a
       lone-value row is folded back into the same column of the
       preceding row instead of appearing as a spurious near-empty row.

    Returns a list of rows (list of str, already through clean_text()),
    or [] if nothing meaningful survives.
    """
    if not raw_rows:
        return []

    def is_empty(value):
        return value is None or not str(value).strip()

    width = max(len(row) for row in raw_rows)
    padded = [list(row) + [None] * (width - len(row)) for row in raw_rows]
    kept_columns = [c for c in range(width) if any(not is_empty(row[c]) for row in padded)]
    if not kept_columns:
        return []

    rows = [
        [
            "" if is_empty(row[c]) else clean_text(str(row[c])).replace("\n", " ").strip()
            for c in kept_columns
        ]
        for row in padded
    ]

    merged = []
    for row in rows:
        non_empty = [i for i, value in enumerate(row) if value]
        if merged and len(non_empty) == 1 and merged[-1][non_empty[0]]:
            i = non_empty[0]
            merged[-1][i] = merged[-1][i] + " " + row[i]
            continue
        if non_empty:
            merged.append(row)
    return _merge_split_columns(merged)


def _merge_split_columns(rows):
    """A third find_tables() quirk clean_table_rows's caller hasn't
    addressed yet: a single logical column sometimes still comes back as
    two ADJACENT surviving columns, because the detector misjudged where
    one cell's text ends and the next begins - e.g. a "sbyte" row's type
    name lands in column 0 while a "byte" row's lands in column 1 of the
    very same logical "type name" column (mutually exclusive per row), or
    a wide cell's text is duplicated verbatim into both of two adjacent
    columns. Two adjacent columns are safely the same logical column if,
    in every row, their values never conflict - either at most one of them
    is non-empty, or both are non-empty and identical.
    """
    if not rows or not rows[0]:
        return rows
    ncols = len(rows[0])
    groups = [[0]]
    for c in range(1, ncols):
        candidate = groups[-1] + [c]
        conflict = any(
            len({row[i] for i in candidate if row[i]}) > 1
            for row in rows
        )
        if conflict:
            groups.append([c])
        else:
            groups[-1] = candidate

    def resolve(row, group):
        for i in group:
            if row[i]:
                return row[i]
        return ""

    return [[resolve(row, group) for group in groups] for row in rows]


def _table_region_is_actually_code(table_rect, page_dict):
    """find_tables() mistakes a code listing's own layout for a table
    surprisingly easily: a line-number gutter reads as one column and the
    wall of code as another. If a "Лістинг/Листинг N.N" caption or a
    monospace-font block sits mostly inside the candidate table's own
    bbox, that candidate isn't a real table - it's a listing, and belongs
    to the code-detection path (see extract_page_blocks) instead."""
    for block in page_dict["blocks"]:
        if block["type"] != 0:
            continue
        block_rect = pymupdf.Rect(block["bbox"])
        block_area = block_rect.width * block_rect.height
        if block_area <= 0:
            continue
        overlap = block_rect & table_rect
        if overlap.is_empty or (overlap.width * overlap.height) / block_area <= 0.5:
            continue
        if is_monospace_font_name(dominant_font_name(block)):
            return True
        if LISTING_CAPTION_RE.match(extract_block_text(block)):
            return True
    return False


def find_page_links(page):
    """External URI and internal cross-page links as (pymupdf.Rect, href)
    pairs. An internal GOTO link resolves to this tool's own "#page-N"
    section anchor (render_page_html already emits id="page-N" for every
    page), so a working table-of-contents/index entry survives the
    conversion instead of just leaving the link's own destination
    unreadable. Other link kinds (launch actions, named destinations) are
    rare and skipped rather than guessed at."""
    try:
        links = page.get_links()
    except Exception:
        return []
    results = []
    for link in links:
        rect = link.get("from")
        if rect is None:
            continue
        kind = link.get("kind")
        if kind == pymupdf.LINK_URI:
            uri = link.get("uri")
            if uri:
                results.append((pymupdf.Rect(rect), uri))
        elif kind == pymupdf.LINK_GOTO:
            target_page = link.get("page")
            if isinstance(target_page, int) and target_page >= 0:
                results.append((pymupdf.Rect(rect), f"#page-{target_page + 1}"))
    return results


def find_page_tables(page, page_dict):
    """Detect real data tables via PyMuPDF's own page.find_tables() - only
    meaningful on a page whose text layer is intact (on an OCR-fallback
    page it tends to mistake a screenshot or a callout box for a table,
    since it works off the same broken text/positions everything else
    struggles with there). Returns a list of (pymupdf.Rect, rows) for
    tables that survive clean_table_rows() with enough real structure to
    be worth rendering as a table at all, excluding candidates that are
    actually a code listing in disguise (see _table_region_is_actually_code).
    """
    try:
        finder = page.find_tables()
    except Exception:
        return []
    results = []
    for table in finder.tables:
        rect = pymupdf.Rect(table.bbox)
        if _table_region_is_actually_code(rect, page_dict):
            continue
        rows = clean_table_rows(table.extract())
        if len(rows) >= TABLE_MIN_ROWS and len(rows[0]) >= TABLE_MIN_COLUMNS:
            results.append((rect, rows))
    return results


def split_block_into_list_items(block):
    """A PDF block sometimes bundles several list items into one block,
    because ordinary paragraph line-spacing sits between them and
    PyMuPDF's own block segmentation has no reason to split there (e.g. a
    numbered quiz-answer list where each answer is only one line) - joining
    every line in the block with a single space, as a normal paragraph
    does, would silently erase the item boundaries instead of just missing
    <li> markup. If at least two of the block's own lines open with a list
    marker, split on those boundaries instead of extracting it as one
    run-on paragraph.

    Returns a list of (text, ordered_or_None) - ordered_or_None is None
    for a leading chunk of text before the first marker (if any); returns
    None (not a list) when fewer than two marker lines are found.
    """
    line_texts = [
        clean_text("".join(span.get("text", "") for span in line.get("spans", [])))
        for line in block.get("lines", [])
    ]
    marker_at = [i for i, t in enumerate(line_texts) if strip_list_marker(t.strip())]
    if len(marker_at) < 2:
        return None

    chunks = []
    if marker_at[0] > 0:
        intro = " ".join(t for t in line_texts[:marker_at[0]] if t.strip())
        if intro.strip():
            chunks.append((intro, None))
    for idx, start in enumerate(marker_at):
        end = marker_at[idx + 1] if idx + 1 < len(marker_at) else len(line_texts)
        item_text, ordered = strip_list_marker(line_texts[start].strip())
        rest = " ".join(t for t in line_texts[start + 1:end] if t.strip())
        full_text = f"{item_text} {rest}".strip() if rest else item_text
        chunks.append((full_text, ordered))
    return chunks


def extract_page_blocks(page, page_dict, page_number, images_dir, save_images, generate_alt,
                         caption_backend, use_gpu, lmstudio_url, lmstudio_model):
    """Extract one page's content into a format-independent structure: a
    list of typed blocks plus margin/callout text (see module docstring for
    why rendering is kept separate from this)."""
    raw_text_blocks = [
        extract_block_raw_text(block) for block in page_dict["blocks"] if block["type"] == 0
    ]
    use_ocr = is_text_garbled(raw_text_blocks)

    blocks = []
    margin_items = []
    callout_items = []

    if use_ocr:
        print(
            f"Страница {page_number}: сломанная кодировка текстового слоя, "
            "распознаю текст через OCR...",
            file=sys.stderr,
        )
        paragraphs, page_height_px = ocr_page_regions(page, page_dict, use_gpu)
        for para in paragraphs:
            text = para["text"]
            if not text.strip():
                continue
            if is_margin_text(para["y0"], para["y1"], page_height_px, text):
                margin_items.extend(split_margin_number(text))
            elif para["region"] > 0:
                # A separate column (e.g. a margin callout box) running
                # alongside the main flow - kept intact as its own block
                # instead of interleaved line-by-line into the main text.
                callout_items.append(text)
            elif is_numeric_heavy(text):
                blocks.append(_with_layout_fields({"type": "p_numeric", "text": text}, para))
            elif para.get("heading_level"):
                blocks.append(_with_layout_fields({"type": "h", "level": para["heading_level"], "text": text}, para))
            else:
                list_item = strip_list_marker(text)
                if list_item:
                    item_text, ordered = list_item
                    blocks.append(_with_layout_fields({"type": "li", "text": item_text, "ordered": ordered}, para))
                else:
                    blocks.append(_with_layout_fields({"type": "p", "text": text}, para))
    else:
        page_height = page.rect.height
        body_size = compute_body_font_size(page_dict, page_height)
        tables = find_page_tables(page, page_dict)
        link_rects = find_page_links(page)
        footnote_markers = collect_superscript_digit_markers(page_dict)
        emitted_tables = set()
        next_block_is_code = False
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            block_rect = pymupdf.Rect(block["bbox"])
            block_area = block_rect.width * block_rect.height
            table_hit_rows = None
            if block_area > 0:
                for table_rect, table_rows in tables:
                    overlap = block_rect & table_rect
                    if overlap.is_empty:
                        continue
                    if (overlap.width * overlap.height) / block_area > TABLE_BLOCK_OVERLAP_MIN_RATIO:
                        table_hit_rows = table_rows
                        break
            if table_hit_rows is not None:
                next_block_is_code = False
                if id(table_hit_rows) not in emitted_tables:
                    emitted_tables.add(id(table_hit_rows))
                    blocks.append({"type": "table", "rows": table_hit_rows})
                continue

            # A code listing's own font is usually monospace (Konovalenko);
            # when it isn't (ci_sharp's subset fonts have no real name to
            # go by), the block immediately following a "Лістинг/Листинг
            # N.N" caption is trusted instead - see LISTING_CAPTION_RE
            # below, which only matches a caption at the very START of a
            # paragraph (never a mid-sentence mention like "см. листинг 3.2").
            if is_monospace_font_name(dominant_font_name(block)) or next_block_is_code:
                next_block_is_code = False
                code_text = clean_code_text(extract_block_code_raw_text(block))
                if code_text.strip():
                    blocks.append({"type": "code", "text": code_text})
                continue

            list_items = split_block_into_list_items(block)
            if list_items is not None:
                next_block_is_code = False
                for item_text, ordered in list_items:
                    if not item_text.strip():
                        continue
                    if ordered is None:
                        blocks.append({"type": "p", "text": item_text})
                    else:
                        blocks.append({"type": "li", "text": item_text, "ordered": ordered})
                continue

            text = extract_block_text(block)
            if not text.strip():
                continue
            if is_margin_text(block["bbox"][1], block["bbox"][3], page_height, text):
                marker_match = FOOTNOTE_MARKER_RE.match(text)
                is_footnote = bool(marker_match and marker_match.group(1) in footnote_markers)
                if not is_footnote:
                    margin_items.extend(split_margin_number(text))
                    continue
            if LISTING_CAPTION_RE.match(text):
                next_block_is_code = True
                blocks.append({"type": "p", "text": text})
                continue
            block_size, bold_ratio = block_font_stats(block)
            level = classify_heading_level(text, block_size, bold_ratio, body_size)
            if level:
                blocks.append({"type": "h", "level": level, "text": text})
            else:
                list_item = strip_list_marker(text)
                if list_item:
                    item_text, ordered = list_item
                    blocks.append({"type": "li", "text": item_text, "ordered": ordered})
                else:
                    p_block = {"type": "p", "text": text}
                    runs = block_emphasis_runs(block, link_rects)
                    if runs:
                        p_block["runs"] = runs
                    blocks.append(p_block)

    image_index = 0
    for block in page_dict["blocks"]:
        if block["type"] == 1:
            image_index += 1
            image_block = extract_image_block(
                page, block, page_number, image_index, images_dir, save_images,
                generate_alt, caption_backend, use_gpu, lmstudio_url, lmstudio_model,
            )
            if image_block is not None:
                blocks.append(image_block)

    return {
        "page_number": page_number,
        "margin": margin_items,
        "blocks": blocks,
        "callout": callout_items,
    }


def state_dir_for(output_dir, input_path):
    return output_dir / f".{input_path.stem}.doc2html-state"


def load_or_init_state(state_dir, doc_identity, current_settings, restart, start_page, end_page):
    """Set up (or validate/resume) the per-page checkpoint directory.

    Only the source document's *identity* (path/size/mtime) is a hard gate -
    mixing fragments from two different documents would be silent data
    corruption. --save-images/--generate-alt are allowed to change between
    invocations (that's how you convert a document in chunks with different
    settings per chunk, or add captions to a later range); a mismatch there
    just prints an informational note, since already-written fragments
    simply keep whatever content they were extracted with.
    """
    meta_path = state_dir / "meta.json"
    pages_dir = state_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
        if not meta or meta.get("identity") != doc_identity:
            raise SystemExit(
                f"Папка состояния {state_dir.name} относится к другому файлу "
                "(другой путь, размер или дата изменения). Укажите другую папку "
                "-o для этого документа."
            )
        previous_settings = meta.get("last_settings")
        if previous_settings and previous_settings != current_settings:
            print(
                "Обратите внимание: --save-images/--generate-alt отличаются от "
                f"предыдущего запуска (было: {previous_settings}, сейчас: "
                f"{current_settings}). Уже готовые страницы сохранят прежнее "
                "содержимое - новыми настройками будут обработаны только страницы, "
                "которых ещё нет в кеше. Чтобы пересчитать уже готовые страницы "
                "текущего диапазона новыми настройками, используйте --restart.",
                file=sys.stderr,
            )

    if restart:
        for page_number in range(start_page, end_page + 1):
            fragment_path = pages_dir / f"{page_number:04d}.json"
            if fragment_path.exists():
                fragment_path.unlink()

    meta_path.write_text(
        json.dumps({"identity": doc_identity, "last_settings": current_settings},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pages_dir


def read_existing_fragments(pages_dir, page_count):
    fragments = {}
    for page_number in range(1, page_count + 1):
        fragment_path = pages_dir / f"{page_number:04d}.json"
        if fragment_path.exists():
            try:
                data = json.loads(fragment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            fragments[page_number] = data
    return fragments


def write_fragment(pages_dir, page_number, page_data):
    (pages_dir / f"{page_number:04d}.json").write_text(
        json.dumps(page_data, ensure_ascii=False), encoding="utf-8",
    )


def assemble_output(out_file, title, fragments, out_format):
    """Atomically (re)write the single assembled output file, in whichever
    format was requested, from whatever page fragments exist so far - so it
    can be opened mid-run to see progress."""
    ordered_pages = [fragments[n] for n in sorted(fragments)]
    dedup_running_heads(ordered_pages)

    if out_format == "md":
        content = build_markdown_document(title, [render_page_markdown(p) for p in ordered_pages])
    elif out_format == "txt":
        content = build_txt_document(title, [render_page_txt(p) for p in ordered_pages])
    else:
        content = build_html_document(title, "\n".join(render_page_html(p) for p in ordered_pages))

    tmp_path = out_file.with_name(out_file.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(out_file)


def convert_document(input_path, output_dir, save_images=True, generate_alt=True,
                      start_page=1, end_page=None, restart=False, clean_state=True,
                      caption_backend="blip", use_gpu=None, out_format="html",
                      lmstudio_url=LMSTUDIO_DEFAULT_URL, lmstudio_model=None):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    resolved_gpu = resolve_use_gpu(use_gpu)

    doc = pymupdf.open(str(input_path))
    page_count = len(doc)
    if end_page is None:
        end_page = page_count
    if not (1 <= start_page <= page_count):
        doc.close()
        raise SystemExit(f"--start-page {start_page} вне диапазона (в документе {page_count} стр.)")
    if not (start_page <= end_page <= page_count):
        doc.close()
        raise SystemExit(f"--end-page {end_page} вне диапазона (в документе {page_count} стр.)")

    title = clean_text(doc.metadata.get("title") or "") or input_path.stem

    stat = input_path.stat()
    doc_identity = {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime": stat.st_mtime,
    }
    current_settings = {"save_images": save_images, "generate_alt": generate_alt}
    state_dir = state_dir_for(output_dir, input_path)
    pages_dir = load_or_init_state(state_dir, doc_identity, current_settings, restart, start_page, end_page)
    fragments = read_existing_fragments(pages_dir, page_count)

    already_done = sum(1 for n in range(start_page, end_page + 1) if n in fragments)
    if already_done:
        print(
            f"Найдено {already_done} уже готовых страниц в диапазоне "
            f"{start_page}-{end_page} - продолжаю с того места, где остановились "
            "(--restart для полной переобработки).",
            file=sys.stderr,
        )

    out_file = output_dir / f"{input_path.stem}.{out_format}"

    for page_number in range(start_page, end_page + 1):
        if page_number in fragments:
            continue

        page = doc[page_number - 1]
        page_dict = page.get_text("dict", sort=True)

        page_data = extract_page_blocks(
            page, page_dict, page_number, images_dir, save_images, generate_alt,
            caption_backend, resolved_gpu, lmstudio_url, lmstudio_model,
        )

        fragments[page_number] = page_data
        write_fragment(pages_dir, page_number, page_data)
        assemble_output(out_file, title, fragments, out_format)

        print(f"Обработана страница {page_number}/{page_count}", file=sys.stderr)

    doc.close()

    # Always (re)assemble at least once, even if every requested page was
    # already cached and the loop above never touched assemble_output - e.g.
    # re-running with a different --out-format against already-extracted
    # pages must still produce that format's file.
    assemble_output(out_file, title, fragments, out_format)

    if clean_state:
        done_count = sum(1 for n in range(1, page_count + 1) if n in fragments)
        if done_count == page_count:
            shutil.rmtree(state_dir, ignore_errors=True)
            print("Документ сконвертирован полностью - служебная папка состояния удалена.", file=sys.stderr)
        else:
            print(
                f"--clean-state: документ ещё не сконвертирован полностью "
                f"({done_count}/{page_count} стр.) - папка состояния сохранена "
                "для последующего продолжения.",
                file=sys.stderr,
            )

    return out_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Конвертация документа (PDF, EPUB, XPS, MOBI, FB2, CBZ и другие форматы, "
            "которые открывает PyMuPDF) в читаемый HTML/Markdown/текст с сохранением картинок."
        )
    )
    parser.add_argument(
        "input_file",
        help="Путь к исходному файлу (PDF, EPUB, XPS, MOBI, FB2, CBZ, изображение и т.п.)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Папка для результата: файл результата и подпапка images (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--out-format", choices=["html", "md", "txt"], default="html",
        help="Формат результата: html (по умолчанию), md (Markdown) или txt (обычный текст)",
    )
    parser.add_argument(
        "--save-images", dest="save_images", action="store_true", default=True,
        help="Сохранять картинки в <output>/images (по умолчанию: включено)",
    )
    parser.add_argument(
        "--no-save-images", dest="save_images", action="store_false",
        help="Не сохранять картинки - вместо изображения вставляется текстовое описание/заглушка",
    )
    parser.add_argument(
        "--generate-alt", choices=["off", "blip", "lmstudio"], default="off",
        help=(
            "Описание картинок для alt/текста-заглушки: off - не генерировать "
            "(по умолчанию, самый быстрый режим, ничего лишнего не ставится/не "
            "запускается), blip - офлайн-модель (требует requirements-blip.txt), "
            "lmstudio - локальный сервер LM Studio"
        ),
    )
    parser.add_argument(
        "--start-page", type=int, default=1,
        help="Первая страница для конвертации, 1-индексация (по умолчанию: 1)",
    )
    parser.add_argument(
        "--end-page", type=int, default=None,
        help="Последняя страница для конвертации, включительно (по умолчанию: последняя)",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Игнорировать состояние предыдущего прерванного запуска и начать заново",
    )
    parser.add_argument(
        "--clean-state", dest="clean_state", action="store_true", default=True,
        help=(
            "Удалить служебную папку с промежуточными фрагментами страниц, когда "
            "весь документ (а не только запрошенный диапазон) окажется полностью "
            "сконвертирован (по умолчанию: включено)"
        ),
    )
    parser.add_argument(
        "--no-clean-state", dest="clean_state", action="store_false",
        help="Не удалять служебную папку состояния автоматически",
    )
    parser.add_argument(
        "--gpu", dest="use_gpu", action="store_true", default=None,
        help="Принудительно использовать GPU (CUDA) для OCR и подписей к картинкам",
    )
    parser.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false",
        help="Принудительно использовать CPU (по умолчанию: автоопределение через torch.cuda)",
    )
    parser.add_argument(
        "--lmstudio-url", default=LMSTUDIO_DEFAULT_URL,
        help=f"Базовый URL OpenAI-совместимого сервера LM Studio (по умолчанию: {LMSTUDIO_DEFAULT_URL})",
    )
    parser.add_argument(
        "--lmstudio-model", default="lfm2.5-vl-450m",
        help=(
            "Имя загруженной в LM Studio vision-модели, используется при "
            "--generate-alt lmstudio (по умолчанию: %(default)s - см. README)"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Ошибка: файл не найден: {input_path}", file=sys.stderr)
        return 1

    generate_alt = args.generate_alt != "off"
    caption_backend = args.generate_alt if generate_alt else "blip"

    if generate_alt and caption_backend == "blip" and not check_blip_available():
        print(
            "Ошибка: пакет transformers не установлен (нужен для --generate-alt blip). "
            "Поставьте: conda run -n pdf2html python -m pip install -r requirements-blip.txt "
            "- либо используйте --generate-alt lmstudio.",
            file=sys.stderr,
        )
        return 1

    if generate_alt and caption_backend == "lmstudio":
        ok, message = check_lmstudio(args.lmstudio_url, args.lmstudio_model)
        if not ok:
            print(f"Ошибка: {message}", file=sys.stderr)
            return 1

    out_file = convert_document(
        input_path,
        args.output_dir,
        save_images=args.save_images,
        generate_alt=generate_alt,
        start_page=args.start_page,
        end_page=args.end_page,
        restart=args.restart,
        clean_state=args.clean_state,
        caption_backend=caption_backend,
        use_gpu=args.use_gpu,
        out_format=args.out_format,
        lmstudio_url=args.lmstudio_url,
        lmstudio_model=args.lmstudio_model,
    )
    print(f"Готово: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
