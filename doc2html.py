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
# A vector-graphics region (see find_callout_boxes) spanning most of the
# page width is a full-width rule (e.g. a header underline), not a margin
# callout box; anything smaller than this is noise, not a real box.
CALLOUT_MAX_WIDTH_RATIO = 0.85
CALLOUT_MIN_SIZE_PT = 20
CALLOUT_IMAGE_OVERLAP_MAX_RATIO = 0.5

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


def cluster_rows(line_boxes):
    """Merge raw OCR detections into visual rows by vertical (y-range)
    OVERLAP rather than center proximity or a fixed gap.

    This is what correctly reattaches a table-of-contents title to its page
    number - "Title......." and "27" are split into two boxes because the
    sparse, undotted dot leader between them isn't recognized as text, but
    both boxes span almost the same y-range (the digits are shorter/smaller
    but still sit within the title's height), so they overlap heavily
    despite the huge horizontal gap. Two genuinely different lines at a
    similar height - including two columns running in parallel, e.g. a
    margin callout beside the main text - normally overlap only a little,
    since their baselines/font sizes rarely align that closely; a plain
    center+tolerance test can't tell these two situations apart nearly as
    reliably as overlap does.
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

    result = []
    for row in rows:
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


def reflow_rows_into_paragraphs(rows):
    """Merge consecutive rows into paragraphs: close vertical spacing plus a
    row whose text doesn't already "look finished" (no terminal punctuation,
    no trailing page number) continues the paragraph; otherwise a new one
    starts. The trailing-page-number check is what keeps table-of-contents
    entries as separate one-line paragraphs instead of one giant run-on
    block, even though they're single-spaced like a normal paragraph.
    """
    if not rows:
        return []
    heights = [r["y1"] - r["y0"] for r in rows] or [10]
    median_height = sorted(heights)[len(heights) // 2] or 10
    gap_threshold = median_height * PARAGRAPH_GAP_RATIO

    paragraphs = []
    current_text = rows[0]["text"]
    current = dict(rows[0])
    for row in rows[1:]:
        gap = row["y0"] - current["y1"]
        prev_text = current_text.rstrip()
        prev_finished = (
            not prev_text
            or prev_text[-1] in PARAGRAPH_TERMINATOR_CHARS
            or TRAILING_PAGE_NUMBER_RE.search(prev_text) is not None
        )
        if gap <= gap_threshold and not prev_finished:
            current_text = prev_text + " " + row["text"]
            current["y1"] = row["y1"]
            current["x1"] = max(current["x1"], row["x1"])
        else:
            paragraphs.append({**current, "text": current_text})
            current_text = row["text"]
            current = dict(row)
    paragraphs.append({**current, "text": current_text})
    return paragraphs


def ocr_region_paragraphs(reader, image, offset_x=0.0, offset_y=0.0):
    """OCR an already-rendered (and already-masked, if relevant) region
    image and return reflowed paragraphs. offset_x/offset_y translate a
    cropped sub-image's local coordinates back into full-page pixel space,
    so a callout box's paragraphs are directly comparable (e.g. for
    header/footer classification) to the main flow's.
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

    rows = cluster_rows(line_boxes)
    paragraphs = []
    for para in reflow_rows_into_paragraphs(rows):
        text = clean_text(para["text"])
        if not text.strip():
            continue
        paragraphs.append({"text": text, "y0": para["y0"], "y1": para["y1"]})
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
    for para in ocr_region_paragraphs(reader, main_image):
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


def is_margin_text(y0, y1, page_height, text):
    """Short text confined to the page's top/bottom margin: a running header,
    chapter title, or lone page number rather than a body paragraph."""
    if not text or len(text) > MARGIN_TEXT_MAX_LEN:
        return False
    top_limit = page_height * MARGIN_ZONE_RATIO
    bottom_limit = page_height * (1 - MARGIN_ZONE_RATIO)
    return y1 <= top_limit or y0 >= bottom_limit


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


def render_block_html(block):
    block_type = block["type"]
    if block_type == "p":
        return f"<p>{escape(block['text'])}</p>"
    if block_type == "p_numeric":
        return f'<p class="ocr-numbers">{escape(block["text"])}</p>'
    if block_type == "img":
        return f'<img src="{block["src"]}" alt="{escape(block["alt"])}" loading="lazy">'
    if block_type == "img_placeholder":
        if block["caption"]:
            return f'<p class="image-placeholder">[Изображение: {escape(block["caption"])}]</p>'
        return '<p class="image-placeholder">[Изображение]</p>'
    raise ValueError(f"unknown block type: {block_type}")


def render_page_html(page_data):
    parts = [f'<section class="page" id="page-{page_data["page_number"]}">']
    margin = page_data.get("margin_display", page_data["margin"])
    if margin:
        parts.append(f'<p class="page-meta">{escape(" · ".join(margin))}</p>')
    parts.extend(render_block_html(b) for b in page_data["blocks"])
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
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        body_html,
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def render_block_markdown(block):
    block_type = block["type"]
    if block_type in ("p", "p_numeric"):
        return block["text"]
    if block_type == "img":
        alt = block["alt"].replace("[", "(").replace("]", ")")
        return f'![{alt}]({block["src"]})'
    if block_type == "img_placeholder":
        return f'*[Изображение: {block["caption"]}]*' if block["caption"] else "*[Изображение]*"
    raise ValueError(f"unknown block type: {block_type}")


def render_page_markdown(page_data):
    parts = []
    margin = page_data.get("margin_display", page_data["margin"])
    if margin:
        parts.append(f'*{" · ".join(margin)}*')
    parts.extend(render_block_markdown(b) for b in page_data["blocks"])
    if page_data["callout"]:
        parts.append("\n".join(f"> {t}" for t in page_data["callout"]))
    return "\n\n".join(p for p in parts if p and p.strip())


def build_markdown_document(title, page_mds):
    body = "\n\n---\n\n".join(page_mds)
    return f"# {title}\n\n{body}\n"


def render_block_txt(block):
    block_type = block["type"]
    if block_type in ("p", "p_numeric"):
        return block["text"]
    if block_type == "img":
        return f'[Изображение: {block["alt"]}]' if block["alt"] else "[Изображение]"
    if block_type == "img_placeholder":
        return f'[Изображение: {block["caption"]}]' if block["caption"] else "[Изображение]"
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
                blocks.append({"type": "p_numeric", "text": text})
            else:
                blocks.append({"type": "p", "text": text})
    else:
        page_height = page.rect.height
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            text = extract_block_text(block)
            if not text.strip():
                continue
            if is_margin_text(block["bbox"][1], block["bbox"][3], page_height, text):
                margin_items.extend(split_margin_number(text))
            else:
                blocks.append({"type": "p", "text": text})

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
