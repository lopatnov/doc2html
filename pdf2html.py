#!/usr/bin/env python
"""Convert a PDF document into a readable UTF-8 HTML document.

Handles the mixed/broken text encodings common in PDFs (custom font
subsets, private-use-area glyph codes, ligatures) and extracts embedded
images into a sibling ``images`` folder, optionally captioning them for
the ``alt`` attribute. Supports converting a page range, streams the
HTML output to disk page by page, and automatically resumes an
interrupted run.
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

import fitz  # PyMuPDF

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
OCR_LANGUAGES = ["ru", "en"]
OCR_RENDER_DPI = 300
IMAGE_RENDER_DPI = 200
# Share of control characters (below 0x20, excluding whitespace) in a page's
# extracted text above which we treat the PDF's text layer as unusable.
GARBLED_TEXT_THRESHOLD = 0.2
GARBLED_TEXT_MIN_LENGTH = 20
# Text confined to the top/bottom N% of a page and short enough is treated as
# a running header/footer (chapter title, page number) rather than body text.
MARGIN_ZONE_RATIO = 0.09
MARGIN_TEXT_MAX_LEN = 70
# OCR paragraphs that are mostly digits are almost always a mangled column of
# page numbers (see module docstring / README) - flagged, not discarded.
NUMERIC_PARAGRAPH_RATIO = 0.6
NUMERIC_PARAGRAPH_MIN_CHARS = 4
LMSTUDIO_DEFAULT_URL = "http://localhost:1234/v1"
LMSTUDIO_TIMEOUT = 120

# A hyphen at a line-wrap point ("бо-\nлее" -> extracted as "бо- лее") should
# collapse back into one word ("более"). Requires a letter on both sides so we
# don't touch numeric ranges ("2020-2021") or list markers.
HYPHEN_WRAP_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])-\s+(?=[A-Za-zА-Яа-яЁё])")

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
    only needed for --caption-backend blip."""
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
                "Пакет transformers не установлен (нужен для --caption-backend blip). "
                "Поставьте: conda run -n pdf2html python -m pip install -r requirements-blip.txt "
                "- либо используйте --caption-backend lmstudio или --no-generate-alt."
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


def ocr_page_paragraphs(page, page_dict, use_gpu):
    """Render a page to an image and OCR it, for PDFs whose text layer is broken.

    Embedded raster images are blanked out before OCR: they're already
    extracted and captioned separately by render_image_block(), and leaving
    them in would have OCR try to read on-screen UI chrome / photo content as
    if it were the page's body text (e.g. a browser screenshot's menu bar).

    Returns (paragraphs, page_height_px) where each paragraph is
    {"text": str, "y0": float, "y1": float} in the same pixel space as
    page_height_px, so callers can classify header/footer position.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    reader = get_ocr_reader(use_gpu)
    pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")

    scale = OCR_RENDER_DPI / 72.0
    draw = ImageDraw.Draw(image)
    for block in page_dict["blocks"]:
        if block["type"] == 1:
            x0, y0, x1, y1 = block["bbox"]
            draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill="white")

    raw_results = reader.readtext(np.array(image), detail=1, paragraph=True)
    paragraphs = []
    for bbox, text in raw_results:
        text = clean_text(text)
        if not text.strip():
            continue
        ys = [point[1] for point in bbox]
        paragraphs.append({"text": text, "y0": min(ys), "y1": max(ys)})
    return paragraphs, image.height


def is_text_garbled(raw_text_blocks):
    """Detect a broken font/ToUnicode mapping: PyMuPDF still emits *a* character
    per glyph, but with no real CMap those characters are meaningless control
    codes rather than actual letters - the classic symptom of the "wrong
    encoding" problem this script exists to work around.

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
    """Normalize PDF/OCR-extracted text so it is well-formed, readable UTF-8.

    PDFs frequently reference fonts with broken or missing ToUnicode CMaps.
    PyMuPDF still maps such glyphs to *something*, but that something is
    often a Private-Use-Area codepoint with no real meaning (renders as
    tofu boxes) - those are stripped. Ligatures and other compatibility
    characters are folded to their plain equivalents via NFKC. Hyphenated
    line-wraps ("бо- лее") are rejoined into whole words.
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
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        body_html,
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def render_image_block(page, block, page_number, image_index, images_dir, save_images,
                        generate_alt, caption_backend, use_gpu, lmstudio_url, lmstudio_model):
    # The raw bytes in block["image"] are the image XObject as stored in the
    # PDF, BEFORE the placement matrix the page applies to it - for images
    # under a flip/rotate/mirror transform those raw bytes come out upside
    # down or mirrored. Rendering the page's own bbox instead reproduces
    # exactly what a reader sees, transform included.
    bbox = fitz.Rect(block["bbox"])
    if bbox.is_empty or bbox.width < 1 or bbox.height < 1:
        return []
    pixmap = page.get_pixmap(clip=bbox, dpi=IMAGE_RENDER_DPI)
    image_bytes = pixmap.tobytes("png")

    caption = None
    if generate_alt:
        caption = caption_image(image_bytes, caption_backend, use_gpu, lmstudio_url, lmstudio_model)

    if save_images:
        filename = f"image_{page_number:04d}_{image_index:04d}.png"
        (images_dir / filename).write_bytes(image_bytes)
        alt_text = escape(caption) if caption else ""
        return [f'<img src="images/{filename}" alt="{alt_text}" loading="lazy">']
    if caption:
        return [f'<p class="image-placeholder">[Изображение: {escape(caption)}]</p>']
    return ['<p class="image-placeholder">[Изображение]</p>']


def render_page(page, page_dict, page_number, images_dir, save_images, generate_alt,
                 caption_backend, use_gpu, lmstudio_url, lmstudio_model):
    """Build one page's <section> fragment. Returns the fragment HTML string."""
    raw_text_blocks = [
        extract_block_raw_text(block) for block in page_dict["blocks"] if block["type"] == 0
    ]
    use_ocr = is_text_garbled(raw_text_blocks)

    body_items = []
    margin_items = []
    image_index = 0

    if use_ocr:
        print(
            f"Страница {page_number}: сломанная кодировка шрифта в PDF, "
            "распознаю текст через OCR...",
            file=sys.stderr,
        )
        paragraphs, page_height_px = ocr_page_paragraphs(page, page_dict, use_gpu)
        for para in paragraphs:
            text = para["text"]
            if not text.strip():
                continue
            if is_margin_text(para["y0"], para["y1"], page_height_px, text):
                margin_items.append(text)
            elif is_numeric_heavy(text):
                body_items.append(f'<p class="ocr-numbers">{escape(text)}</p>')
            else:
                body_items.append(f"<p>{escape(text)}</p>")
    else:
        page_height = page.rect.height
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            text = extract_block_text(block)
            if not text.strip():
                continue
            if is_margin_text(block["bbox"][1], block["bbox"][3], page_height, text):
                margin_items.append(text)
            else:
                body_items.append(f"<p>{escape(text)}</p>")

    for block in page_dict["blocks"]:
        if block["type"] == 1:
            image_index += 1
            body_items.extend(render_image_block(
                page, block, page_number, image_index, images_dir, save_images,
                generate_alt, caption_backend, use_gpu, lmstudio_url, lmstudio_model,
            ))

    parts = [f'<section class="page" id="page-{page_number}">']
    if margin_items:
        parts.append(f'<p class="page-meta">{escape(" · ".join(margin_items))}</p>')
    parts.extend(body_items)
    parts.append("</section>")
    return "\n".join(parts)


def state_dir_for(output_dir, pdf_path):
    return output_dir / f".{pdf_path.stem}.pdf2html-state"


def load_or_init_state(state_dir, run_meta, restart):
    """Set up (or validate/resume) the per-page checkpoint directory.

    Compatibility is keyed on the source PDF's identity plus the flags that
    change fragment *content* (save_images, generate_alt) - not on the page
    range, so a user can convert a book in several chunks (e.g. 1-100, then
    101-end) across separate invocations and still get one assembled file.
    """
    meta_path = state_dir / "meta.json"
    pages_dir = state_dir / "pages"

    if restart and state_dir.exists():
        shutil.rmtree(state_dir)

    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_meta = None
        if existing_meta != run_meta:
            raise SystemExit(
                "Найдено состояние предыдущего (незавершённого) запуска с "
                "другим PDF-файлом или другими параметрами --save-images/"
                "--generate-alt. Запустите с --restart, чтобы начать заново, "
                "либо укажите другую папку -o."
            )
    else:
        state_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pages_dir.mkdir(parents=True, exist_ok=True)
    return pages_dir


def read_existing_fragments(pages_dir, page_count):
    fragments = {}
    for page_number in range(1, page_count + 1):
        fragment_path = pages_dir / f"{page_number:04d}.html"
        if fragment_path.exists():
            content = fragment_path.read_text(encoding="utf-8")
            if content.strip():
                fragments[page_number] = content
    return fragments


def write_fragment(pages_dir, page_number, html_str):
    (pages_dir / f"{page_number:04d}.html").write_text(html_str, encoding="utf-8")


def assemble_output(out_file, title, fragments):
    """Atomically (re)write the single assembled HTML file from whatever page
    fragments exist so far, so it can be opened mid-run to see progress."""
    ordered = [fragments[n] for n in sorted(fragments)]
    html = build_html_document(title, "\n".join(ordered))
    tmp_path = out_file.with_name(out_file.name + ".tmp")
    tmp_path.write_text(html, encoding="utf-8")
    tmp_path.replace(out_file)


def process_pdf(pdf_path, output_dir, save_images=True, generate_alt=True,
                 start_page=1, end_page=None, restart=False,
                 caption_backend="blip", use_gpu=None,
                 lmstudio_url=LMSTUDIO_DEFAULT_URL, lmstudio_model=None):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    resolved_gpu = resolve_use_gpu(use_gpu)

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    if end_page is None:
        end_page = page_count
    if not (1 <= start_page <= page_count):
        doc.close()
        raise SystemExit(f"--start-page {start_page} вне диапазона (в документе {page_count} стр.)")
    if not (start_page <= end_page <= page_count):
        doc.close()
        raise SystemExit(f"--end-page {end_page} вне диапазона (в документе {page_count} стр.)")

    title = clean_text(doc.metadata.get("title") or "") or pdf_path.stem

    stat = pdf_path.stat()
    run_meta = {
        "pdf_path": str(pdf_path.resolve()),
        "pdf_size": stat.st_size,
        "pdf_mtime": stat.st_mtime,
        "save_images": save_images,
        "generate_alt": generate_alt,
    }
    state_dir = state_dir_for(output_dir, pdf_path)
    pages_dir = load_or_init_state(state_dir, run_meta, restart)
    fragments = read_existing_fragments(pages_dir, page_count)

    already_done = sum(1 for n in range(start_page, end_page + 1) if n in fragments)
    if already_done:
        print(
            f"Найдено {already_done} уже готовых страниц в диапазоне "
            f"{start_page}-{end_page} - продолжаю с того места, где остановились "
            "(--restart для полной переобработки).",
            file=sys.stderr,
        )

    out_file = output_dir / (pdf_path.stem + ".html")

    for page_number in range(start_page, end_page + 1):
        if page_number in fragments:
            continue

        page = doc[page_number - 1]
        page_dict = page.get_text("dict", sort=True)

        fragment_html = render_page(
            page, page_dict, page_number, images_dir, save_images, generate_alt,
            caption_backend, resolved_gpu, lmstudio_url, lmstudio_model,
        )

        fragments[page_number] = fragment_html
        write_fragment(pages_dir, page_number, fragment_html)
        assemble_output(out_file, title, fragments)

        print(f"Обработана страница {page_number}/{page_count}", file=sys.stderr)

    doc.close()
    return out_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Конвертация PDF в читаемый HTML (UTF-8) с сохранением картинок."
    )
    parser.add_argument("input_pdf", help="Путь к исходному PDF-файлу")
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Папка для результата: HTML-файл и подпапка images (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--save-images", dest="save_images", action="store_true", default=True,
        help="Сохранять картинки в <output>/images (по умолчанию: включено)",
    )
    parser.add_argument(
        "--no-save-images", dest="save_images", action="store_false",
        help="Не сохранять картинки - вместо <img> вставляется текстовое описание/заглушка",
    )
    parser.add_argument(
        "--generate-alt", dest="generate_alt", action="store_true", default=False,
        help="Генерировать описание картинки для alt/текста-заглушки (по умолчанию: выключено, самый быстрый режим)",
    )
    parser.add_argument(
        "--no-generate-alt", dest="generate_alt", action="store_false",
        help="Не генерировать описания картинок (по умолчанию)",
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
        "--gpu", dest="use_gpu", action="store_true", default=None,
        help="Принудительно использовать GPU (CUDA) для OCR и подписей к картинкам",
    )
    parser.add_argument(
        "--no-gpu", dest="use_gpu", action="store_false",
        help="Принудительно использовать CPU (по умолчанию: автоопределение через torch.cuda)",
    )
    parser.add_argument(
        "--caption-backend", choices=["blip", "lmstudio"], default="blip",
        help="Движок описания картинок: офлайн BLIP (по умолчанию) или локальный сервер LM Studio",
    )
    parser.add_argument(
        "--lmstudio-url", default=LMSTUDIO_DEFAULT_URL,
        help=f"Базовый URL OpenAI-совместимого сервера LM Studio (по умолчанию: {LMSTUDIO_DEFAULT_URL})",
    )
    parser.add_argument(
        "--lmstudio-model", default=None,
        help="Имя загруженной в LM Studio vision-модели (обязательно при --caption-backend lmstudio)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pdf_path = Path(args.input_pdf)
    if not pdf_path.is_file():
        print(f"Ошибка: файл не найден: {pdf_path}", file=sys.stderr)
        return 1

    if args.generate_alt and args.caption_backend == "blip" and not check_blip_available():
        print(
            "Ошибка: пакет transformers не установлен (нужен для --caption-backend blip, "
            "используемого по умолчанию). Поставьте: conda run -n pdf2html python -m pip "
            "install -r requirements-blip.txt - либо используйте --caption-backend lmstudio.",
            file=sys.stderr,
        )
        return 1

    if args.caption_backend == "lmstudio" and args.generate_alt:
        ok, message = check_lmstudio(args.lmstudio_url, args.lmstudio_model)
        if not ok:
            print(f"Ошибка: {message}", file=sys.stderr)
            return 1

    out_file = process_pdf(
        pdf_path,
        args.output_dir,
        save_images=args.save_images,
        generate_alt=args.generate_alt,
        start_page=args.start_page,
        end_page=args.end_page,
        restart=args.restart,
        caption_backend=args.caption_backend,
        use_gpu=args.use_gpu,
        lmstudio_url=args.lmstudio_url,
        lmstudio_model=args.lmstudio_model,
    )
    print(f"Готово: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
