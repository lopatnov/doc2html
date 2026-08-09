#!/usr/bin/env python
"""Convert a PDF document into a readable UTF-8 HTML document.

Handles the mixed/broken text encodings common in PDFs (custom font
subsets, private-use-area glyph codes, ligatures) and extracts embedded
images into a sibling ``images`` folder, optionally captioning them with
an offline image-captioning model for the ``alt`` attribute.
"""

import argparse
import io
import sys
import unicodedata
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

_caption_model = None
_caption_processor = None
_ocr_reader = None


def get_captioner():
    """Lazily load the offline image-captioning model (only when needed)."""
    global _caption_model, _caption_processor
    if _caption_model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        print(
            f"Loading image captioning model '{CAPTION_MODEL_NAME}' "
            "(first run downloads the weights)...",
            file=sys.stderr,
        )
        _caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
        _caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    return _caption_processor, _caption_model


def caption_image(image_bytes):
    """Return a short English description of the image, or None on failure."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    # Skip tiny decorative images (bullets, rules, icons) - captioning them is noise.
    if image.width < 32 or image.height < 32:
        return None

    processor, model = get_captioner()
    inputs = processor(image, return_tensors="pt")
    # no_repeat_ngram_size guards against BLIP's greedy decoding getting stuck
    # in a repetition loop on flat/low-detail images (e.g. "menu menu menu...").
    output = model.generate(**inputs, max_new_tokens=30, no_repeat_ngram_size=3)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip() or None


def get_ocr_reader():
    """Lazily load the offline OCR model (only when a garbled text layer is hit)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        print(
            f"Loading OCR model for languages {OCR_LANGUAGES} "
            "(first run downloads the weights)...",
            file=sys.stderr,
        )
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def ocr_page_text(page):
    """Render a page to an image and OCR it, for PDFs whose text layer is broken."""
    import numpy as np
    from PIL import Image

    reader = get_ocr_reader()
    pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    paragraphs = reader.readtext(np.array(image), detail=0, paragraph=True)
    return [clean_text(p) for p in paragraphs if p and p.strip()]


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


def clean_text(text):
    """Normalize PDF-extracted text so it is well-formed, readable UTF-8.

    PDFs frequently reference fonts with broken or missing ToUnicode CMaps.
    PyMuPDF still maps such glyphs to *something*, but that something is
    often a Private-Use-Area codepoint with no real meaning (renders as
    tofu boxes) - those are stripped. Ligatures and other compatibility
    characters are folded to their plain equivalents via NFKC.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text
        if not (0xE000 <= ord(ch) <= 0xF8FF)
        and not (ord(ch) < 0x20 and ch not in "\n\t")
    )
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


def build_html_document(title, body_parts):
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
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
    ]
    parts.extend(body_parts)
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def render_image_block(page, block, page_index, image_index, images_dir, save_images, generate_alt):
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

    caption = caption_image(image_bytes) if generate_alt else None

    if save_images:
        filename = f"image_{page_index + 1:04d}_{image_index:04d}.png"
        (images_dir / filename).write_bytes(image_bytes)
        alt_text = escape(caption) if caption else ""
        return [f'<img src="images/{filename}" alt="{alt_text}" loading="lazy">']
    if caption:
        return [f'<p class="image-placeholder">[Изображение: {escape(caption)}]</p>']
    return ['<p class="image-placeholder">[Изображение]</p>']


def process_pdf(pdf_path, output_dir, save_images=True, generate_alt=True):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    title = clean_text(doc.metadata.get("title") or "") or pdf_path.stem

    body_parts = []
    image_counter = 0
    page_count = len(doc)

    for page_index in range(page_count):
        page = doc[page_index]
        page_dict = page.get_text("dict", sort=True)
        page_html = [f'<section class="page" id="page-{page_index + 1}">']

        raw_text_blocks = [
            extract_block_raw_text(block) for block in page_dict["blocks"] if block["type"] == 0
        ]
        use_ocr = is_text_garbled(raw_text_blocks)
        if use_ocr:
            print(
                f"Страница {page_index + 1}: сломанная кодировка шрифта в PDF, "
                "распознаю текст через OCR...",
                file=sys.stderr,
            )
            text_blocks = ocr_page_text(page)

        if not use_ocr:
            for block in page_dict["blocks"]:
                if block["type"] == 0:
                    text = extract_block_text(block)
                    if text.strip():
                        page_html.append(f"<p>{escape(text)}</p>")
                elif block["type"] == 1:
                    image_counter += 1
                    page_html.extend(
                        render_image_block(page, block, page_index, image_counter,
                                            images_dir, save_images, generate_alt)
                    )
        else:
            # OCR paragraphs have no reliable per-image position, so they are
            # emitted first, followed by this page's images in document order.
            for text in text_blocks:
                if text.strip():
                    page_html.append(f"<p>{escape(text)}</p>")
            for block in page_dict["blocks"]:
                if block["type"] == 1:
                    image_counter += 1
                    page_html.extend(
                        render_image_block(page, block, page_index, image_counter,
                                            images_dir, save_images, generate_alt)
                    )

        page_html.append("</section>")
        body_parts.append("\n".join(page_html))
        print(f"Обработана страница {page_index + 1}/{page_count}", file=sys.stderr)

    doc.close()

    html = build_html_document(title, body_parts)
    out_file = output_dir / (pdf_path.stem + ".html")
    out_file.write_text(html, encoding="utf-8")
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
        "--generate-alt", dest="generate_alt", action="store_true", default=True,
        help="Генерировать описание картинки (offline-модель) для alt/текста-заглушки (по умолчанию: включено)",
    )
    parser.add_argument(
        "--no-generate-alt", dest="generate_alt", action="store_false",
        help="Не генерировать описания картинок (быстрее, без загрузки ML-модели)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pdf_path = Path(args.input_pdf)
    if not pdf_path.is_file():
        print(f"Ошибка: файл не найден: {pdf_path}", file=sys.stderr)
        return 1

    out_file = process_pdf(
        pdf_path,
        args.output_dir,
        save_images=args.save_images,
        generate_alt=args.generate_alt,
    )
    print(f"Готово: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
