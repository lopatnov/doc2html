#!/usr/bin/env python
"""Deterministic structural auditor for doc2html's cached extraction.

Read-only: does NOT run or re-run conversion. For every page that already
has a cached JSON fragment (see doc2html.state_dir_for), independently
re-derives a "source truth" token stream for that page - either the exact
text layer (page.get_text()) on pages with an intact CMap, or a fresh
EasyOCR pass (embedded images masked out, same as doc2html's own main
pass - those are captioned, not transcribed - but callout boxes left
UNmasked, unlike doc2html's own main pass, so misrouted/dropped callout
content stays visible here) on pages doc2html itself had to route through
OCR - and compares it against everything that actually ended up in the
page's extracted blocks/margin/callout.

Four signals per page, all pure text/geometry, no LLM:
  - coverage: source words never accounted for anywhere in the extraction
  - order:    how much the extraction's reading order diverges from the
              source's (the column/callout-interleaving bug class - see
              the page-11/19 false-column-split bug this was built to
              have caught automatically)
  - routing:  a suspiciously long margin (the footnote-into-margin bug
              class)
  - structural: cheap invariant checks on the blocks themselves (a bullet
              landing mid-paragraph, a flattened code block, a ragged
              table, back-to-back duplicate blocks, a block ending on a
              bare hyphen)

Output is a ranked findings report (JSON + a human-readable top list) - a
punch list to review, not an automatic fix. See CLAUDE.md for why v1
deliberately stops here, with no model in the loop at all.

Run:
  python tools/audit_pages.py "input/Book.pdf" -o output
  python tools/audit_pages.py "input/Book.pdf" -o output --start-page 1 --end-page 50 --top 20
"""
import argparse
import bisect
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import doc2html

TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
# Below this fraction of source words accounted for, a page is flagged for
# missing content.
COVERAGE_FLAG_THRESHOLD = 0.85
# Below this reading-order preservation score (with enough matched tokens
# to be meaningful), a page is flagged for reordering.
ORDER_FLAG_THRESHOLD = 0.85
ORDER_MIN_MATCHED_TOKENS = 8
# A margin longer than this (in words) is unusual for a running head/page
# number and more likely to be misrouted body content.
MARGIN_SUSPICIOUS_WORD_COUNT = 12


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def source_truth_tokens(page, page_dict, use_gpu):
    """(tokens, was_ocr) - was_ocr just labels the report, doesn't change
    scoring."""
    raw_text_blocks = [
        doc2html.extract_block_raw_text(b) for b in page_dict["blocks"] if b["type"] == 0
    ]
    if not doc2html.is_text_garbled(raw_text_blocks):
        text = " ".join(
            doc2html.extract_block_text(b) for b in page_dict["blocks"] if b["type"] == 0
        )
        return tokenize(text), False

    from PIL import Image, ImageDraw
    import numpy as np

    reader = doc2html.get_ocr_reader(use_gpu)
    scale = doc2html.OCR_RENDER_DPI / 72.0
    pixmap = page.get_pixmap(dpi=doc2html.OCR_RENDER_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(image)
    for block in page_dict["blocks"]:
        if block["type"] == 1:
            x0, y0, x1, y1 = block["bbox"]
            draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill="white")
    raw_results = reader.readtext(np.array(image), detail=1, paragraph=False)
    text = " ".join(t.strip() for _bbox, t, _conf in raw_results if t.strip())
    # Without this, a hyphenated line-wrap ("стра- ницы") tokenizes as two
    # half-word fragments that can never match the real pipeline's already
    # dehyphenated "страницы" token, producing systematic false "missing
    # word" hits on every OCR page.
    text = doc2html.clean_text(text)
    return tokenize(text), True


def extracted_tokens_by_field(page_data):
    block_text = []
    for b in page_data["blocks"]:
        text = b.get("text")
        if text:
            block_text.append(text)
        if b["type"] == "table":
            for row in b.get("rows", []):
                block_text.extend(row)
        elif b["type"] in ("img", "img_placeholder"):
            caption = b.get("alt") or b.get("caption")
            if caption:
                block_text.append(caption)
    margin_text = " ".join(page_data.get("margin", []))
    callout_text = " ".join(page_data.get("callout", []))
    return (
        tokenize(" ".join(block_text)),
        tokenize(margin_text),
        tokenize(callout_text),
    )


def order_score(source_tokens, extracted_tokens):
    """1.0 = extraction's reading order exactly follows the source's,
    lower = more reordering. Matches each extracted token to the next
    unconsumed occurrence of it in the source (greedy, left to right),
    then measures what fraction of the resulting source-position sequence
    forms a longest non-decreasing subsequence (patience sorting, O(n log n))
    - a page where two columns got interleaved will match this poorly even
    though every individual word is present."""
    positions = defaultdict(list)
    for i, tok in enumerate(source_tokens):
        positions[tok].append(i)
    consumed = defaultdict(int)
    seq = []
    for tok in extracted_tokens:
        lst = positions.get(tok)
        if not lst:
            continue
        idx = consumed[tok]
        if idx < len(lst):
            seq.append(lst[idx])
            consumed[tok] += 1
    if len(seq) < 4:
        return 1.0, len(seq)
    tails = []
    for x in seq:
        i = bisect.bisect_right(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails) / len(seq), len(seq)


def structural_findings(page_data):
    findings = []
    blocks = page_data["blocks"]
    prev_text = None
    for i, b in enumerate(blocks):
        text = b.get("text", "")
        if b["type"] in ("p", "li") and len(text) > 1 and any(ch in text[1:] for ch in doc2html.LIST_BULLET_CHARS):
            findings.append(f"блок #{i} ({b['type']}): маркер списка внутри текста - похоже на слипшийся пункт списка")
        if b["type"] == "code" and "\n" not in text:
            findings.append(f"блок #{i} (code): нет переноса строк - код мог склеиться в одну строку")
        if b["type"] == "table":
            widths = {len(r) for r in b.get("rows", [])}
            if len(widths) > 1:
                findings.append(f"блок #{i} (table): неровные строки (число колонок: {sorted(widths)})")
        if text and text.rstrip().endswith("-") and b["type"] in ("p", "li", "h"):
            findings.append(f"блок #{i} ({b['type']}): обрывается на дефисе - возможен разрыв слова")
        if prev_text is not None and text and text == prev_text:
            findings.append(f"блок #{i} ({b['type']}): дублирует предыдущий блок дословно")
        prev_text = text
    margin = page_data.get("margin", [])
    margin_words = sum(len(tokenize(m)) for m in margin)
    if margin_words > MARGIN_SUSPICIOUS_WORD_COUNT:
        findings.append(
            f"margin подозрительно длинный ({margin_words} слов) - похоже на текст, "
            "утёкший в колонтитул (класс бага 'сноска в margin')"
        )
    return findings


def audit_page(doc, page_number, page_data, use_gpu):
    page = doc[page_number - 1]
    page_dict = page.get_text("dict", sort=True)
    source_tokens, was_ocr = source_truth_tokens(page, page_dict, use_gpu)
    block_tokens, margin_tokens, callout_tokens = extracted_tokens_by_field(page_data)
    all_extracted = block_tokens + margin_tokens + callout_tokens

    source_counts = Counter(source_tokens)
    extracted_counts = Counter(all_extracted)
    missing = source_counts - extracted_counts
    missing_total = sum(missing.values())
    source_total = sum(source_counts.values())
    coverage = 1.0 - (missing_total / source_total) if source_total else 1.0

    order, matched = order_score(source_tokens, block_tokens)
    structural = structural_findings(page_data)

    score = 0.0
    reasons = []
    if coverage < COVERAGE_FLAG_THRESHOLD:
        score += (COVERAGE_FLAG_THRESHOLD - coverage) * 100
        reasons.append(f"coverage={coverage:.0%} (пропущено ~{missing_total} слов из {source_total})")
    if order < ORDER_FLAG_THRESHOLD and matched >= ORDER_MIN_MATCHED_TOKENS:
        score += (ORDER_FLAG_THRESHOLD - order) * 80
        reasons.append(f"order={order:.0%} (порядок чтения расходится с источником, {matched} слов сопоставлено)")
    if structural:
        score += len(structural) * 10
        reasons.extend(structural)

    return {
        "page": page_number,
        "ocr": was_ocr,
        "coverage": round(coverage, 3),
        "order": round(order, 3),
        "matched_tokens": matched,
        "score": round(score, 1),
        "reasons": reasons,
        "missing_sample": [w for w, _ in missing.most_common(10)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file")
    parser.add_argument("-o", "--output-dir", default="output")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--top", type=int, default=30, help="how many worst pages to print (default: 30)")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--report", default=None, help="write full JSON findings here (default: <output-dir>/audit_findings.json)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        print(f"Ошибка: файл не найден: {input_path}", file=sys.stderr)
        return 1

    doc = doc2html.pymupdf.open(str(input_path))
    page_count = len(doc)
    end_page = args.end_page or page_count

    state_dir = doc2html.state_dir_for(output_dir, input_path)
    pages_dir = state_dir / "pages"
    if not pages_dir.exists():
        print(f"Ошибка: не найдена папка состояния {state_dir} - страницы ещё не сконвертированы?", file=sys.stderr)
        doc.close()
        return 1
    fragments = doc2html.read_existing_fragments(pages_dir, page_count)

    results = []
    checked = 0
    for page_number in range(args.start_page, end_page + 1):
        page_data = fragments.get(page_number)
        if page_data is None:
            continue
        checked += 1
        print(f"Аудит страницы {page_number}/{end_page}...", file=sys.stderr)
        try:
            results.append(audit_page(doc, page_number, page_data, args.gpu))
        except Exception as exc:
            print(f"  пропуск страницы {page_number}: {exc}", file=sys.stderr)
    doc.close()

    if not results:
        print("Ни одной уже сконвертированной страницы в этом диапазоне не найдено.", file=sys.stderr)
        return 0

    results.sort(key=lambda r: -r["score"])

    report_path = Path(args.report) if args.report else output_dir / "audit_findings.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = [r for r in results if r["score"] > 0]
    print(f"\nПроверено страниц: {checked}. С подозрениями: {len(flagged)}.")
    print(f"Полный отчёт: {report_path}\n")
    for r in results[: args.top]:
        if r["score"] <= 0:
            break
        print(f"стр. {r['page']:>4}  score={r['score']:>6.1f}  coverage={r['coverage']:.0%}  order={r['order']:.0%}")
        for reason in r["reasons"][:4]:
            print(f"    - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
