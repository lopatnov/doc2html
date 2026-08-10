#!/usr/bin/env python
"""Regression checks against real pages of actual books in input/.

Not a unit-test suite in the usual sense: these run the real conversion
pipeline (OCR included on garbled pages) against specific pages of real
books and assert on the resulting extracted JSON structure - each check
documents a bug that was found and fixed against that exact page, so a
regression shows up as a failed assertion instead of only being caught by
re-reading the output by eye.

Requires the book files locally (input/ is not committed - see
.gitignore); a book's checks are skipped cleanly if that file is missing
rather than failing, since each fixture is tied to one specific book's
own layout quirks, not something every checkout is expected to have.

Run: python tests/regression_check.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import doc2html

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
NIELSEN = INPUT_DIR / "Веб-дизайн книга Якоба Нильсена.pdf"
KONOVALENKO = INPUT_DIR / "Konovalenko I. .NET-C#.pdf"

CHECKS = []


def check(book, page, description):
    def register(fn):
        CHECKS.append((book, page, description, fn))
        return fn
    return register


def has_block(blocks, **attrs):
    return any(all(b.get(k) == v for k, v in attrs.items()) for b in blocks)


def all_text(page_data):
    return " ".join(b.get("text", "") for b in page_data["blocks"])


@check(NIELSEN, 3, "титульная/колофонная страница остаётся одной колонкой (ложное срабатывание детектора колонок)")
def _(page_data):
    assert not any("column" in b for b in page_data["blocks"]), \
        "страница 3 не должна размечаться как многоколоночная (сложная индентация колофона - не реальные колонки)"


@check(NIELSEN, 4, "оглавление: 'Содержание' распознаётся как заголовок")
def _(page_data):
    assert has_block(page_data["blocks"], type="h", text="Содержание"), \
        "не нашёл <h>'Содержание' - проверь classify_heading_level/compute_body_font_size"


@check(NIELSEN, 4, "пункт оглавления с номером страницы не становится заголовком (ложное срабатывание)")
def _(page_data):
    assert not has_block(page_data["blocks"], type="h", text="Автомобиль в качестве веб-броузера 41"), \
        "TOC-запись с номером страницы не должна получать <h> - проверь TRAILING/LEADING_PAGE_NUMBER_RE guard"


@check(NIELSEN, 10, "'Предисловие' - отдельный заголовок, не склеен со следующим абзацем")
def _(page_data):
    assert has_block(page_data["blocks"], type="h", text="Предисловие"), \
        "'Предисловие' не выделилось в заголовок - проверь prev_looks_like_heading в reflow_rows_into_paragraphs"


@check(NIELSEN, 11, "боковая врезка 'Список опечаток' не смешивается с основным текстом")
def _(page_data):
    assert page_data.get("callout"), "ожидалась непустая врезка (callout) на странице 11"


@check(NIELSEN, 22, "нет фрагментации оправданного (justified) текста на отдельные слова")
def _(page_data):
    combined = all_text(page_data)
    for bad in ("друко осуна", "как я их на-"):
        assert bad not in combined, (
            f"похоже на регрессию word-fragmentation (боковой эффект boundary_x в cluster_rows): "
            f"{bad!r} найдено в тексте"
        )


@check(NIELSEN, 25, "подпункты оглавления получают визуальный отступ (не одна стена текста)")
def _(page_data):
    assert any(b.get("indent_em") for b in page_data["blocks"]), \
        "ни один блок не получил indent_em - список TOC не должен быть плоским"


@check(NIELSEN, 28, "узкий межколоночный зазор не рвёт слово посередине")
def _(page_data):
    combined = all_text(page_data)
    assert "страницы" in combined, \
        "'страницы' не найдено целиком - похоже на регрессию слияния колонок (split_merged_row/gutter)"
    assert any(b.get("column") is not None for b in page_data["blocks"]), \
        "колонки на странице 28 не определились - проверь detect_column_boundary_x (узкий gutter ~11pt)"


@check(KONOVALENKO, 44, "таблица типов C# распознаётся как настоящая таблица, а не россыпь абзацев")
def _(page_data):
    tables = [b for b in page_data["blocks"] if b["type"] == "table"]
    assert len(tables) == 2, f"ожидал 2 таблицы на странице 44, нашёл {len(tables)}"
    types_table = tables[0]
    assert types_table["rows"][0] == ["Тип C#", "Опис", "Діапазон", "Тип .NET Framework"], \
        f"заголовок таблицы не тот, что ожидался: {types_table['rows'][0]!r}"
    sbyte_row = next((r for r in types_table["rows"] if r[0] == "sbyte"), None)
    assert sbyte_row is not None, "строка 'sbyte' не найдена в таблице - проверь clean_table_rows"
    assert sbyte_row == ["sbyte", "8-бітове знакове ціле", "-128...127", "System.SByte"], \
        f"строка 'sbyte' расползлась по не тем колонкам: {sbyte_row!r} - проверь _merge_split_columns"


def main():
    tmp_dir = Path(tempfile.mkdtemp(prefix="doc2html_regression_"))
    try:
        by_book = {}
        for book, page, _, _ in CHECKS:
            by_book.setdefault(book, set()).add(page)

        fragments_by_book = {}
        for book, pages in by_book.items():
            if not book.exists():
                print(f"ПРОПУСК книги {book.name}: файл не найден локально (input/ не коммитится)")
                continue
            for page in sorted(pages):
                doc2html.convert_document(
                    book, tmp_dir,
                    save_images=False, generate_alt=False,
                    start_page=page, end_page=page,
                    restart=True, clean_state=False,
                    out_format="html",
                )
            state_dir = doc2html.state_dir_for(tmp_dir, book)
            fragments_by_book[book] = doc2html.read_existing_fragments(state_dir / "pages", max(pages))

        failed = 0
        skipped = 0
        for book, page, description, fn in CHECKS:
            label = f"[{book.stem}, стр. {page}] {description}"
            if book not in fragments_by_book:
                skipped += 1
                continue
            page_data = fragments_by_book[book].get(page)
            if page_data is None:
                print(f"ОШИБКА  {label}: страница не была обработана")
                failed += 1
                continue
            try:
                fn(page_data)
                print(f"OK      {label}")
            except AssertionError as exc:
                print(f"ОШИБКА  {label}: {exc}")
                failed += 1

        ran = len(CHECKS) - skipped
        print(f"\n{ran - failed}/{ran} проверок пройдено" + (f" ({skipped} пропущено - книга недоступна)" if skipped else ""))
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
