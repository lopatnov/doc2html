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
NON_DESIGNERS = INPUT_DIR / "Non-Designers-Design-Book.pdf"
RESUME = INPUT_DIR / "Oleksandr_Lopatnov_Resume.docx"

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


@check(NIELSEN, 42, "двухколоночная врезка не рассыпается в перемешанный текст")
def _(page_data):
    callout = page_data.get("callout", [])
    combined = " ".join(callout)
    assert "учитывать" in combined and "мониторов" in combined, (
        f"текст левой колонки врезки не найден целиком в {callout!r} - "
        "проверь detect_callout_column_boundary_x"
    )
    assert "Большинство задач выполняется" in combined, (
        f"текст правой колонки врезки не найден целиком в {callout!r} - "
        "проверь detect_callout_column_boundary_x"
    )
    for bad in ("отывать тex", "тельности. Большинство задач выполняется отывать"):
        assert bad not in combined, (
            f"регрессия: колонки врезки снова перемешались ({bad!r} найдено в тексте) - "
            "проверь gutter в вызове ocr_region_paragraphs для callout_boxes"
        )


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


@check(NIELSEN, 25, "запись оглавления не склеивается с соседней колонкой (нулевой запас gutter_right)")
def _(page_data):
    entry = next((b for b in page_data["blocks"] if "Листы стилей" in b.get("text", "")), None)
    assert entry is not None, "пункт 'Листы стилей' не найден среди блоков"
    assert "102" in entry["text"], (
        "номер страницы '102' оторвался от своего заголовка 'Листы стилей' - "
        "проверь классификацию строк по колонкам в ocr_region_paragraphs"
    )
    assert "Продвинутый пользователь" not in entry["text"], (
        "текст ЛЕВОЙ колонки ('Продвинутый пользователь') склеился с ПРАВОЙ ('102 Листы стилей') - "
        "regression в gutter_mid/ocr_region_paragraphs (короткий токен вроде номера страницы "
        "у самого начала колонки садится вплотную к gutter_right без запаса на OCR-джиттер)"
    )


@check(NIELSEN, 28, "узкий межколоночный зазор не рвёт слово посередине")
def _(page_data):
    combined = all_text(page_data)
    assert "страницы" in combined, \
        "'страницы' не найдено целиком - похоже на регрессию слияния колонок (split_merged_row/gutter)"
    assert any(b.get("column") is not None for b in page_data["blocks"]), \
        "колонки на странице 28 не определились - проверь detect_column_boundary_x (узкий gutter ~11pt)"


@check(KONOVALENKO, 16, "сноска не теряется в margin - остаётся содержимым страницы")
def _(page_data):
    assert not any("Його також можна завантажити" in m for m in page_data.get("margin", [])), \
        "сноска попала в margin - проверь collect_superscript_digit_markers/FOOTNOTE_MARKER_RE"
    assert any("Його також можна завантажити" in b.get("text", "") for b in page_data["blocks"]), \
        "текст сноски не найден среди блоков страницы"


@check(KONOVALENKO, 10, "маркер сноски получает <sup> через runs")
def _(page_data):
    superscript_runs = [
        run for b in page_data["blocks"] if b["type"] == "p" and b.get("runs")
        for run in b["runs"] if run.get("superscript")
    ]
    assert superscript_runs, "ни один run не помечен superscript - проверь flags&1 в extract_block_runs"


@check(NON_DESIGNERS, 4,
       "реальные PDF-ссылки (внешний URL и mailto:) сохраняются, а не отбрасываются молча")
def _(page_data):
    hrefs = [
        run.get("href")
        for b in page_data["blocks"] if b["type"] == "p" and b.get("runs")
        for run in b["runs"] if run.get("href")
    ]
    assert any("peachpit.com" in h.lower() for h in hrefs), \
        f"внешняя ссылка на peachpit.com не найдена среди {hrefs!r} - проверь find_page_links/_span_href"
    assert any(h.startswith("mailto:") for h in hrefs), \
        f"mailto: ссылка не найдена среди {hrefs!r}"


def _hrefs(page_data):
    return [
        run.get("href")
        for b in page_data["blocks"] if b["type"] == "p" and b.get("runs")
        for run in b["runs"] if run.get("href")
    ]


@check(RESUME, 1, "голая строка 'linkedin.com/in/...' в .docx получает href из реальной ссылки")
def _(page_data):
    hrefs = _hrefs(page_data)
    assert "https://www.linkedin.com/in/lopatnov/" in hrefs, (
        f"ссылка на LinkedIn не найдена среди {hrefs!r} - голая строка "
        "'linkedin.com/in/lopatnov' в тексте (без http:///www., BARE_URL_RE её не ловит) "
        "должна получить href из реальной .docx-ссылки - проверь extract_docx_hyperlinks/"
        "apply_known_hyperlinks"
    )


@check(RESUME, 6, "реальные .docx-гиперссылки восстанавливаются (page.get_links() для .docx всегда пуст)")
def _(page_data):
    hrefs = _hrefs(page_data)
    assert "https://lopatnov.github.io/pressmark/" in hrefs, (
        f"ссылка на pressmark не найдена среди {hrefs!r} - "
        "проверь extract_docx_hyperlinks/apply_known_hyperlinks"
    )


@check(KONOVALENKO, 24, "курсив внутри абзаца ('налагоджувачем') сохраняется, а не отбрасывается")
def _(page_data):
    p_blocks = [b for b in page_data["blocks"] if b["type"] == "p"]
    runs_blocks = [b for b in p_blocks if b.get("runs")]
    assert runs_blocks, "ни один абзац не получил 'runs' - проверь block_emphasis_runs"
    assert any(
        any(run["italic"] and "налагоджувачем" in run["text"] for run in b["runs"])
        for b in runs_blocks
    ), "курсивный run с 'налагоджувачем' не найден - проверь extract_block_runs"


@check(NON_DESIGNERS, 284,
       "нумерованный список, слипшийся в один PDF-блок, разбивается на отдельные li")
def _(page_data):
    items = [b for b in page_data["blocks"] if b["type"] == "li"]
    assert len(items) == 7, f"ожидал 7 пунктов списка на странице 284, нашёл {len(items)}"
    assert all(b.get("ordered") for b in items), "все пункты должны быть нумерованными (ordered=True)"


@check(NON_DESIGNERS, 79, "картинки стоят на своей позиции в тексте, а не все скопом в конце страницы")
def _(page_data):
    types = [b["type"] for b in page_data["blocks"]]
    # save_images=False in this test harness (see main()) - an image
    # block comes back as "img_placeholder", not "img", but position in
    # the block list is exactly what's under test either way.
    img_indices = [i for i, t in enumerate(types) if t in ("img", "img_placeholder")]
    assert len(img_indices) == 3, f"ожидал 3 картинки на странице 79, нашёл {len(img_indices)}"
    assert any(t == "p" for t in types[:max(img_indices)]), \
        "хотя бы один абзац должен идти ДО последней картинки - иначе картинки снова все в конце (см. merge в extract_page_blocks)"


@check(KONOVALENKO, 90, "код в листинге распознаётся как блок кода с сохранением отступов, не как таблица/врезка")
def _(page_data):
    code_blocks = [b for b in page_data["blocks"] if b["type"] == "code"]
    assert len(code_blocks) == 1, f"ожидал 1 блок кода на странице 90, нашёл {len(code_blocks)}"
    assert not any(b["type"] == "table" for b in page_data["blocks"]), \
        "листинг с колонкой номеров строк не должен ошибочно распознаваться как таблица - проверь _region_is_actually_code"
    assert not page_data.get("callout"), \
        "рамка вокруг листинга не должна распознаваться как врезка - проверь _region_is_actually_code в фильтре callout_boxes"
    text = code_blocks[0]["text"]
    assert "    student.Name" in text, \
        "отступ перед 'student.Name' потерян - проверь extract_block_code_raw_text (не должен .strip() строки)"
    assert "\n" in text, "код должен сохранять переносы строк, а не склеиваться в одну строку"


@check(INPUT_DIR / "ci_sharp.pdf", 73, "безрамочная врезка 'ПРИМЕЧАНИЕ' распознаётся по текстовой метке")
def _(page_data):
    assert page_data.get("callout"), "врезка ПРИМЕЧАНИЕ не найдена - проверь NOTE_LABEL_RE"
    assert any("Рассмотренное приложение" in c for c in page_data["callout"]), \
        "текст врезки не совпадает с ожидаемым"


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


def _assert_image_namespacing(tmp_dir):
    """Runs the actual checks and raises AssertionError on failure. Uses
    explicit `raise` rather than `assert` so the checks still fire under
    `python -O` (which strips assert statements) - required because the
    caller catches AssertionError in a different function/scope, so Sonar's
    "no assert inside its own try/except AssertionError" rule doesn't apply
    here, but the -O hazard is the same either way."""
    doc2html.convert_document(
        NON_DESIGNERS, tmp_dir, save_images=True, generate_alt=False,
        start_page=79, end_page=79, restart=True, clean_state=False,
        out_format="html",
    )
    state_dir = doc2html.state_dir_for(tmp_dir, NON_DESIGNERS)
    fragments = doc2html.read_existing_fragments(state_dir / "pages", 79)
    page_data = fragments.get(79)
    img_blocks = [b for b in page_data["blocks"] if b["type"] == "img"]
    if not img_blocks:
        raise AssertionError("ожидал хотя бы одну сохранённую картинку на странице 79")
    book_dir = tmp_dir / "images" / NON_DESIGNERS.stem
    if not book_dir.is_dir():
        raise AssertionError(f"не нашёл {book_dir} - картинки должны лежать в подпапке на книгу")
    for b in img_blocks:
        if not b["src"].startswith(f"images/{NON_DESIGNERS.stem}/"):
            raise AssertionError(
                f"src={b['src']!r} не содержит подпапку книги - "
                "проверь images_dir в convert_document/extract_image_block"
            )
        if not (tmp_dir / b["src"]).is_file():
            raise AssertionError(f"файл {b['src']!r} не найден на диске")


def check_image_namespacing():
    """Two different books converted into the SAME -o directory used to
    silently overwrite each other's pictures: extract_image_block() named
    files only by page_number/image_index inside one flat <output>/images/
    folder shared across every book, so e.g. book A's page-79-image-1 and
    book B's page-79-image-1 collided on disk and one book ended up
    displaying the other's picture.

    Standalone (not table-driven like CHECKS above) because it needs
    save_images=True and its own tmp_dir, unlike every other check here
    which shares one save_images=False run per book."""
    label = "[namespacing] картинки сохраняются в подпапку images/<книга>/, а не в общую images/"
    if not NON_DESIGNERS.exists():
        print(f"ПРОПУСК {label}: книга недоступна")
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="doc2html_imgtest_"))
    try:
        _assert_image_namespacing(tmp_dir)
        print(f"OK      {label}")
        return True
    except AssertionError as exc:
        print(f"ОШИБКА  {label}: {exc}")
        return False
    except Exception as exc:
        print(f"ОШИБКА  {label}: неожиданное исключение {type(exc).__name__}: {exc}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
            except Exception as exc:
                # A renamed/removed block key (KeyError etc.) after a schema
                # change is exactly the kind of regression this script exists
                # to catch - it must not silently abort the whole run and
                # skip every check after it.
                print(f"ОШИБКА  {label}: неожиданное исключение {type(exc).__name__}: {exc}")
                failed += 1

        ran = len(CHECKS) - skipped
        namespacing_result = check_image_namespacing()
        if namespacing_result is None:
            skipped += 1
        else:
            ran += 1
            if not namespacing_result:
                failed += 1

        print(f"\n{ran - failed}/{ran} проверок пройдено" + (f" ({skipped} пропущено - книга недоступна)" if skipped else ""))
        return 1 if failed else 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
