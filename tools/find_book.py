#!/usr/bin/env python3
"""Interactive wizard for browsing Project Gutenberg and downloading books
into input/ (gitignored - this is for your own local library, nothing here
is meant to be committed).

Personal-use tool, deliberately separate from tools/fetch_random_doc.py
(which is a non-interactive script for the automated /maintain routine and
runs on a GitHub Actions runner, not locally). This one is meant to be run
by hand:

    pip install -r requirements.txt   # brings in questionary
    python tools/find_book.py

Walks through a small wizard (arrow keys where there's a choice to make,
Enter to accept the default shown in brackets):

  1. Random book instead of a filtered search? (default: no)
  2. Keyword (matched against title and author name - this is the one to
     use for "does Gutenberg have a book called X", blank = skip)
  3. Topic/subject (Gutendex "bookshelf"/subject match - a classification
     tag like "Fiction", NOT a title search; blank = skip)
  4. Author's birth/death year range (Gutendex doesn't track a book's own
     publication year - only the author's lifespan - so that's what this
     actually filters on; blank = skip)
  5. Is the author important? -> name/part of name to search for
  6. Do languages matter? -> pick from a list (arrow keys + space)
  7. Sort by... (popularity / newest-added-to-Gutenberg-first / oldest-first -
     Gutendex has no real publication-date field, so "newest" here means
     highest catalog id, i.e. most recently added to the archive)
  8. How many of the N matches do you want? (default: 1)
  9. Pick specific books from the results (arrow keys + space, top N
     pre-checked)
  10. Preferred file format (epub / pdf / html / any - falls back through
      epub -> pdf -> html per book if a pick doesn't have the preferred one)
  11. Download the selection? Either way, prints the Gutenberg book page
      link for each pick so you can grab it by hand later.

Not part of the committed regression suite and not invoked by /maintain -
this is a convenience tool for building your own local input/ collection.
"""
import argparse
import secrets
import sys
import time
import urllib.parse
from pathlib import Path

try:
    import questionary
except ImportError:
    print(
        "Нужен пакет questionary: pip install questionary (или pip install -r requirements.txt)",
        file=sys.stderr,
    )
    raise SystemExit(1)

import gutenberg_http

GUTENDEX_BOOKS_URL = "https://gutendex.com/books"
ALLOWED_DOWNLOAD_HOSTS = {"www.gutenberg.org", "gutenberg.org"}
MAX_GUTENBERG_ID = 75000
INPUT_DIR = Path(__file__).resolve().parent.parent / "input"

_API_OPENER = gutenberg_http.build_opener({"gutendex.com"})
_DOWNLOAD_OPENER = gutenberg_http.build_opener(ALLOWED_DOWNLOAD_HOSTS)

COMMON_LANGUAGES = [
    ("en", "английский"),
    ("ru", "русский"),
    ("uk", "українська"),
    ("de", "немецкий"),
    ("fr", "французский"),
    ("es", "испанский"),
    ("it", "итальянский"),
    ("pl", "польский"),
    ("pt", "португальский"),
    ("nl", "нидерландский"),
    ("sv", "шведский"),
    ("cs", "чешский"),
    ("ja", "японский"),
    ("zh", "китайский"),
]

# (value passed to Gutendex, label shown to the user) - value must be what's
# passed as questionary.Choice(value=...), never the label, or select()'s
# default matching raises ValueError (hit this for real on a live run).
SORT_CHOICES = [
    ("popular", "по популярности (число скачиваний)"),
    ("descending", "сначала новые (по номеру в каталоге Gutenberg, по убыванию)"),
    ("ascending", "сначала старые (по номеру в каталоге Gutenberg, по возрастанию)"),
]

# value is either "any" (fall back through the full PREFERRED_MIME_PREFIXES
# order) or a single mime prefix key into gutenberg_http.EXTENSION_BY_MIME_PREFIX.
FORMAT_CHOICES = [
    ("any", "любой (epub → pdf → html по очереди)"),
    ("application/epub+zip", "только epub"),
    ("application/pdf", "только pdf"),
    ("text/html", "только html (текст)"),
]


# --- small input helpers on top of questionary ---
#
# All prompts use unsafe_ask() rather than ask(): ask() swallows Ctrl-C and
# returns None, which these helpers would otherwise treat as "empty input"
# or a falsy confirm/select result instead of as a cancellation - unsafe_ask()
# raises KeyboardInterrupt instead, which main() actually catches.


def ask_int(message, default):
    while True:
        raw = questionary.text(f"{message} [{default}]:").unsafe_ask()
        raw = raw.strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Введите целое число или просто нажмите Enter.")


def ask_positive_int(message, default):
    while True:
        value = ask_int(message, default)
        if value >= 1:
            return value
        print("Число должно быть больше нуля.")


def ask_optional_int(message):
    while True:
        raw = questionary.text(f"{message} (Enter - пропустить):").unsafe_ask()
        raw = raw.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Введите целое число или просто нажмите Enter.")


# --- Gutendex query building ---


def build_query(topic, author, year_from, year_to, languages, sort, page=1, keyword=None):
    params = {"page": str(page)}
    search_terms = []
    if keyword:
        search_terms.append(keyword)
    if author:
        search_terms.append(author)
    if search_terms:
        params["search"] = " ".join(search_terms)
    if topic:
        params["topic"] = topic
    if year_from is not None:
        params["author_year_start"] = str(year_from)
    if year_to is not None:
        params["author_year_end"] = str(year_to)
    if languages:
        params["languages"] = ",".join(languages)
    if sort:
        params["sort"] = sort
    return f"{GUTENDEX_BOOKS_URL}?{urllib.parse.urlencode(params)}"


def format_book_line(book):
    authors = ", ".join(a.get("name", "?") for a in book.get("authors", [])) or "автор неизвестен"
    langs = ",".join(book.get("languages", []))
    downloads = book.get("download_count", 0)
    flag = "" if book.get("copyright") is False else " [лицензия не подтверждена как public domain]"
    return f"#{book['id']} — {book.get('title', '(без названия)')} — {authors} [{langs}] ({downloads} скачиваний){flag}"


def gutenberg_page_url(book_id):
    return f"https://www.gutenberg.org/ebooks/{book_id}"


def sanitized_filename(book, ext):
    """Build a safe input/ filename for a book. Sanitizes the title *before*
    combining it with the gutenberg_{id}_ prefix - doing it the other way
    around (build the path first, sanitize dest.name after) lets a title
    containing "/" split into path segments, so dest.name would come back
    as just the last segment and silently drop the id prefix, letting two
    different books collide on the same filename.
    """
    title = book.get("title") or "book"
    safe_title = "".join(c if c.isalnum() or c in " ._-" else "_" for c in title)[:60]
    return f"gutenberg_{book['id']}_{safe_title}{ext}"


# --- wizard steps ---


def run_random_flow():
    count = ask_positive_int("Сколько случайных книг скачать?", default=1)
    picked = []
    tried = set()
    attempts = 0
    last_error = None
    while len(picked) < count and attempts < count * 25:
        attempts += 1
        book_id = secrets.randbelow(MAX_GUTENBERG_ID) + 1
        if book_id in tried:
            continue
        tried.add(book_id)
        try:
            book = gutenberg_http.fetch_json(_API_OPENER, f"{GUTENDEX_BOOKS_URL}/{book_id}")
        except Exception as exc:  # a single bad id shouldn't stop the whole search
            last_error = exc
            continue
        if book.get("copyright") is not False:
            continue
        picked.append(book)
    if not picked and last_error is not None:
        print(f"Запросы к Gutendex не проходят: {last_error}", file=sys.stderr)
    return picked


def ask_author_filter():
    if not questionary.confirm("Важен автор?", default=False).unsafe_ask():
        return None
    return questionary.text("Имя/фамилия автора (часть имени):").unsafe_ask()


def ask_language_filter():
    if not questionary.confirm("Важны языки?", default=False).unsafe_ask():
        return None
    chosen = questionary.checkbox(
        "Какие языки? (Пробел - отметить, Enter - продолжить)",
        choices=[questionary.Choice(title=f"{name} ({code})", value=code) for code, name in COMMON_LANGUAGES],
    ).unsafe_ask()
    if chosen:
        return chosen
    extra = questionary.text("Ни один не подошёл? Введите коды через запятую (Enter - пропустить):").unsafe_ask()
    return [c.strip() for c in extra.split(",") if c.strip()] if extra and extra.strip() else None


def ask_sort_choice():
    return questionary.select(
        "Сортируем по...",
        choices=[questionary.Choice(title=label, value=value) for value, label in SORT_CHOICES],
        default=SORT_CHOICES[0][0],
    ).unsafe_ask()


def ask_format_choice():
    return questionary.select(
        "Предпочитаемый формат файла?",
        choices=[questionary.Choice(title=label, value=value) for value, label in FORMAT_CHOICES],
        default=FORMAT_CHOICES[0][0],
    ).unsafe_ask()


def fetch_search_page(url, max_attempts=3):
    """fetch_json with a couple of retries on transient failures (timeouts,
    connection resets) - a live run against Gutendex hit exactly this: one
    read timeout on the very first request killed the whole search even
    though a retry a moment later would very likely have gone through."""
    last_exc = None
    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(1.5 * attempt)
        try:
            return gutenberg_http.fetch_json(_API_OPENER, url)
        except Exception as exc:  # network hiccups vary in type; any of them is worth a retry
            last_exc = exc
    raise last_exc


def collect_search_results(topic, author, year_from, year_to, languages, sort, want, keyword=None):
    """Fetch pages from Gutendex until there are at least `want` (or 25,
    whichever is more) results to show, or the API runs out of pages."""
    results = []
    count = 0
    page = 1
    while len(results) < max(want, 25):
        try:
            data = fetch_search_page(
                build_query(topic, author, year_from, year_to, languages, sort, page=page, keyword=keyword)
            )
        except Exception as exc:
            if page == 1:
                print(f"Не удалось обратиться к Gutendex: {exc}", file=sys.stderr)
            break
        count = data.get("count", len(results))
        if page == 1 and count == 0:
            return 0, []
        results.extend(data.get("results", []))
        if not data.get("next"):
            break
        page += 1
    return count, results


def run_filtered_flow():
    keyword = questionary.text(
        "Ключевое слово (ищет по названию и автору, Enter - пропустить):"
    ).unsafe_ask()
    topic = questionary.text("Тематика книг (Enter - пропустить):").unsafe_ask()
    print("Годы жизни автора (Gutendex не хранит год издания самой книги, только годы жизни автора):")
    year_from = ask_optional_int("  с какого года")
    year_to = ask_optional_int("  по какой год")
    author = ask_author_filter()
    languages = ask_language_filter()
    sort = ask_sort_choice()

    # A cheap first request just to learn the total count before asking
    # "how many of N" - collect_search_results does the real pagination.
    total, _ = collect_search_results(topic, author, year_from, year_to, languages, sort, want=1, keyword=keyword)
    if total == 0:
        print("По этим условиям ничего не нашлось - попробуйте ослабить фильтры.")
        return []

    want = ask_positive_int(f"Сколько книг скачать из {total} найденных?", default=1)
    if want > 20 and not questionary.confirm(f"Точно скачать {want} книг?", default=False).unsafe_ask():
        want = ask_positive_int("Сколько тогда?", default=1)

    _, results = collect_search_results(topic, author, year_from, year_to, languages, sort, want, keyword=keyword)
    shown = results[:25]
    default_picks = shown[: min(want, len(shown))]
    picked = questionary.checkbox(
        "Выберите книги (Пробел - отметить/снять, Enter - продолжить):",
        choices=[
            questionary.Choice(title=format_book_line(b), value=b, checked=(b in default_picks)) for b in shown
        ],
    ).unsafe_ask()
    return picked or []


def download_selection(books, format_choice="any"):
    if not books:
        print("Ничего не выбрано.")
        return

    for book in books:
        print(f"\n{format_book_line(book)}")
        print(f"  Страница книги: {gutenberg_page_url(book['id'])}")

    if not questionary.confirm(f"Скачать выбранное ({len(books)}) в input/?", default=True).unsafe_ask():
        print("Ок, ничего не скачиваю - ссылки выше можно открыть вручную.")
        return

    preferred_prefixes = None if format_choice == "any" else [format_choice]
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for book in books:
        mime_prefix, _format_key, url = gutenberg_http.pick_format(book.get("formats", {}), preferred_prefixes)
        if url:
            url = gutenberg_http.normalize_download_url(url, ALLOWED_DOWNLOAD_HOSTS)
        if not url or not gutenberg_http.is_allowed_download_url(url, ALLOWED_DOWNLOAD_HOSTS):
            wanted = "epub/pdf/html" if format_choice == "any" else format_choice
            print(f"  #{book['id']}: подходящего формата ({wanted}) на gutenberg.org не нашлось, пропускаю")
            continue
        ext = gutenberg_http.EXTENSION_BY_MIME_PREFIX[mime_prefix]
        dest = INPUT_DIR / sanitized_filename(book, ext)
        if dest.exists():
            print(f"  #{book['id']}: уже есть в input/ ({dest.name}), пропускаю")
            continue
        try:
            gutenberg_http.atomic_download(_DOWNLOAD_OPENER, url, dest)
            print(f"  #{book['id']}: скачано -> input/{dest.name}")
        except Exception as exc:
            print(f"  #{book['id']}: скачать не удалось ({exc})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    print("Поиск книги на Project Gutenberg (через Gutendex). Enter - принять значение по умолчанию в [скобках].\n")
    try:
        is_random = questionary.confirm("Нужна случайная книга?", default=False).unsafe_ask()
        books = run_random_flow() if is_random else run_filtered_flow()
        format_choice = ask_format_choice() if books else "any"
        download_selection(books, format_choice)
    except (KeyboardInterrupt, EOFError):
        print("\nОтменено.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
