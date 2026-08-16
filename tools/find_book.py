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
  2. Topic/subject (Gutendex "bookshelf" match, free text, blank = skip)
  3. Author's birth/death year range (Gutendex doesn't track a book's own
     publication year - only the author's lifespan - so that's what this
     actually filters on; blank = skip)
  4. Is the author important? -> name/part of name to search for
  5. Do languages matter? -> pick from a list (arrow keys + space)
  6. Sort by... (popularity / catalog id ascending / catalog id descending)
  7. How many of the N matches do you want? (default: 1)
  8. Pick specific books from the results (arrow keys + space, top N
     pre-checked)
  9. Download the selection? Either way, prints the Gutenberg book page
     link for each pick so you can grab it by hand later.

Not part of the committed regression suite and not invoked by /maintain -
this is a convenience tool for building your own local input/ collection.
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import questionary
except ImportError:
    print(
        "Нужен пакет questionary: pip install questionary (или pip install -r requirements.txt)",
        file=sys.stderr,
    )
    raise SystemExit(1)

GUTENDEX_API_HOST = "gutendex.com"
GUTENDEX_BOOKS_URL = "https://gutendex.com/books"
ALLOWED_DOWNLOAD_HOSTS = {"www.gutenberg.org", "gutenberg.org"}
USER_AGENT = "doc2html-find-book/1.0 (+https://github.com/lopatnov/doc2html)"
MAX_RESPONSE_BYTES = 200 * 1024 * 1024
INPUT_DIR = Path(__file__).resolve().parent.parent / "input"

# Ordered by how well doc2html.py (via PyMuPDF) handles them.
PREFERRED_MIME_PREFIXES = ["application/epub+zip", "application/pdf", "text/html"]
EXTENSION_BY_MIME_PREFIX = {
    "application/epub+zip": ".epub",
    "application/pdf": ".pdf",
    "text/html": ".html",
}

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

SORT_CHOICES = [
    ("popular", "по популярности (число скачиваний)"),
    ("ascending", "по номеру в каталоге Gutenberg, по возрастанию"),
    ("descending", "по номеру в каталоге Gutenberg, по убыванию"),
]


# --- safe HTTP helpers (same allowlist/redirect/size-cap approach as
# tools/fetch_random_doc.py - this script also talks to the real internet
# from wherever you run it, so the same SSRF/size hygiene applies) ---


def _make_allowlisted_redirect_handler(allowed_hosts):
    class _Handler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            parsed = urllib.parse.urlparse(newurl)
            if parsed.scheme == "https" and parsed.hostname in allowed_hosts:
                return super().redirect_request(req, fp, code, msg, headers, newurl)
            raise urllib.error.HTTPError(newurl, code, f"refusing to follow redirect to {newurl!r}", headers, fp)

    return _Handler


_API_OPENER = urllib.request.build_opener(_make_allowlisted_redirect_handler({GUTENDEX_API_HOST}))
_DOWNLOAD_OPENER = urllib.request.build_opener(_make_allowlisted_redirect_handler(ALLOWED_DOWNLOAD_HOSTS))


def _read_capped(resp, max_bytes):
    chunks = []
    total = 0
    while total <= max_bytes:
        chunk = resp.read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > max_bytes:
        raise ValueError(f"response exceeded {max_bytes} byte cap")
    return b"".join(chunks)


def fetch_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _API_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(_read_capped(resp, MAX_RESPONSE_BYTES))


def is_allowed_download_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_DOWNLOAD_HOSTS and parsed.port is None


def download(url, dest, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _DOWNLOAD_OPENER.open(req, timeout=timeout) as resp:
        data = _read_capped(resp, MAX_RESPONSE_BYTES)
    temp_dest = dest.with_name(f".{dest.name}.part")
    try:
        with open(temp_dest, "wb") as f:
            f.write(data)
        temp_dest.replace(dest)
    finally:
        temp_dest.unlink(missing_ok=True)


def pick_format(formats):
    for mime_prefix in PREFERRED_MIME_PREFIXES:
        for key, url in formats.items():
            if key.startswith(mime_prefix):
                return mime_prefix, key, url
    return None, None, None


# --- small input helpers on top of questionary ---


def ask_int(message, default):
    while True:
        raw = questionary.text(f"{message} [{default}]:").ask()
        if raw is None:
            raise KeyboardInterrupt
        raw = raw.strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Введите целое число или просто нажмите Enter.")


def ask_optional_int(message):
    raw = questionary.text(f"{message} (Enter - пропустить):").ask()
    if raw is None:
        raise KeyboardInterrupt
    raw = raw.strip()
    return int(raw) if raw else None


# --- Gutendex query building ---


def build_query(topic, author, year_from, year_to, languages, sort, page=1):
    params = {"page": str(page)}
    search_terms = []
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


# --- wizard steps ---


def run_random_flow():
    count = ask_int("Сколько случайных книг скачать?", default=1)
    import secrets

    max_id = 75000
    picked = []
    tried = set()
    attempts = 0
    while len(picked) < count and attempts < count * 25:
        attempts += 1
        book_id = secrets.randbelow(max_id) + 1
        if book_id in tried:
            continue
        tried.add(book_id)
        try:
            book = fetch_json(f"{GUTENDEX_BOOKS_URL}/{book_id}")
        except Exception:
            continue
        if book.get("copyright") is not False:
            continue
        picked.append(book)
    return picked


def run_filtered_flow():
    topic = questionary.text("Тематика книг (Enter - пропустить):").ask()
    print("Годы жизни автора (Gutendex не хранит год издания самой книги, только годы жизни автора):")
    year_from = ask_optional_int("  с какого года")
    year_to = ask_optional_int("  по какой год")

    author = None
    if questionary.confirm("Важен автор?", default=False).ask():
        author = questionary.text("Имя/фамилия автора (часть имени):").ask()

    languages = None
    if questionary.confirm("Важны языки?", default=False).ask():
        chosen = questionary.checkbox(
            "Какие языки? (Пробел - отметить, Enter - продолжить)",
            choices=[questionary.Choice(title=f"{name} ({code})", value=code) for code, name in COMMON_LANGUAGES],
        ).ask()
        if chosen:
            languages = chosen
        else:
            extra = questionary.text("Ни один не подошёл? Введите коды через запятую (Enter - пропустить):").ask()
            if extra and extra.strip():
                languages = [c.strip() for c in extra.split(",") if c.strip()]

    sort = questionary.select(
        "Сортируем по...",
        choices=[questionary.Choice(title=label, value=value) for value, label in SORT_CHOICES],
        default=SORT_CHOICES[0][1],
    ).ask()

    url = build_query(topic, author, year_from, year_to, languages, sort)
    try:
        data = fetch_json(url)
    except Exception as exc:
        print(f"Не удалось обратиться к Gutendex: {exc}", file=sys.stderr)
        return []

    total = data.get("count", 0)
    if total == 0:
        print("По этим условиям ничего не нашлось - попробуйте ослабить фильтры.")
        return []

    want = ask_int(f"Сколько книг скачать из {total} найденных?", default=1)
    if want > 20:
        if not questionary.confirm(f"Точно скачать {want} книг?", default=False).ask():
            want = ask_int("Сколько тогда?", default=1)

    results = list(data.get("results", []))
    page = 2
    while len(results) < max(want, 25) and data.get("next"):
        try:
            data = fetch_json(build_query(topic, author, year_from, year_to, languages, sort, page=page))
        except Exception:
            break
        results.extend(data.get("results", []))
        page += 1

    shown = results[:25]
    default_picks = shown[: min(want, len(shown))]
    picked = questionary.checkbox(
        "Выберите книги (Пробел - отметить/снять, Enter - продолжить):",
        choices=[
            questionary.Choice(title=format_book_line(b), value=b, checked=(b in default_picks)) for b in shown
        ],
    ).ask()
    return picked or []


def download_selection(books):
    if not books:
        print("Ничего не выбрано.")
        return

    for book in books:
        print(f"\n{format_book_line(book)}")
        print(f"  Страница книги: {gutenberg_page_url(book['id'])}")

    if not questionary.confirm(f"Скачать выбранное ({len(books)}) в input/?", default=True).ask():
        print("Ок, ничего не скачиваю - ссылки выше можно открыть вручную.")
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for book in books:
        mime_prefix, format_key, url = pick_format(book.get("formats", {}))
        if not url or not is_allowed_download_url(url):
            print(f"  #{book['id']}: подходящего формата (epub/pdf/html) на gutenberg.org не нашлось, пропускаю")
            continue
        dest = INPUT_DIR / f"gutenberg_{book['id']}_{book.get('title', 'book')[:60]}{EXTENSION_BY_MIME_PREFIX[mime_prefix]}"
        dest = INPUT_DIR / "".join(c if c.isalnum() or c in " ._-" else "_" for c in dest.name)
        if dest.exists():
            print(f"  #{book['id']}: уже есть в input/ ({dest.name}), пропускаю")
            continue
        try:
            download(url, dest)
            print(f"  #{book['id']}: скачано -> input/{dest.name}")
        except Exception as exc:
            print(f"  #{book['id']}: скачать не удалось ({exc})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    print("Поиск книги на Project Gutenberg (через Gutendex). Enter - принять значение по умолчанию в [скобках].\n")
    try:
        is_random = questionary.confirm("Нужна случайная книга?", default=False).ask()
        if is_random is None:
            raise KeyboardInterrupt
        books = run_random_flow() if is_random else run_filtered_flow()
        download_selection(books)
    except KeyboardInterrupt:
        print("\nОтменено.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
