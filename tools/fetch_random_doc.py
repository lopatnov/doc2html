#!/usr/bin/env python3
"""Pick one random public-domain document from Project Gutenberg.

Runs inside the "Fetch Random Public-Domain Document" GitHub Actions
workflow (a GitHub-hosted runner has normal internet access, unlike the
cloud coding session that consumes its output - see
.claude/commands/maintain.md for why this exists). Not meant to be run
from the cloud session itself.

Queries the Gutendex API (https://gutendex.com), a third-party read-only
mirror of the Project Gutenberg catalog, for random book IDs until one is
found that Gutendex marks as public domain (copyright: false) and that
has a usable format (epub/pdf/html). Downloads it plus a metadata.json
recording the exact license basis, into --out-dir.

Exits 1 (no exception) if no usable book is found in --max-attempts
tries, so the workflow step fails cleanly and .claude/commands/maintain.md's
caller can treat this run as "no document found" rather than crash.
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

GUTENDEX_BOOK_URL = "https://gutendex.com/books/{book_id}"
MAX_GUTENBERG_ID = 75000
USER_AGENT = "doc2html-random-doc-fetcher/1.0 (+https://github.com/lopatnov/doc2html)"

# Ordered by how well doc2html.py (via PyMuPDF) handles them for QA purposes.
PREFERRED_MIME_PREFIXES = [
    "application/epub+zip",
    "application/pdf",
    "text/html",
]
EXTENSION_BY_MIME_PREFIX = {
    "application/epub+zip": ".epub",
    "application/pdf": ".pdf",
    "text/html": ".html",
}


def fetch_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def download(url, dest, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def pick_format(formats):
    for mime_prefix in PREFERRED_MIME_PREFIXES:
        for key, url in formats.items():
            if key.startswith(mime_prefix):
                return mime_prefix, key, url
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--out-dir", default="random_doc")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tried_ids = set()
    for attempt in range(1, args.max_attempts + 1):
        book_id = random.randint(1, MAX_GUTENBERG_ID)
        if book_id in tried_ids:
            continue
        tried_ids.add(book_id)

        try:
            data = fetch_json(GUTENDEX_BOOK_URL.format(book_id=book_id))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                print(f"attempt {attempt}: HTTP {exc.code} for id {book_id}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"attempt {attempt}: error {exc} for id {book_id}", file=sys.stderr)
            continue

        if data.get("copyright") is not False:
            # False = confirmed public domain in the US per Gutendex.
            # None (unknown) or True (copyrighted) are both too risky to use.
            continue

        formats = data.get("formats", {})
        mime_prefix, format_key, url = pick_format(formats)
        if not url:
            continue

        ext = EXTENSION_BY_MIME_PREFIX[mime_prefix]
        dest = out_dir / f"gutenberg_{book_id}{ext}"
        try:
            download(url, dest)
        except Exception as exc:
            print(f"attempt {attempt}: download failed for id {book_id}: {exc}", file=sys.stderr)
            continue

        metadata = {
            "source": "Project Gutenberg (via gutendex.com API)",
            "gutenberg_id": book_id,
            "gutenberg_book_url": f"https://www.gutenberg.org/ebooks/{book_id}",
            "title": data.get("title"),
            "authors": [a.get("name") for a in data.get("authors", [])],
            "license_basis": (
                "Gutendex reports copyright=false for this book (public domain "
                "in the US); the downloaded file is also covered by the Project "
                "Gutenberg License, see https://www.gutenberg.org/policy/license.html"
            ),
            "format_mime": format_key,
            "format_url": url,
            "local_file": dest.name,
            "attempt": attempt,
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"OK: picked '{data.get('title')}' (Gutenberg id {book_id}) -> {dest}")
        return 0

    print(f"FAILED: no usable public-domain document found in {args.max_attempts} attempts", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
