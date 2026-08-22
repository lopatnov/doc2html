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
import secrets
import sys
import urllib.error
from pathlib import Path

import gutenberg_http

GUTENDEX_BOOK_URL = "https://gutendex.com/books/{book_id}"
MAX_GUTENBERG_ID = 75000

# Gutendex's "formats" field hands back a download URL taken from its own
# catalog data, not something we should treat as trusted - restrict where we
# will actually fetch a file from to Gutenberg's own hosts (allowlist, not a
# blocklist) so a compromised/malicious API response can't redirect us into
# fetching an arbitrary URL (SSRF).
ALLOWED_DOWNLOAD_HOSTS = {"www.gutenberg.org", "gutenberg.org"}

_API_OPENER = gutenberg_http.build_opener({"gutendex.com"})
_DOWNLOAD_OPENER = gutenberg_http.build_opener(ALLOWED_DOWNLOAD_HOSTS)


def resolve_out_dir(raw_out_dir):
    """Resolve --out-dir and reject anything that escapes the current
    working directory (e.g. via ../.. segments or an absolute path
    elsewhere), since this value comes straight from an unvalidated CLI
    argument and every filesystem write in this script is derived from it.
    """
    base = Path.cwd().resolve()
    resolved = (base / raw_out_dir).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise SystemExit(f"--out-dir must resolve to a path inside {base} (got {resolved})")
    return resolved


def write_metadata(out_dir, book_id, attempt, data, format_key, url, dest):
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


def try_fetch_book(book_id, attempt, out_dir):
    """Try one candidate Gutenberg id. Returns True and writes the book +
    metadata.json into out_dir on success, False if this id should be
    skipped in favor of another random attempt.

    The `except Exception` below is deliberate, not a lint miss: this
    function tries up to --max-attempts random ids and must survive any
    single candidate misbehaving (network hiccup, malformed JSON, an
    unexpected field shape) without aborting the whole run - narrowing it
    to specific exception types would make one weird response fatal
    instead of just skipped.
    """
    try:
        data = gutenberg_http.fetch_json(_API_OPENER, GUTENDEX_BOOK_URL.format(book_id=book_id))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            print(f"attempt {attempt}: HTTP {exc.code} for id {book_id}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"attempt {attempt}: error {exc} for id {book_id}", file=sys.stderr)
        return False

    if data.get("copyright") is not False:
        # False = confirmed public domain in the US per Gutendex.
        # None (unknown) or True (copyrighted) are both too risky to use.
        return False

    mime_prefix, format_key, url = gutenberg_http.pick_format(data.get("formats", {}))
    if not url:
        return False
    url = gutenberg_http.normalize_download_url(url, ALLOWED_DOWNLOAD_HOSTS)
    if not gutenberg_http.is_allowed_download_url(url, ALLOWED_DOWNLOAD_HOSTS):
        print(f"attempt {attempt}: refusing non-Gutenberg download URL {url!r} for id {book_id}", file=sys.stderr)
        return False

    dest = out_dir / f"gutenberg_{book_id}{gutenberg_http.EXTENSION_BY_MIME_PREFIX[mime_prefix]}"
    try:
        gutenberg_http.atomic_download(_DOWNLOAD_OPENER, url, dest)
    except Exception as exc:
        print(f"attempt {attempt}: download failed for id {book_id}: {exc}", file=sys.stderr)
        return False

    write_metadata(out_dir, book_id, attempt, data, format_key, url, dest)
    print(f"OK: picked '{data.get('title')}' (Gutenberg id {book_id}) -> {dest}")
    return True


def positive_int_up_to_200(raw):
    value = int(raw)
    if not 1 <= value <= 200:
        raise argparse.ArgumentTypeError("--max-attempts must be between 1 and 200")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=positive_int_up_to_200, default=20)
    parser.add_argument("--out-dir", default="random_doc")
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tried_ids = set()
    attempt = 0
    while attempt < args.max_attempts:
        book_id = secrets.randbelow(MAX_GUTENBERG_ID) + 1
        if book_id in tried_ids:
            # Doesn't count as a real attempt - no API call was made for it.
            continue
        tried_ids.add(book_id)
        attempt += 1

        if try_fetch_book(book_id, attempt, out_dir):
            return 0

    print(f"FAILED: no usable public-domain document found in {args.max_attempts} attempts", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
