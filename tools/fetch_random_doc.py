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
import urllib.parse
import urllib.request
from pathlib import Path

GUTENDEX_BOOK_URL = "https://gutendex.com/books/{book_id}"
MAX_GUTENBERG_ID = 75000
USER_AGENT = "doc2html-random-doc-fetcher/1.0 (+https://github.com/lopatnov/doc2html)"

# Gutendex's "formats" field hands back a download URL taken from its own
# catalog data, not something we should treat as trusted - restrict where we
# will actually fetch a file from to Gutenberg's own hosts (allowlist, not a
# blocklist) so a compromised/malicious API response can't redirect us into
# fetching an arbitrary URL (SSRF).
ALLOWED_DOWNLOAD_HOSTS = {"www.gutenberg.org", "gutenberg.org"}
MAX_RESPONSE_BYTES = 200 * 1024 * 1024  # generous cap for even a large scanned-page epub/pdf


def is_allowed_download_url(url):
    parsed = urllib.parse.urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in ALLOWED_DOWNLOAD_HOSTS
        and parsed.port is None  # reject an explicit non-default port, e.g. gutenberg.org:8081
    )


def normalize_download_url(url):
    """Upgrade http -> https for a recognized Gutenberg host (Gutendex
    sometimes hands back http:// links). Never downgrades; anything else
    is returned unchanged and left for is_allowed_download_url to reject.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "http" and parsed.hostname in ALLOWED_DOWNLOAD_HOSTS:
        parsed = parsed._replace(scheme="https")
        return urllib.parse.urlunparse(parsed)
    return url


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urlopen() follows redirects by default without re-checking them
    against ALLOWED_DOWNLOAD_HOSTS, so a same-host response could still
    redirect us to an arbitrary URL (SSRF). Refuse every redirect instead -
    neither Gutendex's API nor Gutenberg's direct file URLs need one for
    our purposes, so treating any 3xx as a hard failure for that attempt
    is simpler and safer than re-validating each hop.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(newurl, code, f"refusing to follow redirect to {newurl!r}", headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirectHandler)

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


def _read_capped(resp, max_bytes):
    """Read at most max_bytes+1 from resp - the +1 lets the caller tell
    "exactly at the cap" apart from "went over" without reading unbounded
    data first. Guards against a malicious/broken response streaming an
    unbounded body at us (runner disk/memory exhaustion)."""
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
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(_read_capped(resp, MAX_RESPONSE_BYTES))


def download(url, dest, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _OPENER.open(req, timeout=timeout) as resp:
        data = _read_capped(resp, MAX_RESPONSE_BYTES)
    with open(dest, "wb") as f:
        f.write(data)


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


def pick_format(formats):
    for mime_prefix in PREFERRED_MIME_PREFIXES:
        for key, url in formats.items():
            if key.startswith(mime_prefix):
                return mime_prefix, key, url
    return None, None, None


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
    skipped in favor of another random attempt."""
    try:
        data = fetch_json(GUTENDEX_BOOK_URL.format(book_id=book_id))
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

    mime_prefix, format_key, url = pick_format(data.get("formats", {}))
    if not url:
        return False
    url = normalize_download_url(url)
    if not is_allowed_download_url(url):
        print(f"attempt {attempt}: refusing non-Gutenberg download URL {url!r} for id {book_id}", file=sys.stderr)
        return False

    dest = out_dir / f"gutenberg_{book_id}{EXTENSION_BY_MIME_PREFIX[mime_prefix]}"
    try:
        download(url, dest)
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
