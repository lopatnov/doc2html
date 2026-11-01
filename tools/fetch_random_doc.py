#!/usr/bin/env python3
"""Pick one random public-domain document from Project Gutenberg.

Runs inside the "Fetch Random Public-Domain Document" GitHub Actions
workflow (a GitHub-hosted runner has normal internet access, unlike the
cloud coding session that consumes its output - see
.claude/commands/maintain.md for why this exists). Not meant to be run
from the cloud session itself.

Random book ids are looked up until one is found that is confirmed public
domain and has a usable format (epub/pdf/html); the file plus a
metadata.json recording the exact license basis are written into --out-dir.
Two independent catalogs can answer the lookup:

  gutendex - the Gutendex API (https://gutendex.com), a third-party
             read-only mirror of the Project Gutenberg catalog; public
             domain means "copyright: false".
  rdf      - Project Gutenberg's own catalog record for the book
             (https://www.gutenberg.org/cache/epub/<id>/pg<id>.rdf); public
             domain means its dcterms:rights says "Public domain in the USA.".

Gutendex is tried first. It has gone dark for weeks at a time (every
request timing out, while gutenberg.org itself answered in milliseconds),
so a failed Gutendex lookup falls back to the RDF record for the same id,
and after a few failures in a row Gutendex is skipped for the rest of the
run. --source pins one of them.

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
from xml.etree import ElementTree

import gutenberg_http

GUTENDEX_BOOK_URL = "https://gutendex.com/books/{book_id}"
GUTENBERG_RDF_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.rdf"
MAX_GUTENBERG_ID = 75000

# Gutendex's "formats" field hands back a download URL taken from its own
# catalog data, not something we should treat as trusted - restrict where we
# will actually fetch a file from to Gutenberg's own hosts (allowlist, not a
# blocklist) so a compromised/malicious API response can't redirect us into
# fetching an arbitrary URL (SSRF).
ALLOWED_DOWNLOAD_HOSTS = {"www.gutenberg.org", "gutenberg.org"}

# GitHub hard-rejects any single pushed file over 100 MiB (confirmed on a
# real run: a 133.43 MB epub was fetched fine but "Publish to a run-specific
# inbox branch" failed with "GH001: Large files detected", since the file
# is committed and pushed via plain git, not Git LFS). Capped well under
# that so an oversized candidate fails inside try_fetch_book's own
# except-and-skip, the same as any other bad candidate, instead of reaching
# git push and failing the whole run.
MAX_PUBLISHABLE_FILE_BYTES = 95 * 1024 * 1024

# A catalog record is a few KB; anything far beyond that is not one.
MAX_RDF_BYTES = 2 * 1024 * 1024
# Consecutive failed Gutendex lookups after which it is skipped for the rest
# of the run (each failure costs a full request timeout).
GUTENDEX_MAX_CONSECUTIVE_FAILURES = 3

# XML namespace identifiers of the catalog record, in ElementTree's {uri}tag
# notation. They must match the document byte for byte and are never fetched.
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
DCTERMS = "{http://purl.org/dc/terms/}"
PGTERMS = "{http://www.gutenberg.org/2009/pgterms/}"

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
        raise SystemExit(f"--out-dir must resolve to a path inside {base} (got {resolved})") from None
    return resolved


def lookup_gutendex(book_id):
    """A candidate dict (see try_fetch_book), or None if this id should be
    skipped (no such book, or not confirmed public domain). Raises on a
    network/API failure."""
    try:
        data = gutenberg_http.fetch_json(_API_OPENER, GUTENDEX_BOOK_URL.format(book_id=book_id))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if data.get("copyright") is not False:
        # False = confirmed public domain in the US per Gutendex.
        # None (unknown) or True (copyrighted) are both too risky to use.
        return None
    return {
        "source": "Project Gutenberg (via gutendex.com API)",
        "title": data.get("title"),
        "authors": [a.get("name") for a in data.get("authors", [])],
        "formats": data.get("formats", {}),
        "license_basis": (
            "Gutendex reports copyright=false for this book (public domain "
            "in the US); the downloaded file is also covered by the Project "
            "Gutenberg License, see https://www.gutenberg.org/policy/license.html"
        ),
    }


def parse_gutenberg_rdf(raw, book_id):
    """A candidate dict from a Project Gutenberg catalog record, or None if
    the record doesn't say public domain. Raises ValueError if it has no
    pgterms:ebook element at all - a format change should be visible in the
    log, not silently skip every id."""
    root = ElementTree.fromstring(raw)
    ebook = root.find(f"{PGTERMS}ebook")
    if ebook is None:
        raise ValueError("catalog record has no pgterms:ebook element")
    rights = ebook.findtext(f"{DCTERMS}rights", default="").strip()
    if not rights.lower().startswith("public domain"):
        return None

    # One book usually has several files per mime type (epub3 / epub with
    # images / epub without images): keep the most complete one.
    best = {}
    for file_el in ebook.iterfind(f"{DCTERMS}hasFormat/{PGTERMS}file"):
        url = file_el.get(f"{RDF}about")
        if not url:
            continue
        rank = (".images" in url, "epub3" in url)
        for value in file_el.iterfind(f"{DCTERMS}format/{RDF}Description/{RDF}value"):
            mime = (value.text or "").strip()
            if mime and (mime not in best or rank > best[mime][0]):
                best[mime] = (rank, url)

    authors = [
        (name.text or "").strip()
        for name in ebook.iterfind(f"{DCTERMS}creator/{PGTERMS}agent/{PGTERMS}name")
    ]
    return {
        "source": "Project Gutenberg (catalog record on gutenberg.org)",
        "title": ebook.findtext(f"{DCTERMS}title"),
        "authors": [a for a in authors if a],
        "formats": {mime: url for mime, (_rank, url) in best.items()},
        "license_basis": (
            f"The Project Gutenberg catalog record for ebook {book_id} states its rights as "
            f"'{rights}'; the downloaded file is also covered by the Project Gutenberg "
            "License, see https://www.gutenberg.org/policy/license.html"
        ),
    }


def lookup_gutenberg_rdf(book_id):
    """Same contract as lookup_gutendex, answered by gutenberg.org itself."""
    try:
        raw = gutenberg_http.fetch_bytes(
            _DOWNLOAD_OPENER, GUTENBERG_RDF_URL.format(book_id=book_id), max_bytes=MAX_RDF_BYTES,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    return parse_gutenberg_rdf(raw, book_id)


class LookupSources:
    """Which catalog(s) to ask for a book id, in order, and when to give up
    on Gutendex for the rest of the run."""

    def __init__(self, mode="auto", gutendex=lookup_gutendex, rdf=lookup_gutenberg_rdf):
        self.mode = mode
        self._lookups = {"gutendex": gutendex, "rdf": rdf}
        self.gutendex_failures = 0
        self.gutendex_skipped = False

    def candidates(self):
        """(name, lookup) pairs to try for the next id."""
        if self.mode == "gutendex":
            return [("gutendex", self._lookups["gutendex"])]
        if self.mode == "rdf" or self.gutendex_skipped:
            return [("rdf", self._lookups["rdf"])]
        return [("gutendex", self._lookups["gutendex"]), ("rdf", self._lookups["rdf"])]

    def record(self, name, ok):
        if name != "gutendex":
            return
        if ok:
            self.gutendex_failures = 0
            return
        self.gutendex_failures += 1
        if self.mode == "auto" and self.gutendex_failures >= GUTENDEX_MAX_CONSECUTIVE_FAILURES \
                and not self.gutendex_skipped:
            self.gutendex_skipped = True
            print(
                f"Gutendex failed {self.gutendex_failures} times in a row - "
                "using the gutenberg.org catalog only for the rest of this run",
                file=sys.stderr,
            )


def find_candidate(book_id, attempt, sources):
    """Ask the catalogs for book_id. A lookup that answers (even "skip this
    id") ends the search; one that raises counts as the source being down and
    the next one is tried."""
    for name, lookup in sources.candidates():
        try:
            candidate = lookup(book_id)
        except Exception as exc:
            print(f"attempt {attempt}: {name} lookup failed for id {book_id}: {exc}", file=sys.stderr)
            sources.record(name, ok=False)
            continue
        sources.record(name, ok=True)
        return candidate
    return None


def write_metadata(out_dir, book_id, attempt, candidate, format_key, url, dest):
    metadata = {
        "source": candidate["source"],
        "gutenberg_id": book_id,
        "gutenberg_book_url": f"https://www.gutenberg.org/ebooks/{book_id}",
        "title": candidate["title"],
        "authors": candidate["authors"],
        "license_basis": candidate["license_basis"],
        "format_mime": format_key,
        "format_url": url,
        "local_file": dest.name,
        "attempt": attempt,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def try_fetch_book(book_id, attempt, out_dir, sources=None):
    """Try one candidate Gutenberg id. Returns True and writes the book +
    metadata.json into out_dir on success, False if this id should be
    skipped in favor of another random attempt.

    The `except Exception` around the download below is deliberate, not a
    lint miss: this function tries up to --max-attempts random ids and must
    survive any single candidate misbehaving (network hiccup, oversized
    file, an unexpected field shape) without aborting the whole run -
    narrowing it to specific exception types would make one weird response
    fatal instead of just skipped.
    """
    candidate = find_candidate(book_id, attempt, sources or LookupSources())
    if candidate is None:
        return False

    mime_prefix, format_key, url = gutenberg_http.pick_format(candidate["formats"])
    if not url:
        return False
    url = gutenberg_http.normalize_download_url(url, ALLOWED_DOWNLOAD_HOSTS)
    if not gutenberg_http.is_allowed_download_url(url, ALLOWED_DOWNLOAD_HOSTS):
        print(f"attempt {attempt}: refusing non-Gutenberg download URL {url!r} for id {book_id}", file=sys.stderr)
        return False

    dest = out_dir / f"gutenberg_{book_id}{gutenberg_http.EXTENSION_BY_MIME_PREFIX[mime_prefix]}"
    try:
        gutenberg_http.atomic_download(_DOWNLOAD_OPENER, url, dest, max_bytes=MAX_PUBLISHABLE_FILE_BYTES)
    except Exception as exc:
        print(f"attempt {attempt}: download failed for id {book_id}: {exc}", file=sys.stderr)
        return False

    write_metadata(out_dir, book_id, attempt, candidate, format_key, url, dest)
    print(f"OK: picked '{candidate['title']}' (Gutenberg id {book_id}, via {candidate['source']}) -> {dest}")
    return True


def positive_int_up_to_200(raw):
    value = int(raw)
    if not 1 <= value <= 200:
        raise argparse.ArgumentTypeError("--max-attempts must be between 1 and 200")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-attempts", type=positive_int_up_to_200, default=20)
    parser.add_argument("--out-dir", default="random_doc")
    parser.add_argument(
        "--source", choices=["auto", "gutendex", "rdf"], default="auto",
        help="catalog to ask: auto (Gutendex, falling back to the gutenberg.org record), or just one of them",
    )
    args = parser.parse_args(argv)

    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = LookupSources(args.source)
    tried_ids = set()
    attempt = 0
    while attempt < args.max_attempts:
        book_id = secrets.randbelow(MAX_GUTENBERG_ID) + 1
        if book_id in tried_ids:
            # Doesn't count as a real attempt - no API call was made for it.
            continue
        tried_ids.add(book_id)
        attempt += 1

        if try_fetch_book(book_id, attempt, out_dir, sources):
            return 0

    print(f"FAILED: no usable public-domain document found in {args.max_attempts} attempts", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
